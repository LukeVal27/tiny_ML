/*
 * Tier 1 deployment: 5-class food classifier + 3-tier portion head.
 * Arduino Nano 33 BLE Sense (nRF52840) + OV7675 on the Tiny ML shield.
 *
 * Two modes, chosen at build time:
 *   BENCH_MODE 1  run inference on the baked-in test image, print INFER
 *                 telemetry, no camera needed. This is the unattended loop.
 *   BENCH_MODE 0  live camera: capture -> downsample -> infer -> print result.
 *
 * Serial commands (live mode): 'c' captures and classifies one frame.
 *
 * IMPORTANT -- why this downsamples instead of cropping:
 * The stock person_detection example centre-crops 96x96 out of the QCIF frame,
 * which throws away most of the field of view. We cannot do that. The portion
 * head measures food area *relative to the plate rim*, so the whole plate has
 * to stay in frame. We centre-crop to square (176x144 -> 144x144) and then
 * area-average down to 96x96, preserving the full plate.
 */

#include <TinyMLShield.h>

#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "model_data.h"
#include "test_image.h"

// Measured on-device: this model reports arena_used_bytes() = 113,240, so the
// naive "largest pair of live tensors" estimate (72 KB) badly under-predicts what
// TFLM's memory planner actually reserves. 120 KB leaves a small margin without
// crowding out the 50,688-byte RGB565 frame buffer in live-camera mode.
//
// Budget at 120 KB arena + live camera:
//   overhead ~51,792 + frame 50,688 + arena 122,880 = 225,360 of 262,144 SRAM.
#ifndef ARENA_SIZE
#define ARENA_SIZE 120000
#endif

#ifndef BENCH_MODE
#define BENCH_MODE 1
#endif

// PROBE_MODE 1 overrides everything else: bring up the camera, switch it to its
// built-in test pattern, and print the RAW byte pairs for a row of sampled
// pixels. No model, no inference.
//
// This exists to settle the RGB565 byte order. Arduino_OV767X::readFrame() does
// no byte swapping -- it stores bytes in sensor-clock order -- and the
// OV7670/OV7675 is documented as emitting "RGB565-swapped" relative to the
// (hi<<8)|lo assumption in unpack565(). Rather than guess, we ship the raw bytes
// to the host and let it decode them both ways against the known colour-bar
// order. Sampling ~64 pixels keeps this to one short serial line instead of
// dumping all 50,688 frame bytes.
#ifndef PROBE_MODE
#define PROBE_MODE 0
#endif

// Camera frame rate. Supported by the driver: 1, 5, 10, 15, 30.
//
// This is the ONLY remaining lever on preview speed. Measured: readFrame() costs
// ~330 ms of fixed GPIO bit-banging (50,688 bytes, no DMA) plus a wait for the
// next VSYNC. At 5 fps that wait is 0-200 ms; at 15 fps it is 0-67 ms. Shrinking
// the serial payload 4x changed nothing (66 -> 16 KB/s, 2.45 -> 2.30 fps), which
// proved transfer was never the constraint.
//
// Trade-off: a higher frame rate shortens sensor integration time, so images can
// be darker or noisier in poor light. Keep 5 for classification runs; 15 is for
// framing, where speed matters and image fidelity does not.
#ifndef CAM_FPS
#define CAM_FPS 5
#endif

// ------------------------------------------------------------ camera layout
#define CAM_W 176
#define CAM_H 144
#define CAM_BPP 2 // RGB565
#define FRAME_BYTES (CAM_W * CAM_H * CAM_BPP)

#define IN_W 96
#define IN_H 96
#define IN_C 3

// Square centre region of the QCIF frame that we downsample from.
#define SQ 144
#define SQ_X0 ((CAM_W - SQ) / 2) // 16
#define SQ_Y0 ((CAM_H - SQ) / 2) // 0

static const char *kClasses[] = {"chicken", "broccoli", "rice", "beef", "potato"};
static const char *kPortions[] = {"small", "medium", "large"};
static const int kNumClasses = 5;
static const int kNumPortions = 3;

// Portion tier -> macro range, mapped from the mass thresholds in
// data/compose_portions.py (small <80 g, medium 80-180 g, large >180 g).
static const char *kMassRange[] = {"<80g", "80-180g", ">180g"};

// ---------------------------------------------------------------- buffers
#if BENCH_MODE == 0 || PROBE_MODE == 1
__attribute__((aligned(16))) static uint8_t frame_buffer[FRAME_BYTES];
#endif
__attribute__((aligned(16))) static uint8_t tensor_arena[ARENA_SIZE];

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter *error_reporter = &micro_error_reporter;
tflite::AllOpsResolver resolver;

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *out_cls = nullptr;
TfLiteTensor *out_portion = nullptr;

// See deploy/nano/smoke_test/smoke_test.ino for why the sbrk() trick reports
// garbage under mbed OS. We binary-search the largest allocatable block instead.
static int largestFreeBlock() {
  size_t lo = 0, hi = 200u * 1024u;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo + 1) / 2;
    void *p = malloc(mid);
    if (p != nullptr) { free(p); lo = mid; } else { hi = mid - 1; }
  }
  return (int)lo;
}

// Bind the two output tensors by width rather than by index: the converter does
// not promise which head lands at output(0), and silently swapping them would
// produce plausible-looking nonsense.
static void bindOutputs() {
  out_cls = out_portion = nullptr;
  for (size_t i = 0; i < interpreter->outputs_size(); i++) {
    TfLiteTensor *t = interpreter->output(i);
    const int n = t->dims->data[t->dims->size - 1];
    if (n == kNumClasses) out_cls = t;
    else if (n == kNumPortions) out_portion = t;
  }
}

#if BENCH_MODE == 0
// RGB565 (big-endian on the wire from the OV767X driver) -> 8-bit channels.
static inline void unpack565(uint8_t hi, uint8_t lo, int &r, int &g, int &b) {
  const uint16_t px = (uint16_t(hi) << 8) | lo;
  r = ((px >> 11) & 0x1F) * 255 / 31;
  g = ((px >> 5) & 0x3F) * 255 / 63;
  b = (px & 0x1F) * 255 / 31;
}

/*
 * 144x144 -> 96x96 is an exact 3:2 reduction, so each output pixel is the mean
 * of a 3x3/2x2 footprint. We approximate with a fixed 2x2 box average taken at
 * the 1.5x sample point, which removes most of the aliasing for a fraction of
 * the cost of a true area filter -- and aliasing matters here because rice and
 * broccoli are distinguished largely by texture.
 */
static void downsampleToInput(int8_t *dst) {
  const float scale = input->params.scale;
  const int zp = input->params.zero_point;

  for (int oy = 0; oy < IN_H; oy++) {
    const int sy = SQ_Y0 + (oy * 3) / 2;
    for (int ox = 0; ox < IN_W; ox++) {
      const int sx = SQ_X0 + (ox * 3) / 2;

      int rs = 0, gs = 0, bs = 0;
      for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
          const int px = min(sx + dx, CAM_W - 1);
          const int py = min(sy + dy, CAM_H - 1);
          const int idx = (py * CAM_W + px) * CAM_BPP;
          int r, g, b;
          unpack565(frame_buffer[idx], frame_buffer[idx + 1], r, g, b);
          rs += r; gs += g; bs += b;
        }
      }

      // The model was trained on float inputs in [0,1], so divide by 255 before
      // applying the tensor's own quantisation parameters.
      const int o = (oy * IN_W + ox) * IN_C;
      const float v[3] = {rs / 4.0f / 255.0f, gs / 4.0f / 255.0f,
                          bs / 4.0f / 255.0f};
      for (int c = 0; c < 3; c++) {
        int q = (int)lroundf(v[c] / scale) + zp;
        dst[o + c] = (int8_t)constrain(q, -128, 127);
      }
    }
  }
}
#endif

#if BENCH_MODE == 0
// ---------------------------------------------------------------- status LED
// The Tiny ML shield claims pin 13 for its button (see TinyMLShield.h, which
// drives BUTTON_PIN as an output held HIGH), so LED_BUILTIN is not usable here.
// The Nano 33 BLE's separate RGB LED is free. All three channels are ACTIVE LOW.
static void ledInit() {
  pinMode(LEDR, OUTPUT); pinMode(LEDG, OUTPUT); pinMode(LEDB, OUTPUT);
  digitalWrite(LEDR, HIGH); digitalWrite(LEDG, HIGH); digitalWrite(LEDB, HIGH);
}
static inline void ledBlue(bool on) { digitalWrite(LEDB, on ? LOW : HIGH); }
static inline void ledGreen(bool on) { digitalWrite(LEDG, on ? LOW : HIGH); }

// Streaming state. Toggled by 'v' over serial.
static bool g_streaming = false;
static bool g_led_phase = false;

/*
 * Send the model's input tensor as a preview frame.
 *
 * MUST be called BEFORE Invoke(). The input tensor lives inside the tensor
 * arena and TFLM aliases that memory with operator scratch, so Invoke()
 * overwrites its own input -- streaming afterwards would transmit activations,
 * not the picture.
 *
 * Costs no extra RAM: this is the same 96x96x3 buffer downsampleToInput() just
 * filled. Values are int8 with zero_point -128 and scale 1/255, so the host
 * recovers uint8 with a single +128.
 */
static void streamPreview() {
  Serial.write((const uint8_t *)"FRM:", 4);
  Serial.write((const uint8_t *)input->data.int8, input->bytes);
}

/*
 * Half-resolution preview: 48x48x3 instead of 96x96x3.
 *
 * Framing only needs to show where the plate rim sits, and that survives a 2x
 * reduction easily. The payoff is 4x less serial data (6,912 vs 27,648 bytes),
 * which matters because transfer was measured at ~66 KB/s -- about 420 ms per
 * full frame, comparable to the capture itself.
 *
 * Emitted row by row from a 144-byte stack buffer, so this costs no static RAM.
 * Nearest-neighbour (every other pixel) is deliberate: it is the cheapest
 * possible reduction and preview fidelity is not the point.
 */
static void streamPreviewHalf() {
  const int8_t *src = input->data.int8;
  uint8_t row[48 * 3];
  Serial.write((const uint8_t *)"FRH:", 4);
  for (int y = 0; y < 96; y += 2) {
    int o = 0;
    for (int x = 0; x < 96; x += 2) {
      const int i = (y * 96 + x) * 3;
      row[o++] = (uint8_t)(src[i + 0] + 128);
      row[o++] = (uint8_t)(src[i + 1] + 128);
      row[o++] = (uint8_t)(src[i + 2] + 128);
    }
    Serial.write(row, sizeof(row));
  }
}
#endif

static void printResult(unsigned long infer_us) {
  int bc = 0, bp = 0;
  for (int i = 1; i < kNumClasses; i++)
    if (out_cls->data.int8[i] > out_cls->data.int8[bc]) bc = i;
  for (int i = 1; i < kNumPortions; i++)
    if (out_portion->data.int8[i] > out_portion->data.int8[bp]) bp = i;

  const float cconf = (out_cls->data.int8[bc] - out_cls->params.zero_point) *
                      out_cls->params.scale;
  const float pconf =
      (out_portion->data.int8[bp] - out_portion->params.zero_point) *
      out_portion->params.scale;

  Serial.print("INFER,cls=");
  Serial.print(kClasses[bc]);
  Serial.print(",cls_idx=");
  Serial.print(bc);
  Serial.print(",cls_conf=");
  Serial.print(cconf, 4);
  Serial.print(",portion=");
  Serial.print(kPortions[bp]);
  Serial.print(",portion_idx=");
  Serial.print(bp);
  Serial.print(",portion_conf=");
  Serial.print(pconf, 4);
  Serial.print(",mass_range=");
  Serial.print(kMassRange[bp]);
  Serial.print(",latency_us=");
  Serial.print(infer_us);
  Serial.print(",arena_used=");
  Serial.print(interpreter->arena_used_bytes());
  Serial.print(",free_sram=");
  Serial.println(largestFreeBlock());
}


#if BENCH_MODE == 1
// Re-emitted periodically from loop(), not just once in setup(): the harness
// opens the serial port a moment AFTER the board resets and starts running, so
// a single setup()-time print is routinely missed and the run looks silent.
// The input tensor lives INSIDE the tensor arena, and TFLM's memory planner
// happily aliases that buffer with scratch space once the first operator has
// consumed it. So Invoke() destroys its own input.
//
// This bit us badly: the bench used to copy the image once and then Invoke 21
// times, meaning runs 2..21 classified whatever activations happened to be left
// in that memory. The device disagreed with the host on the class head, and
// three wrong hypotheses (op support, TFLM version, XNNPACK delegate) were
// chased before a checksum of the input tensor showed input_sum = -1,081,352
// against a baked image summing to -126,959.
//
// Refill before EVERY Invoke.
static void fillInput() {
  const int n = (int)input->bytes;
  for (int i = 0; i < n && i < (int)g_test_image_len; i++)
    input->data.int8[i] = g_test_image[i];
}

static void runBench() {
  fillInput();

  // Warm up once; the first Invoke() pays one-off setup costs that would
  // otherwise be reported as inference latency.
  interpreter->Invoke();

  const int kRuns = 20;
  unsigned long total = 0, worst = 0;
  for (int i = 0; i < kRuns; i++) {
    // Refill outside the timed region so the reported latency stays pure
    // inference and never includes the memcpy.
    fillInput();
    const unsigned long s = micros();
    interpreter->Invoke();
    const unsigned long d = micros() - s;
    total += d;
    if (d > worst) worst = d;
  }

  int bc = 0, bp = 0;
  for (int i = 1; i < kNumClasses; i++)
    if (out_cls->data.int8[i] > out_cls->data.int8[bc]) bc = i;
  for (int i = 1; i < kNumPortions; i++)
    if (out_portion->data.int8[i] > out_portion->data.int8[bp]) bp = i;

  const bool ok = (bc == g_test_image_expected_cls) &&
                  (bp == g_test_image_expected_portion);

  // Byte-level checksums of exactly what this board is running. Host and device
  // disagree on the class head while agreeing on the portion head; TFLM version
  // and XNNPACK are already ruled out, so the remaining question is whether the
  // two sides are even operating on identical bytes. Summing is enough to catch
  // a stale flash, a truncated array or a signedness error.
  long img_sum = 0;
  for (unsigned i = 0; i < g_test_image_len; i++) img_sum += g_test_image[i];
  long mdl_sum = 0;
  for (unsigned i = 0; i < g_model_data_len; i++) mdl_sum += g_model_data[i];
  // Sum the input tensor from a FRESH fill. Sampling it after Invoke() reads
  // arena scratch, not the image -- which is exactly the bug this block caught.
  fillInput();
  long in_sum = 0;
  for (unsigned i = 0; i < input->bytes; i++) in_sum += input->data.int8[i];

  Serial.print("CHECK,img_sum=");          Serial.print(img_sum);
  Serial.print(",img_len=");               Serial.print(g_test_image_len);
  Serial.print(",model_sum=");             Serial.print(mdl_sum);
  Serial.print(",model_len=");             Serial.print(g_model_data_len);
  Serial.print(",input_sum=");             Serial.print(in_sum);
  Serial.print(",input_bytes=");           Serial.print((int)input->bytes);
  Serial.print(",in_scale=");              Serial.print(input->params.scale, 8);
  Serial.print(",in_zp=");                 Serial.println(input->params.zero_point);

  Serial.print("BENCH,runs=");            Serial.print(kRuns);
  Serial.print(",mean_us=");               Serial.print(total / kRuns);
  Serial.print(",max_us=");                Serial.print(worst);
  Serial.print(",arena_used=");            Serial.print(interpreter->arena_used_bytes());
  Serial.print(",model_bytes=");           Serial.print(g_model_data_len);
  Serial.print(",free_sram=");             Serial.print(largestFreeBlock());
  Serial.print(",pred_cls=");              Serial.print(bc);
  Serial.print(",exp_cls=");               Serial.print(g_test_image_expected_cls);
  Serial.print(",pred_portion=");          Serial.print(bp);
  Serial.print(",exp_portion=");           Serial.print(g_test_image_expected_portion);
  Serial.print(",match=");                 Serial.print(ok ? "OK" : "FAIL");
  // Dump raw int8 head outputs so a device-vs-host disagreement can be traced to
  // a specific tensor rather than guessed at from the argmax alone.
  Serial.print(",cls_raw=");
  for (int i = 0; i < kNumClasses; i++) {
    Serial.print((int)out_cls->data.int8[i]);
    if (i < kNumClasses - 1) Serial.print("|");
  }
  Serial.print(",portion_raw=");
  for (int i = 0; i < kNumPortions; i++) {
    Serial.print((int)out_portion->data.int8[i]);
    if (i < kNumPortions - 1) Serial.print("|");
  }
  Serial.println();
}
#endif

#if PROBE_MODE == 1
// Sample one horizontal line across the test pattern and print the raw byte
// PAIRS, untouched. The colour bars run vertically, so a single row crosses
// every bar and is enough to identify the byte order host-side.
static void dumpRawSamples() {
  Camera.readFrame(frame_buffer);

  const int kN = 64;
  const int y = CAM_H / 2;

  Serial.print("RAW,y=");
  Serial.print(y);
  Serial.print(",n=");
  Serial.print(kN);
  Serial.print(",w=");
  Serial.print(CAM_W);
  Serial.print(",px=");
  for (int i = 0; i < kN; i++) {
    const int x = (i * (CAM_W - 1)) / (kN - 1);
    const int idx = (y * CAM_W + x) * CAM_BPP;
    const uint8_t b0 = frame_buffer[idx];      // first byte off the wire
    const uint8_t b1 = frame_buffer[idx + 1];  // second byte off the wire
    if (b0 < 16) Serial.print('0');
    Serial.print(b0, HEX);
    if (b1 < 16) Serial.print('0');
    Serial.print(b1, HEX);
    if (i < kN - 1) Serial.print('|');
  }
  Serial.println();
}
#endif

void setup() {
  Serial.begin(115200);
  const unsigned long t0 = millis();
  while (!Serial && (millis() - t0 < 5000)) {
  }

  Serial.println();
  Serial.println("# food-tinyml Tier 1 classifier");

#if PROBE_MODE == 1
  initializeShield();
  if (!Camera.begin(QCIF, RGB565, CAM_FPS, OV7675)) {
    Serial.println("BOOT,status=FAIL,reason=camera_init");
    while (1) delay(1000);
  }
  Camera.testPattern();
  Serial.println("BOOT,status=OK,mode=probe");
  return;  // no model, no arena binding needed for the byte-order probe
#endif

  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("BOOT,status=FAIL,reason=schema_mismatch");
    while (1) delay(1000);
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, ARENA_SIZE, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.print("BOOT,status=FAIL,reason=allocate_tensors,arena_req=");
    Serial.println(ARENA_SIZE);
    while (1) delay(1000);
  }

  input = interpreter->input(0);
  bindOutputs();
  if (!out_cls || !out_portion) {
    Serial.println("BOOT,status=FAIL,reason=output_binding");
    while (1) delay(1000);
  }

  Serial.print("BOOT,status=OK,model_bytes=");
  Serial.print(g_model_data_len);
  Serial.print(",arena_req=");
  Serial.print(ARENA_SIZE);
  Serial.print(",arena_used=");
  Serial.print(interpreter->arena_used_bytes());
  Serial.print(",in_scale=");
  Serial.print(input->params.scale, 8);
  Serial.print(",in_zp=");
  Serial.print(input->params.zero_point);
  Serial.print(",free_sram=");
  Serial.println(largestFreeBlock());

#if BENCH_MODE == 1
  runBench();
#else
  initializeShield();
  ledInit();
  if (!Camera.begin(QCIF, RGB565, CAM_FPS, OV7675)) {
    Serial.println("BOOT,status=FAIL,reason=camera_init");
    while (1) { digitalWrite(LEDR, LOW); delay(300);
                digitalWrite(LEDR, HIGH); delay(300); }  // red = camera failed
  }
  Serial.println("# camera ready. 'c' = capture+classify, 'v' = toggle live view.");
#endif
}

void loop() {
#if PROBE_MODE == 1
  // Re-emit periodically: the harness attaches the serial port after the board
  // has already booted, so a setup()-time print is routinely missed.
  delay(3000);
  dumpRawSamples();
  return;
#endif

#if BENCH_MODE == 1
  delay(4000);
  runBench();
#else
  bool go = false;
  if (Serial.available()) {
    const int ch = Serial.read();
    if (ch == 'c' || ch == 'C') go = true;
    if (ch == 'v' || ch == 'V') {
      g_streaming = !g_streaming;
      if (!g_streaming) { ledBlue(false); ledGreen(false); }
    }
  }
  if (readShieldButton()) go = true;

  // --- live view: capture -> downsample -> stream, no inference.
  // Skipping Invoke() keeps this at camera speed (~1.5-2 fps) instead of the
  // ~0.6 fps you get once 1.08 s of inference is in the loop. Frame the shot
  // here, then press 'c' to classify it.
  if (g_streaming && !go) {
    Camera.readFrame(frame_buffer);
    downsampleToInput(input->data.int8);
    streamPreviewHalf();
    // Blink blue once per streamed frame: visible proof the camera is live.
    g_led_phase = !g_led_phase;
    ledBlue(g_led_phase);
    return;
  }

  if (!go) return;

  const unsigned long t_cap = micros();
  Camera.readFrame(frame_buffer);
  const unsigned long cap_us = micros() - t_cap;

  const unsigned long t_pre = micros();
  downsampleToInput(input->data.int8);
  const unsigned long pre_us = micros() - t_pre;

  // Stream the frame BEFORE Invoke(), which overwrites its own input.
  if (g_streaming) streamPreview();

  // Solid green while inference runs, so a long pause is visibly "working".
  ledBlue(false);
  ledGreen(true);

  const unsigned long t_inf = micros();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("INFER,status=FAIL");
    ledGreen(false);
    return;
  }
  const unsigned long inf_us = micros() - t_inf;
  ledGreen(false);

  printResult(inf_us);
  // Section 7 asks for total capture-to-result latency, not just inference.
  Serial.print("TIMING,capture_us=");
  Serial.print(cap_us);
  Serial.print(",preprocess_us=");
  Serial.print(pre_us);
  Serial.print(",infer_us=");
  Serial.print(inf_us);
  Serial.print(",total_us=");
  Serial.println(cap_us + pre_us + inf_us);
#endif
}
