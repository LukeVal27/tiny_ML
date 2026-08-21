/*
 * camera_raw — a deliberately minimal camera viewer.
 *
 * No model, no TFLite, no tensor arena. It exists to answer one question:
 * does the CAMERA produce a good image, independent of everything else?
 *
 * The classifier sketch shows the 96x96 tensor, which has already been through a
 * centre-crop, a 2x2 box downsample and int8 quantisation. If that view looks
 * wrong we cannot tell whether the fault is the sensor, the RGB565 decode, or
 * our own preprocessing. This sketch removes all of those stages.
 *
 * Two modes over serial:
 *   (default)  fast half-resolution stream for AIMING       -> "HALF:" + 88*72*3
 *   'r'        one full-resolution raw dump for DIAGNOSIS   -> "RAW1:" + 176*144*2
 *   s/S  saturation -/+     c/C  contrast -/+     b/B  brightness -/+
 *   '0'        restore the sensor defaults
 *
 * The image-control keys exist because a measured raw frame showed a saturation
 * spread of only 3-5 levels out of 255 -- plate, wooden board and browned steak
 * all rendering as the same mid-grey. This model leans on colour (a
 * colour-histogram baseline scores 0.510 against the CNN's 0.559), so a
 * colourless frame removes the main class cue. These registers are the cheapest
 * possible test of whether the colour is recoverable at the sensor.
 *
 * The raw dump is the sensor's own RGB565 bytes, completely untouched, so the
 * host can decode them any way it likes and settle byte-order or row-alignment
 * questions from evidence.
 *
 * Resolution note: QCIF (176x144). QQVGA was tried and produced ZERO output --
 * the driver appears to hang on it. readFrame() is bit-banged over GPIO with no
 * DMA and dominates the frame time, so the only lever left is sensor FPS: it
 * shortens the VSYNC wait. Shrinking the serial payload alone changed nothing
 * when measured (66 -> 16 KB/s moved fps only 2.45 -> 2.30).
 */

#include <TinyMLShield.h>

#define CAM_W 176
#define CAM_H 144
#define CAM_BPP 2
#define FRAME_BYTES (CAM_W * CAM_H * CAM_BPP)

// Half-res preview: every other pixel, so 80x60 RGB = 14,400 bytes.
#define OUT_W (CAM_W / 2)
#define OUT_H (CAM_H / 2)

#ifndef CAM_FPS
#define CAM_FPS 15          // 1, 5, 10, 15, 30 — higher shortens the VSYNC wait
#endif

__attribute__((aligned(16))) static uint8_t frame[FRAME_BYTES];

// Big-endian RGB565: first byte off the wire is the high byte. Verified
// on-device against Camera.testPattern() — big-endian reproduces the colour
// bars; little-endian yields no recognisable colours.
static inline void unpack565(uint8_t hi, uint8_t lo, int &r, int &g, int &b) {
  const uint16_t px = (uint16_t(hi) << 8) | lo;
  r = ((px >> 11) & 0x1F) * 255 / 31;   // only 32 levels of red
  g = ((px >> 5) & 0x3F) * 255 / 63;    // 64 levels of green
  b = (px & 0x1F) * 255 / 31;           // only 32 levels of blue
}

void setup() {
  Serial.begin(115200);
  const unsigned long t0 = millis();
  while (!Serial && (millis() - t0 < 4000)) {
  }
  initializeShield();
  pinMode(LEDB, OUTPUT); pinMode(LEDR, OUTPUT);
  digitalWrite(LEDB, HIGH); digitalWrite(LEDR, HIGH);

  if (!Camera.begin(QCIF, RGB565, CAM_FPS, OV7675)) {
    Serial.println("BOOT,status=FAIL,reason=camera_init");
    while (1) { digitalWrite(LEDR, LOW); delay(300);
                digitalWrite(LEDR, HIGH); delay(300); }
  }
  Serial.print("BOOT,status=OK,mode=camera_raw,w=");
  Serial.print(CAM_W); Serial.print(",h="); Serial.print(CAM_H);
  Serial.print(",fps="); Serial.println(CAM_FPS);
  applyControls();
}

static bool blink = false;

// OV7675 image-control state. The driver exposes these but the defaults are
// whatever the sensor powers up with, which measured as nearly colourless.
static int g_sat = 128, g_con = 64, g_bri = 128;

static void applyControls() {
  Camera.setSaturation(g_sat);   // 0-255
  Camera.setContrast(g_con);     // 0-127
  Camera.setBrightness(g_bri);   // 0-255
  Serial.print("CTRL,saturation="); Serial.print(g_sat);
  Serial.print(",contrast=");      Serial.print(g_con);
  Serial.print(",brightness=");    Serial.println(g_bri);
}

void loop() {
  if (Serial.available()) {
    const int c = Serial.read();
    bool changed = true;
    switch (c) {
      case 's': g_sat = max(0,   g_sat - 32); break;
      case 'S': g_sat = min(255, g_sat + 32); break;
      case 'c': g_con = max(0,   g_con - 16); break;
      case 'C': g_con = min(127, g_con + 16); break;
      case 'b': g_bri = max(0,   g_bri - 16); break;
      case 'B': g_bri = min(255, g_bri + 16); break;
      case '0': g_sat = 128; g_con = 64; g_bri = 128; break;
      default:  changed = false;
    }
    if (changed) { applyControls(); return; }
    if (c == 'r' || c == 'R') {
      Camera.readFrame(frame);
      Serial.write((const uint8_t *)"RAW1:", 5);
      Serial.write(frame, FRAME_BYTES);     // untouched sensor bytes
      return;
    }
  }

  Camera.readFrame(frame);

  // Half-res RGB, emitted a row at a time from a small stack buffer so the
  // sketch needs no second framebuffer.
  uint8_t row[OUT_W * 3];
  Serial.write((const uint8_t *)"HALF:", 5);
  for (int y = 0; y < CAM_H; y += 2) {
    int o = 0;
    for (int x = 0; x < CAM_W; x += 2) {
      const int i = (y * CAM_W + x) * CAM_BPP;
      int r, g, b;
      unpack565(frame[i], frame[i + 1], r, g, b);
      row[o++] = (uint8_t)r; row[o++] = (uint8_t)g; row[o++] = (uint8_t)b;
    }
    Serial.write(row, sizeof(row));
  }

  blink = !blink;
  digitalWrite(LEDB, blink ? LOW : HIGH);   // blue blink = streaming
}
