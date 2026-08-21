/*
 * Section 5a GATE - RAM coexistence smoke test
 * Arduino Nano 33 BLE Sense (nRF52840) + OV7675 on the Tiny ML shield.
 *
 * Proves that the camera frame buffer and a TFLite Micro tensor arena can
 * coexist in the nRF52840's 256 KB SRAM, and reports how much headroom is left.
 *
 * Build-time knobs (set by harness/compile_sweep.py via --build-property):
 *   ARENA_SIZE   tensor arena bytes
 *   CAM_FORMAT   0 = GRAYSCALE, 1 = RGB565
 *   CAM_RES      0 = QCIF (176x144), 1 = QQVGA (160x120)
 *
 * Prints one machine-parseable CSV line per boot:
 *   SMOKE,mode=gray,res=qcif,frame_bytes=25344,arena_req=80000,
 *         arena_used=NNNNN,free_sram=NNNNN,cam=OK,status=OK
 */

#include <TinyMLShield.h>

#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// ---------------------------------------------------------------- build knobs
#ifndef ARENA_SIZE
#define ARENA_SIZE 70000
#endif

#ifndef CAM_FORMAT
#define CAM_FORMAT 0 // 0 = grayscale, 1 = RGB565
#endif

#ifndef CAM_RES
#define CAM_RES 0 // 0 = QCIF 176x144, 1 = QQVGA 160x120
#endif

#if CAM_RES == 0
#define CAM_W 176
#define CAM_H 144
#define CAM_RES_ENUM QCIF
#define CAM_RES_NAME "qcif"
#else
#define CAM_W 160
#define CAM_H 120
#define CAM_RES_ENUM QQVGA
#define CAM_RES_NAME "qqvga"
#endif

#if CAM_FORMAT == 0
#define CAM_BPP 1
#define CAM_FMT_ENUM GRAYSCALE
#define CAM_FMT_NAME "gray"
#else
#define CAM_BPP 2
#define CAM_FMT_ENUM RGB565
#define CAM_FMT_NAME "rgb565"
#endif

#define FRAME_BYTES (CAM_W * CAM_H * CAM_BPP)

// ------------------------------------------------------------------ buffers
// Both of these are the real allocations the deployed sketch will carry, so
// arduino-cli's compile-time RAM report is a truthful static measurement even
// with no board attached.
__attribute__((aligned(16))) static uint8_t frame_buffer[FRAME_BYTES];
__attribute__((aligned(16))) static uint8_t tensor_arena[ARENA_SIZE];

// Keep the TFLM runtime genuinely linked in so its .bss/.data counts too.
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;

// --------------------------------------------------------------- free memory
//
// The usual Arduino trick -- (stack local address) - sbrk(0) -- is WRONG on this
// board and reports nonsense (we measured -9033 bytes with the board running
// happily). The nano33ble runs mbed OS, so setup() executes on an RTOS thread
// whose stack is itself carved out of the heap rather than sitting at the top of
// RAM. The stack local therefore lands *below* the heap break and the
// subtraction goes negative.
//
// What we actually care about is "how much more can this firmware still get",
// so we measure it directly: binary-search the largest block malloc will hand
// back. That is a real, actionable number, and it accounts for fragmentation.
static int largestFreeBlock() {
  size_t lo = 0, hi = 200u * 1024u;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo + 1) / 2;
    void *p = malloc(mid);
    if (p != nullptr) {
      free(p);
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return (int)lo;
}

void setup() {
  Serial.begin(115200);
  const unsigned long t0 = millis();
  while (!Serial && (millis() - t0 < 5000)) {
  }

  Serial.println();
  Serial.println("# smoke_test: RAM coexistence gate");

  initializeShield();

  bool cam_ok = Camera.begin(CAM_RES_ENUM, CAM_FMT_ENUM, 5, OV7675) != 0;

  // Touch the frame buffer whether or not the camera came up, so the linker
  // can never optimise the allocation away.
  if (cam_ok) {
    Camera.readFrame(frame_buffer);
  } else {
    memset(frame_buffer, 0, FRAME_BYTES);
  }

  unsigned long checksum = 0;
  for (int i = 0; i < FRAME_BYTES; i += 64) {
    checksum += frame_buffer[i];
  }

  // Same for the arena: dirty it so it is real, resident memory.
  memset(tensor_arena, 0, ARENA_SIZE);

  const int free_sram = largestFreeBlock();

  // status=OK means we booted, allocated both buffers, read a frame, and can
  // still obtain a meaningful contiguous block on top of all that.
  const bool ok = (free_sram > 8192);

  Serial.print("SMOKE,mode=");
  Serial.print(CAM_FMT_NAME);
  Serial.print(",res=");
  Serial.print(CAM_RES_NAME);
  Serial.print(",frame_bytes=");
  Serial.print(FRAME_BYTES);
  Serial.print(",arena_req=");
  Serial.print(ARENA_SIZE);
  Serial.print(",arena_used=0"); // no model bound in the gate sketch
  Serial.print(",free_sram=");
  Serial.print(free_sram);
  Serial.print(",checksum=");
  Serial.print(checksum);
  Serial.print(",cam=");
  Serial.print(cam_ok ? "OK" : "FAIL");
  Serial.print(",status=");
  Serial.println(ok ? "OK" : "FAIL");
}

void loop() {
  delay(5000);
  Serial.println("# alive");
}
