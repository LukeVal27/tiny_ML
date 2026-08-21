/*
 * camera_view — live preview of what the CLASSIFIER sees.
 *
 * Standalone diagnostic sketch. Flashing it replaces classifier_tier1 on the
 * board; reflash the classifier when you are done framing.
 *
 * It streams the 96x96 RGB image *after* the same centre-crop and downsample the
 * classifier performs, not the raw QCIF frame. That is the decision-relevant
 * view: it shows whether the plate rim is inside the model's field of view and
 * how much detail survives the reduction to 96x96. A raw-frame preview can look
 * perfectly framed while the model's actual input has the rim cropped off.
 *
 * The crop/downsample below deliberately mirrors downsampleToInput() in
 * classifier_tier1.ino. If that path changes, change it here too or the preview
 * stops telling the truth.
 *
 * Wire format, repeated forever:
 *     "FRM:" + 27648 raw bytes (96*96*3, RGB888, row-major)
 * USB CDC ignores the nominal baud rate and runs at USB speed, so this streams
 * at several frames per second despite the frame size.
 */

#include <TinyMLShield.h>

#define CAM_W 176
#define CAM_H 144
#define CAM_BPP 2
#define FRAME_BYTES (CAM_W * CAM_H * CAM_BPP)

#define IN_W 96
#define IN_H 96

// Square centre region of the QCIF frame we downsample from (176x144 -> 144x144).
#define SQ 144
#define SQ_X0 ((CAM_W - SQ) / 2)
#define SQ_Y0 ((CAM_H - SQ) / 2)

__attribute__((aligned(16))) static uint8_t frame_buffer[FRAME_BYTES];
__attribute__((aligned(16))) static uint8_t out_rgb[IN_W * IN_H * 3];

// RGB565 big-endian: first byte off the wire is the high byte. Verified
// on-device against Camera.testPattern() -- big-endian reproduces the colour
// bars, little-endian yields no recognisable colours. See
// results/rgb565_byteorder.json.
static inline void unpack565(uint8_t hi, uint8_t lo, int &r, int &g, int &b) {
  const uint16_t px = (uint16_t(hi) << 8) | lo;
  r = ((px >> 11) & 0x1F) * 255 / 31;
  g = ((px >> 5) & 0x3F) * 255 / 63;
  b = (px & 0x1F) * 255 / 31;
}

static void buildPreview() {
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
      const int o = (oy * IN_W + ox) * 3;
      out_rgb[o + 0] = (uint8_t)(rs / 4);
      out_rgb[o + 1] = (uint8_t)(gs / 4);
      out_rgb[o + 2] = (uint8_t)(bs / 4);
    }
  }
}

void setup() {
  Serial.begin(115200);
  const unsigned long t0 = millis();
  while (!Serial && (millis() - t0 < 5000)) {
  }

  initializeShield();
  if (!Camera.begin(QCIF, RGB565, 5, OV7675)) {
    Serial.println("BOOT,status=FAIL,reason=camera_init");
    while (1) delay(1000);
  }
  Serial.println("BOOT,status=OK,mode=camera_view");
}

void loop() {
  Camera.readFrame(frame_buffer);
  buildPreview();
  Serial.write((const uint8_t *)"FRM:", 4);
  Serial.write(out_rgb, sizeof(out_rgb));
  Serial.flush();
}
