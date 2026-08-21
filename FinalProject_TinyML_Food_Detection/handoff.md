# EE446 TinyML — Food Detection & Portion Estimation
## Project handoff, 2026-08-18

Written to be read cold. Every figure was read from a file in `results/`
immediately before writing. Anything unverified is labelled as such.

---

## 0. One-paragraph summary

A 5-class food classifier plus a 3-tier portion-size head, quantized to int8 and
running on an Arduino Nano 33 BLE Sense with an OV7675 camera. The host-side
model is finished and hardware-verified. **The open problem is real-world camera
accuracy**: on live captures broccoli works, rice is marginal, and beef fails
completely. The cause has been measured, not guessed (Section 6).

---

## 1. Where everything is

| what | path |
|---|---|
| deployed int8 model | `results/models/real_cw_int8.tflite` (221,888 B) |
| trainable checkpoint | `results/checkpoints/real_cw.keras` |
| flashed C array | `deploy/nano/classifier_tier1/model_data.cc` |
| deploy sketch | `deploy/nano/classifier_tier1/classifier_tier1.ino` |
| raw camera diagnostic | `deploy/nano/camera_raw/camera_raw.ino` |
| accuracy metrics | `results/quantization_real_cw.json` |
| model comparison | `results/compare_on_real_test.json` |
| on-device telemetry | `results/device_runs.jsonl` |
| RAM sweep | `results/compile_sweep.jsonl` |
| dataset counts | `results/foodseg103_counts.json` |
| live captures (clean) | `results/live_captures.jsonl` |
| live captures (confounded) | `results/live_captures_confounded.jsonl` |
| presentation | `EE446_final_presentation.pptx` (12 slides) |

**Read these first, in this order:** `CLAUDE.md` (working rules + settled
questions), this file, then `results/REPORT.txt`.

Other docs: `STATUS.md` and `PROJECT_HANDOFF.md` are earlier snapshots — accurate
at the time but superseded by this file for anything about live-camera results.

---

## 2. Hardware

Arduino Nano 33 BLE Sense (nRF52840, Cortex-M4F @ 64 MHz, 262,144 B SRAM,
983,040 B flash) + OV7675 on the Arduino Tiny ML shield.
FQBN `arduino:mbed_nano:nano33ble`. Capture: QCIF 176×144 RGB565.

**Verified board facts** (`results/compile_sweep.jsonl`, `results/device_runs.jsonl`):

- fixed overhead (mbed OS + TFLM + Serial): **51,792 B**
- RGB565 QCIF frame: **50,688 B**
- tensor arena in use: **113,516 B**
- largest free block at runtime: **54,284 B**
- sketch flash: **500,648 B (50.9%)**
- max viable arena: ~135 KB RGB565, ~160 KB grayscale (linker overflow at 190,000)

---

## 3. Model

157,512 parameters. Input 96×96×3 RGB. Shared depthwise-separable backbone,
155,968 parameters shared between two heads.

```
96²×3 → stem s2 → 48²×16 → b1-b2 → 24²×32 → b3-b4 → 12²×64
      → b5-b6 → 6²×128 → 1×1 widen → 6²×192
                                   ├── MAX pool 6×6 → 5 classes  (965 params)
                                   └── AVG pool 6×6 → 3 tiers    (579 params)
```

Deployed ops: `CONV_2D ×13 · DEPTHWISE_CONV_2D ×6 · ADD ×3 · MAX_POOL_2D ×1 ·
AVERAGE_POOL_2D ×1 · FULLY_CONNECTED ×2 · SOFTMAX ×2`.

Design decisions that are load-bearing — do not revert without reading
`CLAUDE.md`:

- **The heads pool differently.** Class head max-pools (presence); portion head
  average-pools (extent). With one shared global-average pool the classifier
  stalled at ~0.34 validation accuracy while the portion head was already >0.90.
- **`MaxPooling2D`/`AveragePooling2D` sized to the grid, not `Global*`** — same
  values, but exports better-tested TFLM kernels.
- **Export from a batch-1 clone** (`model.to_batch1`) — a `None` batch emits
  `SHAPE`/`STRIDED_SLICE`/`PACK`.
- **Bind output tensors by width** (5 = class, 3 = portion), never by index.
- **The sketch downsamples, it does not centre-crop** — the portion tier is
  defined against the plate rim, so the whole plate must stay in frame.

---

## 4. Data

**Source:** FoodSeg103, `EduardoPacheco/FoodSeg103` on HuggingFace, Apache-2.0,
ungated. Wu et al., ACM MM 2021.

**Class mapping** (`results/foodseg103_mapping.json`) — lossy, and this matters:

| our class | FoodSeg103 | idx | caveat |
|---|---|---|---|
| broccoli | `broccoli` | 87 | clean |
| rice | `rice` | 66 | clean |
| potato | `potato` | 70 | `french fries` (3) excluded |
| chicken | `chicken duck` | 48 | **merged label**, no pure chicken |
| beef | `steak` | 46 | **substitution**, no beef class exists |

**Pipeline:** masks → 4,996 RGBA cutouts (`data/foodseg_cutouts.py`) → 18,000
composited plates (`data/compose_real.py`), tiers balanced 5054/5051/4895.

Cutouts kept per class (train/test): chicken 838/388, potato 773/300,
beef 717/332, broccoli 689/297, rice 457/205.

**Portion labels are constructed, not estimated.** Food is pasted at a *sampled*
fraction of plate area, so the label is exact by definition:
`mass_g ≈ f × 513` (22 cm plate, 1.5 cm depth, 0.9 g/cm³) → small `f<0.156`
(<80 g), medium `0.156–0.351` (80–180 g), large `f>0.351` (>180 g).
Plate apparent diameter is jittered to 62–92% of frame width so raw pixel area
is an ambiguous cue and the model must measure against the rim.

`data/composed` (18,000 synthetic plates) is the superseded original corpus.
`data/composed_v4` exists but is **unused**.

---

## 5. Results — host-side, verified

### Corpus ablation, all on the same real test set, n = 3,000
(`results/compare_on_real_test.json`)

| training corpus | macro-F1 | acc | portion acc |
|---|---|---|---|
| synthetic only | 0.5074 | 0.5403 | 0.8290 |
| real, unweighted | 0.5859 | 0.6393 | 0.9233 |
| real + synthetic | 0.5330 | 0.5957 | 0.8770 |
| **real + class weights** | **0.6462** | **0.6460** | **0.9237** |

Mixing corpora is worse than real alone. Replace, do not augment.

### Baseline vs compressed (`results/quantization_real_cw.json`)

| | macro-F1 | acc | portion acc |
|---|---|---|---|
| FP32 train | 0.6832 | 0.6850 | 0.9261 |
| FP32 validation | 0.6475 | 0.6376 | 0.9253 |
| FP32 test | 0.6462 | 0.6460 | 0.9237 |
| **int8 test (deployed)** | **0.6432** | **0.6433** | **0.9187** |

int8 cost: **−0.0030** macro-F1 at 2.8× compression (621,956 → 221,888 B).
Train-to-test gap 0.037 — not overfitting.

Per-class F1 (int8): broccoli 0.969, beef 0.662, rice 0.557, potato 0.519,
chicken 0.509.

**Portion off-by-two errors: 0 out of 3,000.**

### Class weighting is required

Without it, chicken collapses to F1 0.115 / recall 0.063 on a perfectly balanced
corpus — confusability, not frequency. Weights (inverse recall, clipped 4×,
mean-normalised, in CLASSES order):
`--class-weights 2.077,0.524,1.159,0.552,0.688`. Chicken recovers to 0.508.

### On-device (`results/device_runs.jsonl`, 2026-08-13)

`arena_used=113,516` · `mean_us=1,082,349` · `match=OK` ·
`img_sum == input_sum == 449,425`.

Live camera path: capture 331–583 ms, preprocess ~68 ms, inference 1,082 ms,
**capture-to-result ≈ 1,482–1,733 ms**.

---

## 6. The open problem: real-world captures

**Current clean session** (`results/live_captures.jsonl`, 19 labelled):

| class | correct | n |
|---|---|---|
| broccoli | 3 | 3 |
| rice | 5 | 11 |
| beef | 0 | 5 |

All 19 returned `portion=large`.

An earlier confounded session of 69 captures is archived in
`results/live_captures_confounded.jsonl` (broccoli 6/6, rice 2/5, beef 0/15).

### What has been measured about the failure

Measured from a raw sensor frame (`results/raw_frame.bin`, decoded big-endian):

| region | R | G | B | R−B |
|---|---|---|---|---|
| **training beef, plate centre** | 141.9 | 114.2 | 102.0 | **+39.8** |
| **camera beef, plate centre** | 148.7 | 142.7 | 147.4 | **+1.4** |

Training beef is strongly red-dominant. The camera renders it neutral grey.
That is the measured reason beef is misclassified as rice or broccoli.

Per-pixel saturation is **not** the problem: camera 21–24 vs training beef 26.2.

### Ruled out — do not re-investigate

- **RGB565 byte order.** Big-endian `(hi<<8)|lo` is correct. Verified twice:
  against `Camera.testPattern()` (7 clean colour bars; little-endian gives zero
  recognisable colours) and against a real scene
  (`results/raw_decode_compare.png` — big-endian decodes show plate and food,
  little-endian decodes are noise). Evidence: `results/rgb565_byteorder.json`.
- **Red/blue channel swap.** Red decodes as red.
- **Low saturation.** Measured close to training (above).
- **CMSIS-NN as a latency fix.** Already active — 26 `_s8` symbols, 251 DSP SIMD
  instructions — and measured *no* speedup (1,092,437 µs vs 1,076,311 µs
  reference). Evidence: `results/cmsisnn_findings.json`. Census tool:
  `harness/dsp_symbols.sh <elf>`.

### Unresolved

- **Why the persistent `portion=large`.** 19/19 in the clean session. The tier is
  defined against the plate rim; raw frames show the plate overflowing the frame.
  Not yet confirmed fixed.
- **Whether R−B can be raised** by lighting or by a redder cut. Untested.
- **Latency 1,082 ms.** Cause unknown; kernel selection is ruled out.
- **`Camera.begin` hangs at `CAM_FPS=15` and at `QQVGA`.** Both produced zero
  serial output. QCIF @ 5 fps is the only configuration confirmed working.
  Preview ceiling measured at **2.4 fps**, bounded by the bit-banged
  `readFrame()`, not by serial throughput (shrinking the payload 4× moved fps
  only 2.45 → 2.30).

---

## 7. Tools

```bash
# host
python3 -m data.foodseg_cutouts            # FoodSeg103 -> 4,996 cutouts
python3 -m data.compose_real               # -> 18,000 composited plates
python3 -m train.train --alpha 1.0 --source real --aug-level real \
        --epochs 100 --class-weights 2.077,0.524,1.159,0.552,0.688 --tag real_cw
python3 -m compress.quantize_int8 --ckpt results/checkpoints/real_cw.keras \
        --source real --tag real_cw
python3 -m deploy.to_c_array --model results/models/real_cw_int8.tflite
python3 -m eval.make_report real_cw

# hardware
python3 harness/compile_sweep.py                       # RAM sweep, no board
python3 harness/flash_and_measure.py --sketch <dir> --define KEY=VAL
python3 harness/live_capture.py                        # labelled capture session
python3 harness/camera_view.py                         # live preview + classify
python3 harness/camera_raw.py                          # raw camera, r = decode test
```

Sketch build flags: `ARENA_SIZE` (use 120000), `BENCH_MODE` (1 = baked image,
0 = live camera), `PROBE_MODE`, `CAM_FPS` (**keep at 5**).

---

## 8. Traps already paid for

- **`Invoke()` destroys its own input.** The input tensor is aliased with arena
  scratch. Any sketch invoking more than once must refill first — see
  `fillInput()`. This produced a device/host mismatch that cost three wrong
  hypotheses before a byte checksum isolated it.
- **`pgrep -f` self-matches.** Five polling shells ran 90+ minutes because each
  matched its own command line. Use `[t]rain` or don't poll.
- **Deleting pptx slides needs `drop_rel`.** Removing from `sldIdLst` alone
  orphans the part and produces duplicate zip entries that PowerPoint may reject.
- **`arduino-cli upload` takes no `--build-property`.** Compile with an explicit
  `--build-path`, upload with a matching `--input-dir`.
- **The Arduino IDE's `serial-monitor` holds the port** and blocks the bootloader
  touch. The harness kills it before each upload.
- **"NO TELEMETRY" from the harness is often a false alarm** — `BENCH_MODE=0` and
  the raw viewer emit no matching telemetry tags by design.
- **Matplotlib's default keymap** (`s` = save, `c`/`v` = navigate) collides with
  viewer controls; `harness/camera_raw.py` clears it.

---

## 9. Presentation

`EE446_final_presentation.pptx`, 12 slides, ordered to the rubric's six mandatory
topics (`EE446_TinyML_Summer2026_Project_Presentation_And_Demo_Guidelines_and_Rubric.pdf`).

LibreOffice is installed; render to check layout with:
```bash
/Applications/LibreOffice.app/Contents/MacOS/soffice --headless \
    --convert-to pdf --outdir /tmp/slides EE446_final_presentation.pptx
```

**Known gap against the rubric:** it requires 10–15 on-device samples per class
with counts explicitly stated. Current clean session has broccoli 3, rice 11,
beef 5. Slide 11 reports the confounds as measured findings rather than quoting
an accuracy figure from contaminated data.

One asset outstanding: the capture-rig photo on slide 10, marked "PHOTO TO ADD".

---

## 10. Known limitations

1. **FoodSeg103 has no beef and no pure chicken.** Our beef is `steak`, our
   chicken is `chicken duck`. Fixed property of the only free mask-annotated
   corpus covering these foods; caps chicken near 0.51. Training cannot fix it.
2. **Rice and potato are not separable at 96×96.** 183 rice → potato and 126
   potato → rice on the test set. Higher input resolution would break the RAM
   budget.
3. **Training food is real but the plate is drawn.** Real shadows, specular
   highlights and depth are absent from training.
4. **Portion thresholds are modelled, not weighed.** The head scores 0.9187
   against its own definition; real-world gram accuracy is unverified.
5. **Single-label only.** The class head ends in softmax, so it reports one food
   per plate. Multi-label was assessed as feasible at near-zero memory cost
   (5 sigmoids + a 5×3 tier head, output tensor 8 → 20 values) but is not built.
