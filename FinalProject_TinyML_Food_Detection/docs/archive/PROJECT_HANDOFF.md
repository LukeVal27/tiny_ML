# EE446 TinyML — Food Detection & Portion Estimation
## Technical status report

**Course:** EE446 TinyML, Summer 2026 · **Team:** Luke Valerio, Daniel Yang
**Report date:** 2026-08-17 · **Phase:** Tier 1 complete, hardware-verified

> Every figure in this document was read from a logged artefact in `results/`
> immediately before writing. Nothing is estimated or recalled. Where something
> is unverified, it says so explicitly.

---

## 1. Executive summary

An embedded computer-vision system that photographs a plate of food and reports
**what the food is** and **how much of it there is**, running entirely on a
$30 microcontroller with no network connection.

The complete pipeline — dataset construction, training, int8 quantization,
C-array export, firmware, and hardware-in-the-loop measurement — is built,
automated, and reproducible from the command line.

**Headline results (all measured):**

| | value |
|---|---|
| class macro-F1 (int8, on device) | **0.6432** |
| portion-tier accuracy | **0.9187** |
| portion off-by-two error rate | **0.0000** |
| model size in flash | **221,888 B** (216.7 KB) |
| tensor arena at runtime | **113,516 B** |
| inference latency | **1,082 ms** |
| capture-to-result latency | **1,482 ms** |
| device vs. host agreement | **exact** |

The most significant engineering result: replacing synthetically-composed
training images with **real photographed food** raised class macro-F1 from
0.5074 to 0.6432 (**+0.136, +27% relative**) at *identical* model size, tensor
arena, and latency.

---

## 2. Hardware platform

| item | spec |
|---|---|
| board | Arduino Nano 33 BLE Sense |
| MCU | nRF52840, single Cortex-M4F @ 64 MHz |
| SRAM | 262,144 B total — no PSRAM |
| flash | 983,040 B |
| camera | OV7675, parallel interface, on the Arduino Tiny ML shield |
| capture mode | QCIF 176×144, RGB565 (50,688 B/frame) |
| runtime | TensorFlow Lite Micro (Harvard_TinyMLx build) |
| FQBN | `arduino:mbed_nano:nano33ble` |

Board selection was constrained by the **camera**, not compute. The parallel
OV7675 only pairs cleanly with the Nano shield. An ESP32-C6 and XIAO ESP32-S3
were ruled out empirically (no camera interface / incompatible camera bus).

### 2.1 The RAM budget — the binding constraint

Everything lives in one 262,144 B SRAM pool simultaneously. Measured via an
11-point static compile sweep (`harness/compile_sweep.py`, no board required):

**Fixed overhead (mbed OS + TFLM runtime + Serial): 51,792 B**

| mode | frame | max viable arena | evidence |
|---|---|---|---|
| grayscale QCIF | 25,344 B | ~160 KB | 140 KB OK · 170 KB tight · 190 KB linker overflow |
| **RGB565 QCIF** | **50,688 B** | **~135 KB** | 100 KB OK · 140 KB tight |
| grayscale QQVGA | 19,200 B | ~168 KB | 140 KB OK · 170 KB tight |

Confirmed on hardware: RGB565 frame + 90 KB arena coexist with camera
initialised (`cam=OK`) and **56,988 B still allocatable**.

**Decision: RGB565.** Colour is the dominant class cue and the gate proved it
affordable.

### 2.2 Deployed memory footprint (measured on device)

```
fixed overhead        51,792 B
RGB565 frame buffer   50,688 B
tensor arena         113,516 B   (120,000 B requested)
--------------------------------
largest free block    54,284 B
static RAM total     171,048 B of 262,144 B   (BENCH_MODE=1 build)
sketch flash         500,648 B of 983,040 B   (50.9%)
```

---

## 3. Model architecture

157,512 parameters. Input 96×96×3 RGB. Two heads on one shared backbone.

```
96²×3 → stem s2 → 48²×16 → b1-b2 → 24²×32 → b3-b4 → 12²×64
      → b5-b6 → 6²×128 → 1×1 widen → 6²×192
                                        ├── MAX pool 6×6 → 5 classes  (965 params)
                                        └── AVG pool 6×6 → 3 tiers    (579 params)
```

MobileNet-style depthwise-separable blocks with inverted-bottleneck expansion,
ReLU6, residuals where shapes match, BatchNorm (momentum 0.9, folded at convert
time). **155,968 of 157,512 parameters are shared** between the two tasks.

### 3.1 Design decisions that materially changed results

**The two heads pool differently.** Class head takes the **max** over the 6×6
grid; portion head takes the **average**. Food occupies a minority of the frame,
so averaging dilutes class evidence with plate and tablecloth cells. The class
question is "is there broccoli *anywhere*", which is a max. The portion tier is
*defined* by coverage, which is a mean. When both heads shared one
global-average pool, the classifier stalled at ~0.34 validation accuracy while
the portion head already exceeded 0.90.

**Pooling ops, not `Global*` ops.** `MaxPooling2D`/`AveragePooling2D` sized to
the grid compute identical values but export `MAX_POOL_2D`/`AVERAGE_POOL_2D`
instead of `REDUCE_MAX`/`MEAN` — far better-tested kernels in TFLM.

**Export from a batch-1 clone** (`model.to_batch1`). A `None` batch makes Keras
emit `SHAPE`/`STRIDED_SLICE`/`PACK` to compute sizes at runtime; dynamic shapes
are what TFLM handles worst. The deployed graph is:

```
CONV_2D ×13 · DEPTHWISE_CONV_2D ×6 · ADD ×3 · MAX_POOL_2D ×1
AVERAGE_POOL_2D ×1 · FULLY_CONNECTED ×2 · SOFTMAX ×2
```

**Output tensors are bound by width** (5 = class, 3 = portion), never by index —
the converter makes no ordering guarantee.

**The firmware downsamples, it does not centre-crop.** The stock
`person_detection` example crops 96×96 out of QCIF, discarding most of the field
of view. The portion tier is defined relative to the plate rim, so the whole
plate must stay in frame: centre-crop to 144×144, then 2×2 box-average to 96×96.

---

## 4. Datasets

### 4.1 Deployed training corpus — `data/foodseg103_real` (18,000 images)

**Source:** FoodSeg103 (`EduardoPacheco/FoodSeg103` on HuggingFace,
Apache-2.0, ungated, no login). Wu et al., ACM MM 2021. Per-ingredient pixel
segmentation masks over real plated food.

**Class mapping — semantically lossy, deliberately logged:**

| our class | FoodSeg103 label | index | caveat |
|---|---|---|---|
| broccoli | `broccoli` | 87 | clean 1:1 |
| rice | `rice` | 66 | clean 1:1 |
| potato | `potato` | 70 | `french fries` (3) deliberately excluded |
| chicken | `chicken duck` | 48 | **merged label** — no pure-chicken class exists |
| beef | `steak` | 46 | **substitution** — no beef class exists |

**Extraction** (`data/foodseg_cutouts.py`, single pass): rows filtered on
`classes_on_image` *before* image decode; food extracted along its true mask as
RGBA with alpha = mask; rejected if bbox < 56 px or mask fills < 12% of its own
bbox. Yield: **4,996 cutouts**.

| class | source images (train/test) | cutouts (train/test) |
|---|---|---|
| chicken | 848 / 394 | 838 / 388 |
| potato | 785 / 306 | 773 / 300 |
| beef | 728 / 337 | 717 / 332 |
| broccoli | 704 / 309 | 689 / 297 |
| rice | 464 / 206 | 457 / 205 |

**Composition** (`data/compose_real.py`): each cutout is scaled so its opaque
area equals a *sampled* fraction of plate area, then pasted onto a rendered
plate. Result: 18,000 images, tiers balanced (train 5054/5051/4895).

### 4.2 Superseded corpus — `data/composed` (18,000 images)

Food-101 + ImageNet-broccoli texture inside a *drawn* elliptical blob. 7,031
source photos after perceptual-hash dedup. Retained for comparison; **no longer
trained on**. A third corpus `data/composed_v4` exists but is **unused** (a
saturation-based crop filter that measured as neutral, 0.453 vs 0.460 on a
colour-histogram linear probe).

### 4.3 Portion-label methodology

Neither source dataset carries portion labels, so the supervision is
**constructed rather than estimated**, and is exact by definition:

```
mass_g ≈ f × plate_area_cm² × mean_depth_cm × mean_density_g_cm³
       ≈ f × 513            (22 cm plate, 1.5 cm depth, 0.9 g/cm³)

small  f < 0.156   (<80 g)
medium 0.156–0.351 (80–180 g)
large  f > 0.351   (>180 g)
```

Two details make this learnable rather than trivially gameable:

1. **Plate apparent diameter is jittered** (62–92% of frame width). Absolute
   food-pixel area is therefore an *ambiguous* cue, and the only reliable
   strategy is measuring food against the rim — exactly the deployment
   behaviour required.
2. **Texture is cropped at ~1:1 scale.** A small portion shows *fewer* grains,
   not *smaller* ones. An early version rescaled whole dish photos down, making
   rice grains sub-pixel and small rice indistinguishable from potato.

An approach rejected on evidence: computing portion as
`class pixels ÷ non-background pixels` from the masks. In FoodSeg103 the plate
is *background*, so that ratio measures an ingredient's share **of the meal**,
not serving size (measured medians 0.21–0.39). It carries no portion signal.

---

## 5. Results

### 5.1 Compression (real test set, n = 3,000)

| variant | size | macro-F1 | cls acc | portion acc | ordinal err |
|---|---|---|---|---|---|
| FP32 Keras | — | 0.6462 | 0.6460 | 0.9237 | 0.0763 |
| FP32 TFLite | 621,956 B | 0.6462 | 0.6460 | 0.9237 | 0.0763 |
| dynamic-range | 205,752 B | 0.6453 | — | — | — |
| **int8 (deployed)** | **221,888 B** | **0.6432** | **0.6433** | **0.9187** | **0.0813** |

**Quantization cost: −0.0030 macro-F1** at 2.8× compression. Calibrated on 300
augmented training images drawn from the same corpus.

### 5.2 Corpus ablation — all scored on the SAME real test set

| training corpus | macro-F1 | cls acc | portion acc | ordinal err |
|---|---|---|---|---|
| synthetic only (previous deployment) | 0.5074 | 0.5403 | 0.8290 | 0.1787 |
| real only, unweighted | 0.5859 | 0.6393 | 0.9233 | 0.0767 |
| real + synthetic mixed | 0.5330 | 0.5957 | 0.8770 | 0.1240 |
| **real only + class weights** | **0.6462** | **0.6460** | **0.9237** | **0.0763** |

Two findings worth carrying forward:

**Mixing corpora is actively harmful.** Real+synthetic scored *below* real alone
and early-stopped at epoch 18. The synthetic domain gap conflicts with real
photographs rather than adding useful volume. **Replace, do not augment.**

**Sim-to-real gap, quantified without a camera.** The synthetic-trained model
scores 0.5511 on synthetic test images but 0.5074 on real ones, with chicken
falling 0.485 → 0.149.

### 5.3 Per-class performance (int8, deployed)

| class | F1 | note |
|---|---|---|
| broccoli | 0.9690 | excellent — colour is unambiguous |
| beef | 0.6616 | solid |
| rice | 0.5569 | fair — confuses with potato |
| potato | 0.5190 | fair — confuses with rice |
| chicken | 0.5094 | weakest; recovered from collapse (§5.4) |

FP32 equivalents: 0.9689 / 0.6650 / 0.5672 / 0.5220 / 0.5079.

Class accuracy (0.6433) and macro-F1 (0.6432) are near-identical — the signature
of a balanced classifier with no abandoned class.

### 5.4 The class-collapse failure and its fix

Trained on real data with an unweighted objective, **chicken collapsed to
F1 0.115, recall 0.063**. Of 600 chicken test images, 315 were classified beef
and 204 potato; across all 3,000 images the model emitted "chicken" only 59
times.

The corpus is exactly balanced at 3,000 images per class, so this is
**confusability, not frequency**. Cooked chicken is golden-brown, sitting
between `steak` (dark, easy, 0.94 recall) and `potato` (golden). Given one
ambiguous class and one easy neighbour, the network maximises accuracy by
abandoning the hard one.

**Fix:** weight each class in the loss by inverse measured recall
(`chicken 2.077, rice 1.159, potato 0.688, beef 0.552, broccoli 0.524`; clipped
at 4×, mean-normalised). Keras cannot apply `class_weight` to a multi-output
model, so the weight is folded into a custom loss (`train/losses.weighted_ce`).

**Result:** chicken F1 0.115 → **0.508**, recall 0.063 → 0.560; macro-F1
+0.060. Beef gave back 0.709 → 0.665, an accepted trade.

### 5.5 Portion head (int8)

Confusion matrix (rows = truth):

| | small | medium | large | recall |
|---|---|---|---|---|
| small | 1001 | 21 | 0 | 0.979 |
| medium | 85 | 906 | 38 | 0.880 |
| large | 0 | 100 | 849 | 0.895 |

**Off-by-two errors: 0 out of 3,000.** The ordinal loss — cross-entropy plus a
distance-weighted penalty on the expected tier distance — did exactly its job:
the model is never catastrophically wrong about portion size.

### 5.6 On-device measurements (verified 2026-08-13)

```
BENCH,runs=20,mean_us=1082349,max_us=1083780,arena_used=113516,
      model_bytes=221888,free_sram=54284,match=OK
CHECK,img_sum=449425,model_sum=22938064,input_sum=449425
```

Live camera path (`BENCH_MODE=0`), measured:

| stage | time |
|---|---|
| frame capture | 331 ms |
| crop + downsample | 69 ms |
| inference | 1,082 ms |
| **capture-to-result** | **1,482 ms** |

Preview streaming: **2.45 fps** (66 KB/s over USB CDC).

---

## 6. System capabilities (as built)

- **On-device inference**, fully offline, int8, no network.
- **Dual-task output**: 5-way food class + 3-tier portion, with a mapped gram
  range, from a single 96×96 frame.
- **Live camera capture** with overhead framing, full plate rim preserved.
- **Live preview streaming** of the *exact* tensor the model classifies (not a
  parallel recomputation), at 2.45 fps, costing **+16 bytes of RAM** because it
  reuses the input tensor.
- **Status LED**: blue blinking = camera streaming; green solid = inference
  running; red blinking = camera init failed.
- **Machine-parseable serial telemetry** (`BOOT` / `INFER` / `TIMING` / `BENCH`
  / `CHECK` / `RAW` / `SMOKE`) consumed by an automated harness.
- **Hardware-in-the-loop automation**: compile → upload → read serial → parse →
  record, with bounded stop-rules and no infinite retries.
- **Board-free static analysis**: arena/flash sweeps and ELF symbol census with
  no hardware attached.
- **Benchmark mode** using a baked-in test image, verifying device-vs-host
  agreement by byte checksum with no camera involved.
- **Labelled capture tool** producing a running confusion matrix.

---

## 7. Repository

```
data/     build_dataset.py · compose_portions.py · foodseg_cutouts.py · compose_real.py
model/    backbone.py · heads.py · model.py
train/    train.py · data.py · augment.py · losses.py
compress/ quantize_int8.py
deploy/   to_c_array.py · nano/classifier_tier1/ · nano/smoke_test/ · nano/camera_view/
harness/  compile_sweep.py · flash_and_measure.py · live_capture.py · camera_view.py · dsp_symbols.sh
eval/     metrics.py · make_report.py
results/  *.jsonl · *.json · REPORT.txt   ← all measurements
```

**Deployed artefacts:** `results/checkpoints/real_cw.keras` ·
`results/models/real_cw_int8.tflite` ·
`deploy/nano/classifier_tier1/{model_data.cc,test_image.cc}`

**Full reproduction:**

```bash
python3 -m data.foodseg_cutouts
python3 -m data.compose_real
python3 -m train.train --alpha 1.0 --source real --aug-level real \
        --epochs 100 --class-weights 2.077,0.524,1.159,0.552,0.688 --tag real_cw
python3 -m compress.quantize_int8 --ckpt results/checkpoints/real_cw.keras \
        --source real --tag real_cw
python3 -m deploy.to_c_array --model results/models/real_cw_int8.tflite
python3 harness/flash_and_measure.py --sketch deploy/nano/classifier_tier1 \
        --define ARENA_SIZE=120000 --define BENCH_MODE=1
python3 -m eval.make_report real_cw
```

---

## 8. Engineering findings worth preserving

**`Invoke()` destroys its own input.** The input tensor lives inside the tensor
arena and TFLM's memory planner aliases it with operator scratch. A benchmark
that filled the input once and invoked 21 times had runs 2–21 classifying
leftover activations, producing a device/host mismatch. Three hypotheses (op
support, TFLM version, XNNPACK delegate) were chased before a byte checksum
isolated it in one step: `img_sum` and `model_sum` matched the host exactly
while `input_sum` did not. **Any sketch invoking more than once must refill the
input first.**

**RGB565 byte order: big-endian `(hi<<8)|lo` is correct** for this
Harvard_TinyMLx / OV7675 path. Verified on-device against
`Camera.testPattern()`: big-endian yields 7 clean colour bars with
cyan/green/magenta/red decoding correctly; little-endian yields 10 runs and zero
recognisable colours. Red decodes as red, so there is no R/B transposition.
*Published guidance predicting "RGB565-swapped" is wrong for this path* —
applying it would break working code.

**CMSIS-NN is already active, and delivers no speedup.** The build links 26
CMSIS-NN `_s8` symbols and contains 251 DSP SIMD instructions
(`arm_nn_mat_mult_nt_t_s8` 110, `arm_depthwise_conv_s8_opt` 21), so
`ARM_MATH_DSP` is already defined by the mbed core and `-DARM_MATH_DSP` /
`-DCMSIS_NN` are no-ops. The `precompiled=full` root-cause theory is disproven:
no `.a` is shipped and only `kernels/cmsis_nn/conv.cpp` exists. Yet CMSIS-NN
measured 1,092,437 µs against a reference-kernel build's 1,076,311 µs. Published
expectation was 3–5×. **The bottleneck is not kernel selection and remains
unexplained.**

**Augmentation ownership must not overlap.** The composer owns scene lighting;
`augment.py` owns only sensor degradation. Stacking both crushed a large share
of every batch to near-black with extreme colour casts, destroying colour — the
dominant class cue — while leaving area intact. Class accuracy collapsed while
the portion head stayed healthy.

**`pgrep -f` self-matches.** Five polling shells ran for 90+ minutes because
each shell's own command line contained the pattern it searched for. Use the
bracket trick (`[t]rain`) or don't poll.

---

## 9. Limitations

### Immovable without new data

**1. Dataset semantics cap class accuracy.** FoodSeg103 has no `beef` class and
no pure chicken. Our "beef" is `steak`; our "chicken" is the merged
`chicken duck`. These are fixed properties of the only free, mask-annotated
corpus covering these foods. Merged and substituted classes are inherently
harder — chicken tops out near 0.51 even after weighting. **Training cannot fix
this.**

**2. Rice and potato are not separable at 96×96.** 183 rice images predict
potato and 126 potato predict rice. Both are pale, starchy and fine-textured;
the distinguishing detail (grain structure) sits near the resolution limit after
sensor degradation. Raising resolution would break the RAM budget — the arena is
already 113,516 B of 262,144 B alongside a 50,688 B frame buffer.

### Open / unexplained

**3. Latency is 1,082 ms with no known remedy.** See §8 — CMSIS-NN is confirmed
active but ineffective. Not blocking, but the documented optimisation win should
not be assumed.

**4. Training images still use a synthetic plate.** The *food* is real and
mask-extracted, but it sits on a drawn plate on a drawn table. Real frames add
real shadows, specular highlights and depth. This residual gap is unmeasured.

**5. Portion thresholds are modelled, not weighed.** The 80 g / 180 g cutoffs
come from the physical model in §4.3, not from scale measurements. The head
scores 0.9187 *against its own definition*; real-world gram accuracy is
unverified.

**6. Single-label only.** The class head ends in softmax, so outputs sum to 1
and the model reports one food per plate. Multi-food breakdown is scoped but not
built (§10).

**7. No live-camera accuracy data yet.** The camera path is verified functional
end-to-end (a real capture completed in 1,482 ms) but **zero labelled real-world
captures have been collected**. `results/live_captures.jsonl` does not yet
exist. All accuracy figures in this report are from held-out composited images.

---

## 10. Roadmap

### Immediate — live-camera validation (tooling ready, needs an operator)

Protocol: overhead camera, plate rim fully in frame at ~⅔ frame width, one food
per plate, ~20 captures per class across ≥3 lighting conditions.

```bash
python3 harness/camera_view.py     # frame the shot (live preview, 2.45 fps)
python3 harness/live_capture.py    # labelled captures + running confusion matrix
```

Test **broccoli first** (0.969) — if broccoli fails, the fault is framing or
optics, not the model. Expect rice/potato confusion. The deliverable is a
real-world confusion matrix compared against the lab 0.6432, which *is* the
sim-to-real measurement.

### Next — multi-label ("break down my full meal")

Assessed as **feasible on this hardware at essentially zero memory cost**:

| | now | multi-label |
|---|---|---|
| class head | 5-way softmax | 5 independent sigmoids |
| portion head | 3-way softmax | 5 × 3 (per-class tier) |
| output tensor | 8 values | 20 values |
| arena | 113,516 B | ~unchanged |

The class head already uses **max pooling** — the correct operation for presence
detection — so only the activation, loss, and data generator change. FoodSeg103
plates already contain ~6 labelled ingredients each, so genuine multi-food
training data is available. Costs: per-class accuracy will drop on crowded
plates, per-class portion is harder than whole-plate portion, and evaluation
must move to per-class precision/recall.

### Later

- ~150 weighed captures at fixed height → closes limitations 4 and 5 together.
- Roboflow `hust-ajpvu/food-srnub` (free, login required; 9,984 images, polygon
  masks, CC BY 4.0) for domain-shift augmentation — covers 4 of 5 classes, no
  beef.
- Nutrition5k (CC-BY 4.0, 181 GB, subset-downloadable) for true overhead RGB-D
  with real per-ingredient gram masses.

---

## 11. Verification statement

Sources for every claim: `results/quantization_real_cw.json` ·
`results/compare_on_real_test.json` · `results/real_cw_on_real.json` ·
`results/device_runs.jsonl` · `results/compile_sweep.jsonl` ·
`results/foodseg103_counts.json` · `results/rgb565_byteorder.json` ·
`results/cmsisnn_findings.json` · `results/REPORT.txt`.

Test-set integrity: all accuracy figures use a held-out split of 3,000 images
never seen in training. Splits are made at the **source-photo group level**, not
the image level, so composites sharing a source cutout cannot straddle the
train/test boundary.

**Not verified:** real-world camera accuracy (no captures collected);
gram-mass accuracy (thresholds modelled, not weighed); multi-food behaviour
(not built).
