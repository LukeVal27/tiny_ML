# Food Detection & Portion Estimation on an Arduino Nano 33 BLE Sense

EE 446 TinyML final project, Summer 2026 — Luke Valerio · Daniel Yang.

Two predictions from one camera frame — **what food** (5 classes) and **how
much** (3 ordinal portion tiers) — running fully on-device in int8 on an
nRF52840 with an OV7675 camera. No network, no companion phone, no cloud
inference.

## Headline results

| | macro-F1 | class acc | portion acc |
|---|---|---|---|
| float32 baseline, held-out test (n = 3,000) | 0.6462 | 0.6460 | 0.9237 |
| **int8 PTQ, deployed** (n = 3,000) | **0.6432** | **0.6433** | **0.9187** |
| **live OV7675 captures** (n = 75) | **0.6676**¹ | **0.6667** | see below |

¹ over the four classes with on-device support; potato was never captured.

Quantization cost **−0.0030 macro-F1** at **2.80×** compression
(621,956 B → 221,888 B). Zero off-by-two portion errors in 3,000 test images.

**On-device measurements** (`results/device_runs.jsonl`, 20-run mean):

| | |
|---|---|
| model size | 221,888 B (216.7 KB) |
| tensor arena | 113,516 B |
| flash | 500,648 B (50.9% of 983,040) |
| static RAM | 222,688 B of 262,144 |
| inference | 1,082 ms |
| capture-to-result | 1,482–1,733 ms |

**Per-class, live camera** (`results/live_captures.jsonl`, session `final`):

| class | samples | correct | recall | F1 |
|---|---|---|---|---|
| beef | 23 | 17 | 0.739 | 0.850 |
| chicken | 20 | 6 | 0.300 | 0.387 |
| broccoli | 19 | 15 | 0.789 | 0.833 |
| rice | 13 | 12 | 0.923 | 0.600 |
| potato | 0 | — | food not available | |

## Model

157,512 parameters · 13.58 M MACs per 96×96 frame · depthwise-separable
backbone with 6 inverted-residual blocks and 3 residual adds.

```
96²×3 → stem s2 → 48²×16 → b1–b2 → 24²×32 → b3–b4 → 12²×64
      → b5–b6 → 6²×128 → 1×1 widen → 6²×192
                        ├── MAX pool 6×6 → 5 classes  (965 params)
                        └── AVG pool 6×6 → 3 tiers    (579 params)
```

155,968 parameters are shared; the two heads cost 1,544 between them. **The
heads pool differently on purpose** — max for presence, average for extent. A
single shared global-average pool stalled the classifier at ~0.34 validation
accuracy. See `CLAUDE.md` for the full list of load-bearing design decisions.

## Datasets — not committed, rebuilt by script

The corpora total ~2.2 GB and are **generated locally**, not stored here. All
sources are public:

| dataset | link | used for |
|---|---|---|
| **FoodSeg103** | https://huggingface.co/datasets/EduardoPacheco/FoodSeg103 | **the deployed model.** Apache-2.0. Mask cutouts → 18,000 composited plates. |
| Food-101 | https://huggingface.co/datasets/ethz/food101 | the earlier synthetic corpus, since superseded |
| ImageNet broccoli subset | https://huggingface.co/datasets/mlnomad/imnet1k_broccoli | broccoli class for the synthetic corpus |

Class mapping is lossy and this is load-bearing: FoodSeg103 has **no beef and
no pure chicken**, so beef is substituted from `steak` (46) and chicken is the
merged `chicken duck` (48) label. Full mapping in
`results/foodseg103_mapping.json`.

Portion labels are **constructed, not estimated** — food is pasted at a sampled
fraction `f` of plate area, so `mass ≈ f × 513 g` is exact by definition
(22 cm plate, 1.5 cm deep, 0.9 g/cm³). Plate apparent diameter is jittered to
62–92% of frame width so raw pixel area is an ambiguous cue.

## Reproducing the results

Requires Python 3.11+, TensorFlow 2.20, and `arduino-cli` with the
`arduino:mbed_nano` core and the `Harvard_TinyMLx` library.

```bash
# 1. build the corpus (~30 min, downloads FoodSeg103)
python3 -m data.foodseg_cutouts          # -> 4,996 RGBA cutouts
python3 -m data.compose_real             # -> 18,000 composited plates

# 2. train (class weights are REQUIRED -- see below)
python3 -m train.train --alpha 1.0 --source real --aug-level real \
        --epochs 100 --class-weights 2.077,0.524,1.159,0.552,0.688 --tag real_cw

# 3. quantize to int8 and export
python3 -m compress.quantize_int8 --ckpt results/checkpoints/real_cw.keras \
        --source real --tag real_cw
python3 -m deploy.to_c_array --model results/models/real_cw_int8.tflite
python3 -m eval.make_report real_cw

# 4. deploy and measure on hardware
python3 harness/compile_sweep.py                      # RAM sweep, no board needed
python3 harness/flash_and_measure.py --sketch deploy/nano/classifier_tier1 \
        --define ARENA_SIZE=120000 --define BENCH_MODE=0

# 5. evaluate on live camera input
python3 harness/camera_view.py                        # live dashboard + guide ring
python3 harness/live_capture.py --session final       # labelled capture session
python3 harness/live_capture.py --summary --session final
```

**Class weights are required, and not for the usual reason.** The corpus is
perfectly balanced at 3,000 images per class, yet without weighting chicken
collapses to F1 0.115 / recall 0.063 — it is absorbed by beef and potato. This
is *confusability*, not frequency. The weights are inverse per-class recall from
a prior run, clipped at 4× and mean-normalised. Keras cannot apply
`class_weight` to a multi-output model, so the weight is folded into the loss
(`weighted_ce` in `train/losses.py`).

## Layout

```
data/       build_dataset.py (Food-101 -> 5 classes), compose_portions.py,
            compose_real.py, foodseg_cutouts.py   [corpora are gitignored]
model/      backbone.py (depthwise-separable), heads.py (dual pooling), model.py
train/      train.py, data.py, augment.py, losses.py (ordinal + weighted CE)
compress/   quantize_int8.py (PTQ + representative dataset)
deploy/     to_c_array.py, nano/classifier_tier1/ (deployed sketch),
            nano/camera_raw/, nano/camera_view/, nano/smoke_test/
harness/    flash_and_measure.py (HITL), compile_sweep.py (no board),
            live_capture.py, camera_view.py, camera_raw.py,
            portion_probe.py, make_confusion.py, build_deck.py
eval/       metrics.py, make_report.py
results/    measured outputs -- *.json, *.jsonl, REPORT.txt, figures/,
            checkpoints/, models/
docs/       evidence/ (on-device photos), archive/ (superseded working notes)
```

## Deployed artifacts

- checkpoint — `results/checkpoints/real_cw.keras`
- int8 model — `results/models/real_cw_int8.tflite` (221,888 B)
- C arrays — `deploy/nano/classifier_tier1/model_data.cc`, `test_image.cc`
- sketch — `deploy/nano/classifier_tier1/classifier_tier1.ino`
- metrics — `results/quantization_real_cw.json`, `results/REPORT.txt`

## Two traps worth knowing

**`Invoke()` destroys its own input.** TFLM's memory planner aliases the input
tensor with arena scratch, so any sketch invoking more than once must refill the
input first. This produced a device/host mismatch that survived three wrong
hypotheses before an on-device checksum isolated it. See `fillInput()`.

**RGB565 byte order is big-endian `(hi << 8) | lo`** on the
Harvard_TinyMLx/OV7675 path, verified twice on-device. Evidence in
`results/rgb565_byteorder.json`.

## Known limitations

1. **FoodSeg103 has no beef and no pure chicken** — merged and substituted
   labels cap chicken near F1 0.51. Training cannot fix this.
2. **Rice and potato are not separable at 96×96** — grain structure sits at the
   resolution limit, and higher resolution breaks the RAM budget.
3. **The portion head is degenerate on real camera input** — it emitted `large`
   on all 75 live captures at exactly 0.9961 (255/256, the int8 softmax
   ceiling), never producing another tier. Host diagnosis
   (`harness/portion_probe.py`) rules out white balance, contrast and mean/std
   matching; a scale sweep shows the head is correct from 0.45× to 1.35× plate
   zoom. Unresolved.
4. **Portion tiers are modelled, not weighed** — 0.9187 is scored against our
   own definition of a tier; real gram accuracy is unverified.
5. **Single-label only** — the class head ends in softmax, so it reports one
   food per plate.
6. **1,082 ms inference has no known remedy.** CMSIS-NN is confirmed active
   (26 int8 symbols, 251 DSP SIMD instructions) and measured no speedup.

## Authorship

All project work — model design, training, quantization, hardware bring-up,
data collection and on-device evaluation — was carried out by Luke Valerio and
Daniel Yang. Claude (an AI assistant) was used for code assistance, analysis
and preparing this repository.
