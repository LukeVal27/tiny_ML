# EE446 TinyML — Food Detection & Portion Estimation

Tier 1 deliverable: a 5-class food classifier plus a 3-tier ordinal portion head,
running int8 on an **Arduino Nano 33 BLE Sense (nRF52840)** with an **OV7675**
camera on the Tiny ML shield. See `handoff.md` for the full brief.

## Standing instructions

**Edit, don't heredoc.** Use the `Edit` tool for changes to existing files.
Never patch with `python3 - <<'EOF'`, `sed -i`, or shell redirection — each one
triggers a system-reminder that echoes the whole file back. `Write` is for new
files only. See the `token-discipline` skill.

**Filter every subprocess.** `arduino-cli`, training runs, and dataset tooling
all emit far more than you need. Pipe through `grep`/`tail` at the call site.
Never print a dataset's full label list or feature schema.

**Reuse measured numbers.** Everything in `results/` is a result. Do not re-run
training, quantization, or a flash cycle to confirm a number already logged
there. The key figures are tabulated in `README.md` and `results/REPORT.txt`.

**Cap debug loops.** Hardware debugging is capped at ~3 flash cycles; then
summarize the evidence and ask how to proceed. Ruled-out hypotheses stay ruled
out — record them rather than re-testing.

**Long jobs run in the background**, redirected to a file in `results/`, and are
inspected with `grep`/`tail`. Never stream full training output.

**Do not write polling loops to wait for background jobs.** The harness
re-invokes you when a background task finishes. Five `until ! pgrep -f
"train.train"; do sleep 30; done` shells were left spinning for over 90 minutes
because `pgrep -f` matches the *full command line*, so each shell matched its own
command string and could never see its condition become true. They also matched
each other. Each stuck loop additionally burned a 600 s tool timeout. If you ever
genuinely must match a process name, use the bracket trick — `pgrep -f
"[t]rain\.train"` — so the pattern cannot match itself.

## Relevant skills

- `arduino-hitl` — flashing, serial timing, arena sizing, board facts. Load
  before any compile/upload/serial work.
- `token-discipline` — editing, dataset queries, cheap bisection, log handling.

## Data corpora (pick with `--source`)

| source | dir | what it is |
|---|---|---|
| `synthetic` | `data/composed` | Food-101 texture inside a drawn elliptical blob. 18,000 plates. The original corpus; macro-F1 0.5493. |
| `real` | `data/foodseg103_real` | FoodSeg103 **mask cutouts** — real food outlines and structure — on the same synthetic plate. |
| `both` | both of the above | concatenation |

`data/foodseg103_cutouts/` holds the 4,996 extracted RGBA cutouts (alpha = the
real mask). Built by `python3 -m data.foodseg_cutouts` (single pass: counts, area
diagnostics and extraction together; rows filtered on `classes_on_image` before
any image decode). Counts live in `results/foodseg103_counts.json`.

Both corpora share `draw_plate()` in `data/compose_portions.py`, so plate
geometry — including `PLATE_DIAM_JITTER` — is identical and the portion labels
stay exact and comparable.

Augmentation level `real` = geometry + sensor degradation only, no photometric
jitter: FoodSeg103 images are real photographs that already carry real scene
lighting, and stacking jitter on top is the same failure that once crushed
batches to black.

## Repo map

```
data/      build_dataset.py (HF pull -> 5 classes), compose_portions.py (synthetic plates)
model/     backbone.py (depthwise-separable), heads.py (dual pooling), model.py
train/     train.py, data.py, augment.py, losses.py (ordinal)
compress/  quantize_int8.py (PTQ + representative dataset)
deploy/    to_c_array.py, nano/smoke_test/, nano/classifier_tier1/
harness/   compile_sweep.py (no board needed), flash_and_measure.py (HITL)
eval/      metrics.py, make_report.py
results/   *.jsonl, *.json, REPORT.txt  <- measured results, reuse these
```

## Hard-won design decisions (do not silently revert)

- **The two heads pool differently.** Class head uses max pooling (presence);
  portion head uses average pooling (extent). Food covers a minority of the
  frame, so averaging dilutes the class signal. Sharing one GAP cost ~20 points
  of validation accuracy.
- **Use `MaxPooling2D`/`AveragePooling2D` sized to the grid, not the `Global*`
  variants.** Same maths, but exports `MAX_POOL_2D`/`AVERAGE_POOL_2D` instead of
  `REDUCE_MAX`/`MEAN`, which are far better tested in TFLM.
- **Export from a batch-1 clone** (`model.to_batch1`). A `None` batch makes Keras
  emit `SHAPE`/`STRIDED_SLICE`/`PACK` to compute sizes at runtime; dynamic shapes
  are what TFLM handles worst.
- **The deploy sketch downsamples, it does not centre-crop.** The portion tier is
  defined relative to the plate rim, so the whole plate must stay in frame. The
  stock `person_detection` example crops and would break the portion head.
- **Bind output tensors by width** (5 = class, 3 = portion), never by index — the
  converter does not promise ordering.
- **The composer owns scene lighting; `augment.py` owns only sensor
  degradation.** Stacking both crushed images to black and destroyed the colour
  cue that separates the classes.
- **Class list is deliberately narrow** (`chicken_wings`, `risotto`+`fried_rice`,
  `steak`+`filet_mignon`+`prime_rib`, `french_fries`). Broader dish mappings made
  the classes overlap in colour and drove chicken and rice to F1 = 0.000.

## Current state (all figures verified from `results/`, 2026-08-12)

Tier 1 is **deployed and hardware-verified**, trained on REAL food.
See `STATUS.md` for the full picture. Headline: class macro-F1
**0.6432 int8** on the real test set, up from 0.5074, at identical size,
arena and latency. Deployed artifacts are the `real_cw` tag.
Tier 2 was not attempted.

**Deployed artifacts** (these exact paths):
- checkpoint `results/checkpoints/real_cw.keras`
- int8 model `results/models/real_cw_int8.tflite` — **221,888 B (216.7 KB)**
- C arrays `deploy/nano/classifier_tier1/model_data.cc` + `test_image.cc`
- sketch `deploy/nano/classifier_tier1/classifier_tier1.ino`
- metrics `results/quantization_real_cw.json`, `results/REPORT.txt`, `STATUS.md`

**Accuracy** — measured on the REAL test set (n = 3,000). Note the benchmark
changed: the old synthetic-corpus figures (macro-F1 0.5493) are NOT comparable,
because they were scored on synthetic test images. On the real test set the old
model scores 0.5074.

| model | macro-F1 | acc | portion | ordinal err |
|---|---|---|---|---|
| old, synthetic-trained | 0.5074 | 0.5403 | 0.8290 | 0.1787 |
| **real_cw, int8 (deployed)** | **0.6432** | 0.6433 | 0.9187 | 0.0813 |
| real_cw, FP32 | 0.6462 | 0.6460 | 0.9237 | 0.0763 |

Quantization cost: **−0.0030 macro-F1**. Per-class F1 (FP32): broccoli 0.969,
beef 0.665, rice 0.567, potato 0.522, chicken 0.508.

**Class weights are required.** Without them chicken collapses to F1 0.115 /
recall 0.063 (absorbed by beef and potato). The corpus is balanced, so this is
confusability, not frequency. Use
`--class-weights 2.077,0.524,1.159,0.552,0.688` (inverse recall, clipped at 4×,
mean-normalised, in `CLASSES` order).

**Training on `both` is worse than `real`** (0.5330 vs 0.5859) and early-stops.
Replace the synthetic corpus, do not augment with it.

**On-device** (`ARENA_SIZE=120000`, `BENCH_MODE=1`, 20 runs, 2026-08-13):
`arena_used=113,516` · `mean_us=1,082,349` (**1.082 s**) · sketch flash
**500,648 B (50.9%)** · static RAM **171,048 B** · largest free block **54,284 B**
· `match=OK`.

**RAM gate**, on hardware: RGB565 QCIF frame **50,688 B** + 90,000 B arena,
`cam=OK`, **56,988 B** still allocatable. Static sweep gave fixed overhead
**51,792 B**; grayscale ceiling ~160 KB (linker overflow at 190,000), RGB565
ceiling ~135 KB.

**Dataset**: `data/raw` 7,031 images — train `chicken 749, broccoli 975,
rice 1400, beef 1400, potato 750`; test `250 / 325 / 466 / 466 / 250`.
`data/composed` holds **18,000** synthetic plates — the ORIGINAL corpus, now
superseded. `data/foodseg103_real` holds **18,000** real-cutout composites and is
the set the deployed model is actually trained on. `data/composed_v4` also holds
18,000 plates but is **unused** — it adds a saturation-based "foodiness" crop
filter that measured as neutral (colour-histogram linear probe 0.453 vs 0.460),
so it was not adopted.

## Solved: the device/host class-head mismatch

Symptom: device predicted a different class from the host on a byte-identical
input, while the portion head agreed.

Root cause: **the input tensor lives inside the tensor arena, and TFLM's memory
planner aliases that buffer with scratch space once the first operator consumes
it — `Invoke()` destroys its own input.** The benchmark filled the input once and
then invoked 21 times, so runs 2–21 classified leftover activations.

Found by checksumming on-device: `img_sum` and `model_sum` matched the host
exactly (−126,959 and 22,952,540) while `input_sum` did not (−1,081,352). After
refilling the input before every `Invoke()`, `input_sum == img_sum == -126,959`
and `match=OK`.

**Any sketch that invokes more than once must refill the input each time** — see
`fillInput()` in `classifier_tier1.ino`. Ruled out and not to be re-tested: TFLM
runtime version (two versions byte-identical) and the XNNPACK delegate (two host
kernel paths agree).

## Settled hardware questions — do NOT re-open

**RGB565 byte order: the current code is CORRECT.** `unpack565()` uses big-endian
`(hi << 8) | lo` and that is right for the Harvard_TinyMLx / OV7675 path.
Verified on-device with `Camera.testPattern()`: big-endian yields 7 clean colour
bars with cyan/green/magenta/red decoding correctly, little-endian yields 10 runs
and **zero** recognisable colours. Red decodes as red, so there is no R/B
transposition either.

`RESEARCH_FINDINGS.md` Q5 predicted the opposite ("RGB565-swapped", try
`(lo << 8) | hi`). **That prediction is wrong for this hardware path.** Applying
it would break a working decode. Evidence in `results/rgb565_byteorder.json`.

**CMSIS-NN is already enabled and active.** The `TFLM_New` build links 26 CMSIS-NN
`_s8` symbols and contains 251 DSP SIMD instructions (`arm_nn_mat_mult_nt_t_s8`
alone has 110, `arm_depthwise_conv_s8_opt` has 21). So `ARM_MATH_DSP` is already
defined by the mbed cortex-m4 core, and `-DARM_MATH_DSP` / `-DCMSIS_NN` would be
no-ops. `RESEARCH_FINDINGS.md` Q6's `precompiled=full` root cause is **disproven**:
no `.a` is shipped, the build logs "Precompiled library not found", and only
`kernels/cmsis_nn/conv.cpp` exists (no reference duplicate to remove).

Despite this, CMSIS-NN measured 1,092,437 us against the reference-kernel build's
1,076,311 us — no speedup. The bottleneck is therefore NOT kernel selection, and
remains unexplained. Evidence in `results/cmsisnn_findings.json`; census tool is
`harness/dsp_symbols.sh <elf>`.

## Known limitation

Class macro-F1 is ~0.55, and the ceiling is set by the **source data**, not the
model: a colour-histogram linear probe reaches only ~0.51 on the raw
full-resolution photos and ~0.46 on the composed plates, so composition costs
only ~0.05. Improving this needs deployment-domain images (Roboflow with boxes,
or real captures from the OV7675), not a bigger network.
