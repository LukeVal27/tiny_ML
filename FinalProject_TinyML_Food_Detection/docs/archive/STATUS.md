# Project status — 2026-08-13

Every number here is measured and logged in `results/`. Nothing is estimated.

**Bottom line:** Tier 1 is deployed and hardware-verified, now trained on real
food. Class macro-F1 improved **0.5074 → 0.6432** on the real test set (+0.136,
+27% relative) at identical model size, arena and latency.

---

## 1. Deployed model

| item | value |
|---|---|
| checkpoint | `results/checkpoints/real_cw.keras` |
| int8 model | `results/models/real_cw_int8.tflite` — **221,888 B** |
| C array | `deploy/nano/classifier_tier1/model_data.cc` |
| sketch | `deploy/nano/classifier_tier1/classifier_tier1.ino` |
| architecture | 157,512 params, 96×96×3 RGB, dual-pooling heads (unchanged) |
| trained on | `data/foodseg103_real` — FoodSeg103 mask cutouts on synthetic plate |

### Accuracy (real test set, n = 3,000)

| model | macro-F1 | acc | portion | ordinal err |
|---|---|---|---|---|
| old, synthetic-trained | 0.5074 | 0.5403 | 0.8290 | 0.1787 |
| real, unweighted | 0.5859 | 0.6393 | 0.9233 | 0.0767 |
| real + synthetic (`both`) | 0.5330 | 0.5957 | 0.8770 | 0.1240 |
| **real + class weights (FP32)** | **0.6462** | 0.6460 | 0.9237 | 0.0763 |
| **real + class weights (int8, deployed)** | **0.6432** | 0.6433 | 0.9187 | 0.0813 |

int8 costs **−0.0030** macro-F1. Per-class F1 (FP32): broccoli 0.969, beef 0.665,
rice 0.567, potato 0.522, chicken 0.508.

### On-device (measured 2026-08-13, `BENCH_MODE=1`)

`arena_used` **113,516 B** · latency **1,082 ms** · sketch flash **500,648 B
(50.9%)** · static RAM 171,048 B · largest free block 54,284 B ·
**device/host agreement OK** · `img_sum == input_sum == 449,425`.

Arena and flash are unchanged from the previous model because the architecture
was deliberately held constant, so every earlier hardware measurement remains
valid.

---

## 2. What changed this session

**Real data replaced synthetic.** FoodSeg103 (`EduardoPacheco/FoodSeg103`,
Apache-2.0) gave per-ingredient pixel masks. 4,996 cutouts were extracted with
the mask as alpha, then composited onto the *same* synthetic plate at controlled
area fractions — so the food's outline and texture are real while portion labels
stay exact by construction.

**Class weighting fixed a collapse.** Trained on real data alone, chicken
collapsed to F1 0.115 / recall 0.063: 315 of 600 chicken images were predicted
beef and 204 potato, with only 59 of 3,000 predictions being "chicken" at all.
The corpus was perfectly balanced, so this was confusability, not frequency —
cooked chicken is golden-brown, sitting between `steak` (easy, 0.94 recall) and
`potato`. Weighting the loss by inverse recall (`chicken 2.077, rice 1.159,
potato 0.688, beef 0.552, broccoli 0.524`) recovered chicken to **F1 0.508 /
recall 0.560** and lifted macro-F1 by +0.060. Beef gave back 0.709 → 0.665, an
accepted trade.

**Mixing corpora hurt.** `both` (synthetic + real) scored 0.5330 against real's
0.5859 and early-stopped at epoch 18. This answers the A4 replace-vs-augment
gate: **replace**. The synthetic domain gap conflicts with real data rather than
adding useful volume.

**Sim-to-real gap quantified without a camera.** The old model scored 0.5511 on
synthetic but 0.5074 on real, with chicken falling 0.485 → 0.149.

---

## 3. Settled — do not re-investigate

**RGB565 byte order: current code is correct.** `unpack565()`'s big-endian
`(hi<<8)|lo` verified on-device via `Camera.testPattern()`: big-endian gives 7
clean colour bars with cyan/green/magenta/red decoding correctly; little-endian
gives 10 runs and zero recognisable colours. Red decodes as red, so there is no
R/B transposition. `RESEARCH_FINDINGS.md` Q5 predicted the opposite and is
**wrong for this hardware path** — applying it would break working code.
Evidence: `results/rgb565_byteorder.json`.

**CMSIS-NN is already active.** 26 CMSIS-NN `_s8` symbols linked, 251 DSP SIMD
instructions (`arm_nn_mat_mult_nt_t_s8` 110, `arm_depthwise_conv_s8_opt` 21). So
`ARM_MATH_DSP` is already defined and the proposed flags are no-ops.
`RESEARCH_FINDINGS.md` Q6's `precompiled=full` root cause is **disproven**: no
`.a` is shipped, the build logs "Precompiled library not found", and only
`kernels/cmsis_nn/conv.cpp` exists. Evidence: `results/cmsisnn_findings.json`;
tool: `harness/dsp_symbols.sh <elf>`.

**`Invoke()` destroys its own input.** The input tensor is aliased with arena
scratch, so any sketch invoking more than once must refill first — see
`fillInput()`. This caused a device/host mismatch that cost three wrong
hypotheses before a byte checksum isolated it.

---

## 4. Immovable roadblocks

**1. Class accuracy is bounded by dataset semantics, not by the model.**
FoodSeg103 has no `beef` class, so our "beef" is `steak` (46), and no pure
chicken, so our "chicken" is the merged `chicken duck` (48). Those are fixed
properties of the only free, mask-annotated corpus that covers these foods. The
merged/substituted classes are inherently harder, and chicken remains the weakest
class at 0.508 even after weighting. **This cannot be fixed by training.** It
needs either a differently-labelled dataset or our own captures.

**2. Rice and potato confuse each other and cannot be fully separated at 96×96.**
183 rice images predict potato and 126 potato predict rice. Both are pale,
fine-textured and starchy; at 96×96 after sensor degradation, the distinguishing
detail (grain structure) is near the resolution limit. Raising input resolution
would break the RAM budget — the arena is already 113,516 B of 262,144 B SRAM
with a 50,688 B frame buffer.

**3. Latency is 1.08 s and the cause is unknown.** CMSIS-NN is confirmed active
yet measured *no* speedup over reference kernels (1,092,437 µs vs 1,076,311 µs).
The published expectation was 3–5×. Since kernel selection is ruled out, the
bottleneck is elsewhere and unexplained. Not blocking — but do not expect the
documented CMSIS-NN win to materialise.

**4. No true deployment-domain data exists.** Training images are real food on a
*drawn* plate on a *drawn* table. Real frames will add real shadows, specular
highlights on the plate, and depth. That gap is unmeasured and can only be closed
with camera captures.

**5. Portion thresholds are modelled, not measured.** The 80 g / 180 g cutoffs
come from `mass ≈ f × plate_area × depth × density` (≈513 g per unit
area-fraction), not from weighed food. The portion head scores 0.9187 *against
its own definition*; real-world accuracy is unverified.

---

## 5. Next steps

1. **Live camera** (`BENCH_MODE=0`) — the only untested path. Start with
   **broccoli** (0.969); if broccoli fails, the fault is the pipeline, not the
   model. Byte order is already settled, so colour should be correct.
2. **~150 weighed captures** at fixed height with the plate rim in frame →
   closes roadblocks 4 and 5 simultaneously.
3. Optional: Roboflow `hust-ajpvu/food-srnub` for domain-shift augmentation
   (needs a free login; no beef class).

## Reproduce

```bash
python3 -m data.foodseg_cutouts                      # -> 4,996 cutouts
python3 -m data.compose_real                         # -> 18,000 real composites
python3 -m train.train --alpha 1.0 --source real --aug-level real \
        --epochs 100 --class-weights 2.077,0.524,1.159,0.552,0.688 --tag real_cw
python3 -m compress.quantize_int8 --ckpt results/checkpoints/real_cw.keras \
        --source real --tag real_cw
python3 -m deploy.to_c_array --model results/models/real_cw_int8.tflite
python3 harness/flash_and_measure.py --sketch deploy/nano/classifier_tier1 \
        --define ARENA_SIZE=120000 --define BENCH_MODE=1
python3 -m eval.make_report real_cw
```
