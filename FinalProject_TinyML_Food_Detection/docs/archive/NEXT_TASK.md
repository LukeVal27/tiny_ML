# NEXT TASK — FoodSeg103 real-data integration + two HW unblocks

Context: research is done (see RESEARCH_FINDINGS.md). Class macro-F1 is ceilinged
by synthetic source data (~0.55). The fix is real deployment-domain images with
real masks. FoodSeg103 is the chosen first integration. Two hardware items
(RGB565 byte order, CMSIS-NN) are unblocked and cheap; do them in parallel with
data work since they're independent.

Follow all CLAUDE.md standing instructions: Edit-don't-heredoc, filter every
subprocess, reuse measured numbers in `results/`, cap debug at ~3 flash cycles,
long jobs to background files, NO polling loops.

---

## TASK A — FoodSeg103 ingestion (highest priority)

**Goal:** replace synthetic portion labels with real per-image food-area
fractions derived from real segmentation masks, on real plated food.

**A0. Decide + log the class mapping (blocking, do first).**
FoodSeg103 does not map 1:1 to our five classes. Confirmed mapping to use:
- broccoli → `broccoli` (idx 87)
- rice → `rice` (idx 66)
- potato → `potato` (idx 70)   [decide separately whether `french fries` idx 3 folds in — default NO]
- chicken → `chicken duck` (idx 48)  [merged w/ duck; accept it]
- beef → `steak` (idx 46)  [no beef class exists; this is a deliberate substitution]
Write this mapping as a constant dict in `data/build_dataset.py` with a comment
explaining each caveat. Log the decision to `results/` (one line). Do NOT silently
pick a different mapping later.

**A1. Pull + count (one pass, background, filtered).**
Add a FoodSeg103 path to `data/build_dataset.py` using
`datasets.load_dataset("EduardoPacheco/FoodSeg103")` (ungated, Apache-2.0, no
login). In the SAME pass compute exact per-class image counts over
`classes_on_image` for indices {46,48,66,70,87} (+ optionally 3). Write counts to
`results/foodseg103_counts.json`. Run in background, redirect to
`results/foodseg103_ingest.log`, inspect with `tail`/`grep`. Do not print the full
label list. (Counts are UNVERIFIED in research — this pass is the source of truth.)

**A2. Derive portion labels from masks.**
For each image containing one of our classes, compute food-area fraction =
(pixels == class_idx) / (non-background pixels). This is the real replacement for
the synthetic elliptical-blob portion labels. Map fraction → tier with thresholds
chosen to reproduce the small/medium/large split; start from the current
composer's tier boundaries and log the mapping. This is per-image, per-class.

**A3. Reuse the composer as augmentation only — do NOT replace it yet.**
Per CLAUDE.md design decisions, the composer owns scene lighting and `augment.py`
owns sensor degradation. Feed real FoodSeg103 crops through the SAME sensor-
degradation path (not the lighting composer) to match deployment. Keep
`data/composed` intact; write real-data output to a new `data/foodseg103_real/`
so nothing measured is overwritten. Do not retrain on it until A1/A2 counts are
sane (enough support per class).

**A4. Gate before training.**
If any of {steak, chicken duck, rice, broccoli, potato} has too little support
(check the counts), flag it and STOP for a decision rather than training a
degenerate class (recall chicken/rice went to F1=0.000 before). Report counts +
proposed train/test split; ask how to proceed.

**Deliverable A:** `results/foodseg103_counts.json`, updated `build_dataset.py`
with logged mapping, `data/foodseg103_real/` populated, a short written summary
of counts + tier-threshold choice. No full-training run until gated.

---

## TASK B — RGB565 byte order (1 flash cycle)

**Finding:** `Arduino_OV767X::readFrame()` does NO byte-swap — it stores bytes in
sensor-clock order. The OV7670/OV7675 RGB565 output is "RGB565-swapped" relative
to our current `(hi<<8)|lo` big-endian assumption. Our sketch is very likely
decoding with the wrong endianness.

**B1.** Capture `Camera.testPattern()` output once (already have a capture path).
Decode the SAME raw bytes on host TWO ways: `(hi<<8)|lo` vs `(lo<<8)|hi`. Compare
to the expected color-bar/zig-zag reference. Pick whichever gives correct bar
colors.

**B2.** If R/B still transposed after fixing endianness, swap the R and B mask
extractions (see RESEARCH_FINDINGS.md Q5 code). Change endianness FIRST, re-check,
THEN R/B — never both in one cycle.

**B3.** Log the winning combo to `results/` and mark the alternative ruled-out
(don't re-test). Update `fillInput()`/decode in `classifier_tier1.ino`.

This is ≤1 flash cycle (host-side comparison of one captured frame). Cite:
OV767X.cpp readFrame (no-swap) + Adafruit RGB565-swapped doc.

---

## TASK C — CMSIS-NN enable (compile-only first, no board needed)

**Finding:** the `tflite-micro-arduino-examples` `library.properties` has
`precompiled=full` → Arduino links a PREBUILT archive and may never compile the
cmsis_nn sources you added. That explains "compiled against the lib, no speedup."
Repo is also archived read-only (2025-02-24).

**C1. Diagnose without a board** (use `harness/compile_sweep.py`):
Grep the build/link artifacts for `arm_convolve_s8` / `arm_depthwise_conv_s8`
symbols. Absent → CMSIS-NN never compiled. Confirm whether a prebuilt `.a` is
being linked because of `precompiled=full`.

**C2. Enable path** (in order, stop when symbols appear):
1. Set `precompiled=false` in the library's `library.properties`; ensure cmsis_nn
   kernel `.cc` present AND their reference-kernel duplicates removed from `src/`
   (no duplicate symbols).
2. Force the DSP macro at compile:
   ```
   arduino-cli compile -b arduino:mbed_nano:nano33ble \
     --build-property "compiler.cpp.extra_flags=-DARM_MATH_DSP -DCMSIS_NN" \
     --build-property "compiler.c.extra_flags=-DARM_MATH_DSP" \
     <sketch>
   ```
3. If you hit `undefined reference to __sxtb16 / __smlabb / __smlatt` — that's
   TFLM issue #2331 and it means the optimized path IS being pulled in; fix by
   aligning CMSIS include paths with the mbed_nano core's CMSIS (or use the
   Gostas fork referenced in #2331).

**C3. Fallback decision (raise, don't silently switch):** if the precompiled
archive can't be dislodged cleanly, the realistic alternative is a from-source
TFLM build or an Edge-Impulse-exported library that ships CMSIS-NN correctly
wired. This conflicts with "adapt not replace" — so flag it and ask before
swapping runtimes.

**C4. Expected result:** depthwise-separable int8 @ 96×96 on M4F@64MHz →
~3–5× overall (pointwise ~6–7×, depthwise ~2.2×), i.e. ~1,076 ms → ~250–350 ms.
Measure with existing `BENCH_MODE=1`, 20 runs, and remember the input-refill
requirement (`fillInput()` before every `Invoke()`). Log new `mean_us` to
`results/`; compare against the logged 1,076,311 µs. Don't re-flash to reconfirm
an already-logged number.

C1 is free (no board). Do C1 before spending any flash cycle on C2.

---

## Ordering
- A0 → A1 (background) can start immediately.
- B and C1 are independent and cheap; run alongside A while the ingest job runs.
- Do NOT write polling loops for the A1 background job — the harness re-invokes on
  completion.
- Gate at A4, B3, C3 for human decisions.

## What NOT to do
- Don't guess FoodSeg103 counts — compute them (A1).
- Don't retrain before A4 gate.
- Don't change byte order and R/B in the same flash cycle.
- Don't spend a flash cycle on CMSIS-NN before C1 confirms it's even compiling.
- Don't overwrite anything in `data/composed`, `results/`, or the deployed
  artifacts.
