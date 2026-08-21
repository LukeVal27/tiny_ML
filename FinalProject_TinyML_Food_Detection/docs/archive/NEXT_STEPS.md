# How to proceed

Written 2026-08-12. Every number quoted here is verified from `results/`.
Tier 1 is deployed and device-verified; what remains is validation on real
plates, then the graded write-up items.

Steps are ordered by dependency. Steps 1–3 are the critical path to a demo.

---

## Step 1 — Live-camera validation (BLOCKING, needs a human)

**Why it matters:** this is the only major claim not yet verified. `BENCH_MODE=1`
proves inference is correct on a baked-in image. `BENCH_MODE=0` — capture →
downsample → classify — has **never been run end-to-end**. Until it is, the demo
workflow in handoff §7 is unproven.

The RAM arithmetic already works out, from measured numbers:

```
overhead 51,792 + RGB565 frame 50,688 + arena 120,000 = 222,480 of 262,144 SRAM
```

**Run it:**

```bash
python3 harness/flash_and_measure.py --sketch deploy/nano/classifier_tier1 \
    --define ARENA_SIZE=120000 --define BENCH_MODE=0 --timeout 40
```

Expect `BOOT,status=OK,...` then `# camera ready`. Send `c` over serial (or press
the shield button) to capture. Each capture prints `INFER,...` and `TIMING,...`.

**What to check, in order:**

1. `BOOT,status=OK` — if `reason=camera_init`, the camera failed to start;
   re-seat the shield. The smoke test already proved `cam=OK`, so this should pass.
2. **Point at a plate and confirm the prediction is not constant.** A fixed
   output across very different plates means the preprocessing path is wrong, not
   the model.
3. `TIMING,total_us` — this is the capture-to-result latency handoff §7 asks for.
   Inference alone is 1.076 s; capture and preprocessing add to it.

**Most likely failure, and how to tell:** RGB565 byte order. `unpack565()` in
`classifier_tier1.ino` assumes big-endian (high byte first). If the driver
delivers little-endian, red and blue swap — and since this model leans heavily on
colour, broccoli would misclassify while overall behaviour still looks plausible.

To test cheaply, point the camera at something strongly red and something
strongly green. If predictions look colour-inverted, swap the two bytes in
`unpack565()`:

```cpp
const uint16_t px = (uint16_t(lo) << 8) | hi;   // little-endian variant
```

Do not guess — verify with a known-colour target first.

---

## Step 2 — Self-collected deployment-domain set (needs a human)

Handoff §3 and §7 both require this, and it is where the sim-to-real number comes
from. The model trained entirely on synthetic composites; nobody knows yet how it
behaves on real OV7675 frames.

**Capture protocol** (hold these constant or the portion head is meaningless):

- fixed camera height, recorded in the notes
- a plate with a visible rim in every frame — this is the scale fiducial the
  portion head was explicitly trained to use
- ≥3 lighting conditions (daylight, warm indoor, dim)
- all 5 classes × 3 portion tiers, ideally ≥10 frames per cell (~150 frames)
- **weigh each serving** and record grams — this is what makes Step 4 possible

Save images plus a CSV of `filename, class, grams` into `data/self_collected/`
(the directory already exists and is empty).

**Then report it separately** from the public-data test split. Reporting them
merged would hide exactly the gap the professor asked to see.

---

## Step 3 — Latency (optional but high-value)

**1.076 s/inference is the weakest measured number.** The bundled
`Harvard_TinyMLx` TFLM uses reference kernels only — verified: it contains no
`cmsis_nn` directory.

A current TFLM is already cloned at `~/Documents/Arduino/libraries/TFLM_New`
(commit `7948b85`, 2023-12-21) and **does** ship `third_party/cmsis_nn` with
optimized `conv`, `depthwise_conv`, `pooling`, `fully_connected`, `softmax` and
`add` — exactly this model's op set.

Important caveat, measured: simply compiling against that library gave
**1,092,437 µs — no faster**, so the CMSIS-NN kernels were *not* actually
selected by the build. Making them take effect requires the right build flags
(e.g. `-DCMSIS_NN`/`ARM_MATH_DSP`-style defines), not merely the library's
presence. Treat this as a build-configuration task, not a drop-in swap.

That work is done in a scratch sketch, not the deliverable, and the API differs:
`MicroInterpreter` takes no `ErrorReporter`, and `MicroMutableOpResolver<8>` with
the 8 ops registered is smaller than `AllOpsResolver`.

The other lever is the model itself: the stem dominates the arena
(96×96×3 input → 48×48×16). Dropping `--alpha` to 0.75 or the input to 64×64
would cut both latency and the 113,516 B arena, at some accuracy cost.

---

## Step 4 — Recalibrate portion thresholds from real masses

Currently the tier boundaries come from a *stated physical model*, not
measurement:

```
mass ≈ f × plate_area × depth × density  →  ≈513 g per unit area-fraction
80 g → f = 0.156     180 g → f = 0.351
```

All constants sit at the top of `data/compose_portions.py`
(`PLATE_DIAM_CM`, `MEAN_FOOD_DEPTH_CM`, `MEAN_DENSITY_G_CM3`,
`MASS_SMALL_MAX_G`, `MASS_MEDIUM_MAX_G`). With Step 2's weighed captures, fit the
real grams-per-area-fraction, replace those constants, regenerate, and retrain.

Being able to say "our thresholds are measured, not assumed" directly addresses
the vague-portion-tier deduction.

---

## Step 5 — Improve class accuracy (the real ceiling)

macro-F1 0.5493 is data-limited, and this is established rather than assumed: a
colour-histogram linear probe scores **0.51** on the raw full-resolution source
photos and **0.46** on the composed plates. Composition costs only ~0.05; the
coarse classes are simply hard to separate.

Weakest classes: **rice 0.4130** and **chicken 0.4802** (broccoli is 0.7398).

In expected value order:

1. **Real deployment-domain data** (Step 2, scaled up). Highest value by far.
2. **Roboflow with real boxes** — `ROBOFLOW_API_KEY` was never set this session.
   Real boxes also make the portion head supervised from real geometry instead of
   synthetic composition, and re-open Tier 2.
3. **More source diversity per class** — chicken and potato have only 749 and 750
   train source photos, the two smallest, and chicken is among the weakest.
4. Architecture changes last. A bigger network cannot beat a 0.51 data ceiling.

Do **not** re-run the `data/composed_v4` foodiness-filter experiment. It was
measured as neutral (0.453 vs 0.460) and is already built if anyone wants it.

---

## Step 6 — Write-up

`results/REPORT.txt` already contains every §7 metric except capture-to-result
latency (Step 1) and the deployment-domain split (Step 2). Regenerate with:

```bash
python3 -m eval.make_report          # defaults to the deployed a10_pool tag
```

Points to make explicitly, each backed by a measured number in `results/`:

- **Hardware choice was validated, not assumed** — the RAM gate ran as an 11-point
  static sweep, then confirmed on hardware.
- **Portion tiers are concrete** — explicit gram thresholds from a stated physical
  model, and an ordinal loss whose off-by-two rate is **0.000333**.
- **The demo workflow is defined** — capture → downsample → classify → serial
  output of class, tier and mapped gram range.
- **Sim-to-real is quantified separately** (once Step 2 lands).
- **The accuracy limit is diagnosed, not hand-waved** — linear-probe evidence that
  the ceiling is in the data.

---

## Working rules for whoever continues

Read `CLAUDE.md` first. In short: use `Edit` not heredocs; filter all
`arduino-cli` output; attach the serial reader *before* the board boots; reuse
the numbers in `results/` instead of re-measuring; cap hardware debugging at ~3
flash cycles then ask.

Two traps already paid for, both now encoded in `.claude/skills/`:

- **Any sketch invoking more than once must refill the input tensor each time.**
  `Invoke()` overwrites its own input, because the input buffer is aliased with
  arena scratch.
- **Never poll for background jobs with `pgrep -f`.** The pattern matches the
  polling shell's own command line, so the loop never exits. Five such shells ran
  for over 90 minutes.
