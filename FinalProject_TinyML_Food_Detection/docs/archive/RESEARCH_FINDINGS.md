# Research findings — TinyML food classifier (answers to RESEARCH_BRIEF.md)

Compiled 2026-08-12. Every slug, URL, and index below was pulled from a primary
source (dataset card, paper, library source, or GitHub issue) and is cited.
Where a fact could not be confirmed to the required precision, it is marked
**UNVERIFIED** rather than guessed.

---

## TL;DR — ranked recommendation (read this first)

Given the goal (real overhead plate images, masks or masses, covering
chicken/broccoli/rice/beef/potato, adaptable to the existing synthetic-composition
pipeline), integrate in this order:

1. **FoodSeg103** (`EduardoPacheco/FoodSeg103` on HF) — **integrate first.**
   Ungated, Apache-2.0, one-line `load_dataset`, real plated Western food,
   per-ingredient pixel masks. Masks give exact food-area fractions → drop-in
   replacement for synthetic portion labels. Covers 4 of 5 classes directly and
   the 5th (beef) via `steak`. This is the best fit by a wide margin.
2. **Nutrition5k** — best for *mass* labels + true overhead RGB-D, but 181 GB and
   overhead set is only 3.5k dishes; subset-download only. Integrate second, only
   if you want real gram masses rather than mask-area proxy.
3. **Roboflow `hust-ajpvu/food-srnub`** — instance-seg polygons, 4/5 classes
   (no beef), free with login. Useful as domain-shift augmentation, not primary.
4. UNIMIB2016 / UECFoodPixComplete — situational; see Q4.

The single highest-value action: **FoodSeg103 masks → per-image food-area
fraction → portion tier**, keeping your composer for augmentation only.

---

## Q1 — FoodSeg103 (HIGHEST PRIORITY)

**Dataset:** `EduardoPacheco/FoodSeg103` · https://huggingface.co/datasets/EduardoPacheco/FoodSeg103
**Free:** yes · **Login:** no (ungated) · **License:** Apache-2.0
**Download:** `datasets.load_dataset("EduardoPacheco/FoodSeg103")` — train 4983,
validation 2135. Columns: `image`, `label` (PNG mask, class encoded as pixel
value), `classes_on_image` (int64 list), `id`.
**Source paper:** Wu, Xiongwei et al., "A Large-Scale Benchmark for Food Image
Segmentation", ACM MM 2021. Project: https://xiongweiwu.github.io/foodseg103.html
Original benchmark repo: https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1
**Masks:** per-ingredient / multi-region. Each plate has several labelled regions
(avg ~6 ingredient labels per image). Pixel value == class index below.

### 1. Full index → name map (mask pixel value = index)
Verified from the dataset's own `id2label.json` structure + the official category
list. Range is **0–103** (104 values incl. background). `label` PNG stores the
index directly as the pixel value.

```
0  background         26 date            52 sauce             78 ginger
1  candy              27 apricot         53 crab              79 okra
2  egg tart           28 avocado         54 fish              80 lettuce
3  french fries       29 banana          55 shellfish         81 pumpkin
4  chocolate          30 strawberry      56 shrimp            82 cucumber
5  biscuit            31 cherry          57 soup              83 white radish
6  popcorn            32 blueberry       58 bread             84 carrot
7  pudding            33 raspberry       59 corn              85 asparagus
8  ice cream          34 mango           60 hamburg           86 bamboo shoots
9  cheese butter      35 olives          61 pizza             87 broccoli
10 cake               36 peach           62 hanamaki baozi    88 celery stick
11 wine               37 lemon           63 wonton dumplings  89 cilantro mint
12 milkshake          38 pear            64 pasta             90 snow peas
13 coffee             39 fig             65 noodles           91 cabbage
14 juice              40 pineapple       66 rice              92 bean sprouts
15 milk               41 grape           67 pie               93 onion
16 tea                42 kiwi            68 tofu              94 pepper
17 almond             43 melon           69 eggplant          95 green beans
18 red beans          44 orange          70 potato            96 French beans
19 cashew             45 watermelon      71 garlic            97 king oyster mushroom
20 dried cranberries  46 steak           72 cauliflower       98 shiitake
21 soy                47 pork            73 tomato            99 enoki mushroom
22 walnut             48 chicken duck    74 kelp              100 oyster mushroom
23 peanut             49 sausage         75 seaweed           101 white button mushroom
24 egg                50 fried meat      76 spring onion      102 salad
25 apple              51 lamb            77 rape              103 other ingredients
```
(Source: FoodSeg103 category list, cross-checked against Dataset Ninja + HF
`id2label.json`. Note the raw HackMD list has minor typos in delimiters, e.g.
`37 lemon`; names above are normalized.)

### 2. ⚠️ Which classes map to your five — READ THIS, it changes the project
This is the single most important finding in Q1. **Your class list does not map
cleanly onto FoodSeg103.** Exact strings:

| Your class | FoodSeg103 class (exact) | Index | Caveat |
|---|---|---|---|
| broccoli | `broccoli` | 87 | clean 1:1 ✅ |
| rice     | `rice` | 66 | clean 1:1 ✅ |
| potato   | `potato` | 70 | clean, but `french fries` (3) is separate — decide if fries count |
| chicken  | `chicken duck` | 48 | **merged class** — chicken and duck share one label. No pure-chicken. |
| beef     | *(none)* | — | **No `beef` class.** Closest is `steak` (46). Also `pork` 47, `fried meat` 50, `lamb` 51. |

Implication: if you keep beef, you must decide beef ≡ `steak` (46), and accept
`chicken` is really `chicken+duck` (48). This is arguably *cleaner* than your
current Food-101 mapping (which merged `steak`+`filet_mignon`+`prime_rib` etc.,
per CLAUDE.md) — but it must be an explicit, logged decision in `build_dataset.py`.

### 3. Approximate per-class image counts
**UNVERIFIED at exact counts** — the HF viewer times out and per-class tallies
aren't published in the card. What IS known from the paper: 7,118 images, ~40k
masks, avg ~6 ingredient labels/image, ≥5 images per retained category (rare
classes were dropped). rice/potato/broccoli are common Western-plate ingredients
and will have materially more support than tail classes. **Action for Claude
Code:** compute exact counts locally in one pass over `classes_on_image` — do NOT
guess. Cheap:
```python
from collections import Counter
from datasets import load_dataset
ds = load_dataset("EduardoPacheco/FoodSeg103", split="train")
c = Counter()
for row in ds:
    c.update(set(row["classes_on_image"]))
# c[48], c[66], c[70], c[87], and steak c[46]
```
Log the result to `results/` and reuse it (per token-discipline rules).

### 4. License / paper — see header above (Apache-2.0; Wu et al. ACM MM 2021).

### 5. Masks per-ingredient? **Yes.** One plate → several labelled regions, one
class index per region. This is exactly what you need: food-area fraction per
class = (pixels of that class) / (non-background pixels), giving a real,
non-synthetic portion signal.

---

## Q2 — Nutrition5k

**Repo:** https://github.com/google-research-datasets/Nutrition5k
**Free:** yes · **Login:** no · **License:** CC-BY 4.0 (share/adapt, incl. commercial)
**Paper:** Thames et al., "Nutrition5k", CVPR 2021 (arXiv:2103.03375)

1. **Download (two options):**
   - Whole tarball: `nutrition5k_dataset.tar.gz` (**181.4 GB**) via
     `https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset.tar.gz`
   - **Subset via gsutil (do this):**
     `gsutil -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/{FILE_OR_DIR_PATH}" .`
     e.g. pull only overhead images + metadata, skip the 20k side-angle videos.
2. **A sample contains:** 5,006 dishes. Per dish: 4 rotating side-angle videos;
   **overhead RGB-D from Intel RealSense for ~3.5k of the 5k dishes** (raw 16-bit
   depth, 10,000 units = 1 m); fine-grained ingredient list; **per-ingredient
   mass (g)**; total dish mass + calories; fat/protein/carb masses. Metadata in
   `metadata/dish_metadata_cafe1.csv` + `cafe2.csv` (per-ingredient mass +
   macros); `metadata/ingredient_metadata.csv` (ingredient IDs + per-gram USDA
   nutrition). Overhead imagery under `imagery/realsense_overhead/<dish_id>/`.
3. **Size:** 181.4 GB total. **Subset:** yes — gsutil path selectors let you grab
   just `imagery/realsense_overhead/` + `metadata/` (a few GB) without videos.
4. **License:** CC-BY 4.0.
5. **Your five foods:** collected in California cafeterias → chicken, rice,
   broccoli, beef, potato are all plausible and ingredients ARE labelled by name
   with masses. **UNVERIFIED** that all five appear by those exact strings —
   confirm by grepping `ingredient_metadata.csv` after a metadata-only pull (tiny
   download). Do not assume string forms (e.g. may be "chicken breast",
   "white rice").

**Caveat:** Nutrition5k has **no segmentation masks** — it gives mass, not
pixel masks. Overhead RGB-D + mass is its value. Portion tiers would come from
real gram mass (better than FoodSeg103's area proxy) but you lose per-pixel spatial
labels. Mass thresholds map directly onto your small/medium/large (<80/80–180/>180 g).

---

## Q3 — Roboflow Universe

**Free tier (Q3.6):** Yes, download is free with a personal account. Free tier has
generous limits for dataset *export/download*; a few thousand images is fine.
(Rate limits mainly bite on hosted-inference API calls, not dataset downloads.)
The SDK prompts for your API key on first `download()`.

**Overhead plate shots (Q3.7):** The strong candidate below (`food-srnub`) is the
MyFoodRepo/Food-Recognition-2022 canteen imagery — mostly **near-overhead tray/plate
shots**, not angled restaurant photos. Good domain match. Roboflow food sets that
are top-down are rarer than angled ones; `food-srnub` is the best top-down-ish one
with masks covering your classes.

### Recommended dataset #1 (VERIFIED)
```
workspace slug : hust-ajpvu
project slug   : food-srnub
version number : 1
classes        : broccoli, chicken, rice, potatoes-steamed   (NO beef)
image count    : 9,984
annotation     : polygon (instance segmentation / masks)
license        : CC BY 4.0
URL            : https://universe.roboflow.com/hust-ajpvu/food-srnub
```
Full 30-class list includes: apple, banana, tomato, carrot, **broccoli**,
**chicken**, water, egg, **rice**, cucumber, cheese, tea, avocado, butter,
bread-white, bread-wholemeal, coffee, espresso, hard-cheese, jam,
mixed-salad, mixed-vegetables, pasta-spaghetti, **potatoes-steamed**,
salad-leaf, sweet-pepper, tomato-sauce, wine-red, wine-white. (This is the
Food-Recognition-2022 / MyFoodRepo taxonomy.)

SDK call (note: for masks use a segmentation export format, not plain yolov8 bbox):
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
rf.workspace("hust-ajpvu").project("food-srnub").version(1).download("yolov8")
# for polygon masks use .download("coco-segmentation") or ("yolov8") w/ seg — verify
# export options in the UI; "yolov8" here exports seg polygons for this project.
```
**Beef gap:** none of the clean Roboflow food-seg sets found cover beef as a
labelled mask class alongside the other four. If beef must come from Roboflow,
it'll be a second dataset merged in — or drop beef / fold into `steak` per Q1.

### Secondary candidate (bbox only, VERIFIED classes)
```
workspace slug : food-hofna
project slug   : food-detection-fme3o
version number : UNVERIFIED (check UI — likely 1)
classes        : grilled chicken breast, potato salad, white rice (+banana, black beans, pizza, spaghetti, milk, orange juice)
image count    : 294
annotation     : bbox
license        : CC BY 4.0
URL            : https://universe.roboflow.com/food-hofna/food-detection-fme3o
```
Small and bbox-only → lower value than `food-srnub`. Listed for completeness.

---

## Q4 — Other candidates

| Dataset | URL | Free | Login | License | Annotation | Size | Overhead? |
|---|---|---|---|---|---|---|---|
| **FoodSeg103** | (see Q1) | ✅ | no | Apache-2.0 | per-ingredient masks | 7,118 img | plated, varied angle |
| **Nutrition5k** | (see Q2) | ✅ | no | CC-BY 4.0 | mass + RGB-D (no masks) | 5k dishes / 181GB | **yes, true top-down** |
| **UNIMIB2016** | https://mldta.com/dataset/unimib2016-food-database/ + Kaggle mirror `dangvanthuc0209/unimib2016` | ✅ | Kaggle login for CLI | research use (verify) | polygon masks | 1,027 tray img, 73 cat | **yes, top-view canteen trays** |
| **UECFoodPixComplete** | https://mm.cs.uec.ac.jp/uecfoodpix/ | ✅ | no | research use | full-dish masks (not per-ingredient) | ~10k img, 102 cat | mixed |
| **UEC Food-100 / 256** | http://foodcam.mobi/dataset.html | ✅ | no | research use | **bbox only** | 100/256 cat | angled — **skip**, bbox-only per your brief |
| **MetaFood3D** | https://lorenz.ecn.purdue.edu/~food3d/ | ⚠️ | **request form + password** | CC-BY-NC 4.0 | 3D models + nutrition | 637 obj/108 cat | N/A (3D) — **fails "no application" constraint** |
| **NutritionVerse (3D2D)** | search "NutritionVerse" HF/GitHub | ✅ (mostly) | varies | CC (verify per subset) | synthetic 2D renders + bbox + 3D | large | synthetic multi-view |

Notes / flags:
- **UNIMIB2016** is genuinely top-view and mask-annotated — the closest analogue
  to your deployment geometry after Nutrition5k. But it's *Italian canteen* food;
  class overlap with your five is weak (no clean chicken/beef/broccoli/rice/potato
  guarantee). Value = geometry/domain, not class coverage. Annotations are `.mat`
  (TrainingSet.mat/TestSet.mat) — needs conversion.
- **UECFoodPix Complete** masks cover the *entire dish*, not ingredients — less
  useful than FoodSeg103 for area-fraction portion labels.
- **MetaFood3D** is gated (password on request) → **excluded** by your "not email
  the authors" rule. Mention only for completeness.
- **UEC Food-100/256** are bbox-only → excluded by your "ignore classification/
  no-spatial" rule (bbox is weak spatial, and no masks/mass).

---

## Q5 — OV7675 RGB565 byte order (hardware test unblock)

**Primary sources:**
- Driver source: `Arduino_OV767X/src/OV767X.cpp` `readFrame()` —
  https://github.com/arduino-libraries/Arduino_OV767X/blob/master/src/OV767X.cpp
- Adafruit camera docs (RGB565 byte-order convention):
  https://learn.adafruit.com/capturing-camera-images-with-circuitpython/working-with-image-data

### 1. Byte order `readFrame()` delivers
**`readFrame()` does NO byte-swapping.** It bit-bangs GPIO and stores each byte
in the exact order the OV767X clocks it out (see the inner loop: `*b++ = in;`
per PCLK, no reordering). So the byte order you receive == the sensor's output
convention. For the OV7670/OV7675 in RGB565 mode, the sensor clocks the pixel as
**two bytes that are byte-swapped relative to the naive `(hi<<8)|lo` big-endian
assumption** — Adafruit explicitly documents this mode as **"RGB565-swapped"**
(the left/right 8 bits of the 16-bit pixel are swapped).

**Consequence for your sketch:** your current code assumes the first byte is the
high byte:
```cpp
const uint16_t px = (uint16_t(hi) << 8) | lo;   // <-- assumes big-endian
```
This is very likely **wrong** for this sensor path. Test the swapped combine:
```cpp
const uint16_t px = (uint16_t(lo) << 8) | hi;   // swapped — try this
```
i.e. swap which incoming byte you treat as high. This is a **1-flash-cycle test**:
capture the known test pattern (`Camera.testPattern()`), decode both ways on host,
compare to the expected zig-zag/color-bar reference. Whichever yields correct bar
colors is your byte order. (Fits your 3-flash-cycle debug cap.)

### 2. Red/blue channel swap
**Yes, this is a known, widely-reported issue** with the OV767X + RGB565 path
(Adafruit "RGB565-swapped"; multiple Arduino-forum + OpenMV reports of R/B
appearing swapped or washed out; HarvardX TinyML kit forum thread reports
saturated/incorrect color: forum.arduino.cc/t/ov7675-camera-from-tinyml-kit-not-capturing-image-properly/1304179).

**Accepted fix:** it's fundamentally the same byte-order root cause. After you fix
the byte order (item 1), verify channel assignment. If R and B are still
transposed, swap the extraction masks (treat the low 5 bits as R and the high 5
as B, or equivalently swap r/b after unpacking):
```cpp
// if R/B still swapped after fixing endianness:
b = ((px >> 11) & 0x1F) * 255 / 31;   // was r
g = ((px >> 5)  & 0x3F) * 255 / 63;
r = ( px        & 0x1F) * 255 / 31;   // was b
```
Do byte-order first, then R/B — don't change both blindly in one cycle or you
can't tell which fix worked. **Ruled-out-if-confirmed:** log whichever combo
gives correct test-pattern colors and don't re-test (per CLAUDE.md debug rules).

### 3. Citation
Driver behavior: `OV767X.cpp` `readFrame()` (raw, no swap) — link above.
Swap convention: Adafruit "Working with Image Data" (RGB565-swapped) — link above.
Symptom corroboration: Arduino forum HarvardX-kit thread — link above.

---

## Q6 — CMSIS-NN in TFLite Micro on Arduino (1,076 ms → ?)

**Primary sources:**
- TFLM CMSIS-NN kernels README (OPTIMIZED_KERNEL_DIR mechanism):
  https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels/cmsis_nn/README.md
- Arduino examples `library.properties` (`precompiled=full`):
  https://github.com/tensorflow/tflite-micro-arduino-examples/blob/main/library.properties
- Link-failure evidence on Nano 33 BLE: TFLM issue #2331
  https://github.com/tensorflow/tflite-micro/issues/2331
- Required DSP macro: CMSIS_5 issue #1429
  https://github.com/ARM-software/CMSIS_5/issues/1429
- Speedup figures: TF Blog (Feb 2021) + TFLM survey (Heim et al. 2021)

### 1. What selects `kernels/cmsis_nn/*` over `kernels/*`
In the **upstream Make build**, it's the flag `OPTIMIZED_KERNEL_DIR=cmsis_nn`:
this physically substitutes which per-operator `.cc` gets compiled. Each operator
has exactly ONE implementation file in the build (reference `kernels/conv.cc` OR
`kernels/cmsis_nn/conv.cc`) — never both (duplicate symbols otherwise). The
cmsis_nn `.cc` internally calls `arm_convolve_*` / `arm_depthwise_*` **only when
the CMSIS-NN intrinsics are actually available**, i.e. when the compiler defines
**`ARM_MATH_DSP`** (and `ARM_MATH_MVEI` on M55). Without those macros the same
file can fall back to a generic path.

**On the Arduino build there is no Makefile.** The Arduino IDE/CLI compiles
whatever `.cpp/.cc` sources are physically present in the library `src/` tree
(subject to `library.properties`). So kernel selection is decided by **which
kernel sources the library ships** + **whether `ARM_MATH_DSP` is defined at
compile time**. This is the crux of your "compiled but no speedup" result:

- If the reference `kernels/*.cc` are present (or the cmsis_nn ones compile
  without `ARM_MATH_DSP`), you get reference-speed math regardless of the
  cmsis_nn directory existing on disk.
- The optimized path only engages when `ARM_MATH_DSP` is defined AND the CMSIS-NN
  Source/Include tree is compiled AND the reference duplicates are excluded.

### 2. `precompiled=full` — YES this is very likely your root cause (Q6.3)
`tflite-micro-arduino-examples/library.properties` contains **`precompiled=full`**.
In Arduino's library spec, `precompiled=full` tells the toolchain to **link a
prebuilt `.a` archive and NOT recompile the library sources**. If a prebuilt
archive is being linked (or the flag confuses the source-vs-archive resolution),
your freshly-added cmsis_nn sources may **never be compiled at all** — exactly
consistent with "1,092,437 µs, no speedup." **Also note: that repo was archived
read-only on 2025-02-24**, so it's frozen; don't expect upstream fixes.

**Concrete checks (cheap, no board needed — use `harness/compile_sweep.py`):**
- Grep the build output / `.a` for `arm_convolve_s8` symbols. If absent → CMSIS-NN
  never compiled.
- Confirm whether a `libtensorflowlite.a` (or similar) is being linked due to
  `precompiled=full`. If so, the archive predates your source edits.

### 3. Known issue where sources ship but never compile
Yes — two documented failure modes:
  (a) **`precompiled=full`** causing sources to be ignored in favor of a prebuilt
      archive (this repo).
  (b) When they DO compile without the Mbed CMSIS providing DSP intrinsics, you
      get **link errors** `undefined reference to __sxtb16 / __smlabb / __smlatt`
      (TFLM issue #2331, on `arduino:mbed_nano` core, `-mcpu=cortex-m4
      -mfloat-abi=softfp -mfpu=fpv4-sp-d16 -mthumb`). Getting THOSE errors is
      actually a sign the optimized path is being pulled in; the fix is ensuring
      `ARM_MATH_DSP` + correct DSP intrinsics from the mbed CMSIS core.

### 4. Concrete enable steps (arduino-cli)
There is no clean `--build-property` that flips `OPTIMIZED_KERNEL_DIR` for a
precompiled Arduino lib, so the reliable route is to **stop using `precompiled`
and force the DSP macro**:

1. In the library's `library.properties`, set **`precompiled=false`** (and remove
   `dot_a_linkage`) so sources actually compile.
2. Ensure the cmsis_nn kernel `.cc` are present and the **reference duplicates for
   the same ops are removed** from `src/` (no duplicate symbols).
3. Force the DSP macro via build property:
   ```
   arduino-cli compile -b arduino:mbed_nano:nano33ble \
     --build-property "compiler.cpp.extra_flags=-DARM_MATH_DSP -DCMSIS_NN" \
     --build-property "compiler.c.extra_flags=-DARM_MATH_DSP" \
     <sketch>
   ```
   (Some setups also need `-D__ARM_FEATURE_DSP=1` if the core doesn't set it.)
4. Confirm the CMSIS-NN `Source/` + `Include/` tree compiles and links
   (`arm_nnfunctions.h` resolvable). If you hit `__sxtb16`/`__smlabb` undefined
   refs, that's issue #2331 — the mbed core's CMSIS must supply those DSP
   intrinsics; align CMSIS include paths with the mbed_nano core version.

Realistically, given the archived/precompiled upstream, the **cleaner path may be
to switch runtimes**: use a from-source TFLM build (or the
`tflite-micro-arduino-examples` fork with updated scripts referenced in issue
#2331, https://github.com/Gostas/tflite-micro-arduino-examples), or an Edge-Impulse-
exported library which ships CMSIS-NN correctly wired. Weigh this against your
"adapt not replace" preference — but the precompiled archive is a real wall.

### 5. Realistic expected speedup (Q6.4)
For a Cortex-M4F @ 64 MHz, int8, published figures:
- **Conv / fully-connected: ~6–7×** faster with CMSIS-NN vs reference (TF Blog;
  Heim et al. 2021). Latency-per-MAC roughly halves when major dims are multiples
  of 4 (data-pack alignment).
- **Depthwise conv: only ~2.2×** (limited filter reuse).

Your model is **depthwise-separable**, so the aggregate is a blend: the pointwise
(1×1) convs behave like regular conv (~6–7×) while the depthwise stages get ~2.2×.
**Expected end-to-end: roughly 3–5× overall**, i.e. **~1,076 ms → ~250–350 ms**,
plausibly better if pointwise convs dominate MACs. Do not expect a flat 7×.
This alone likely won't hit real-time, but combined with your existing arena it's
a large, free win once the kernels actually compile.

---

## Verification status summary
- **Verified from primary source:** FoodSeg103 index map, license, paper, mask
  type; Nutrition5k download/contents/license; `food-srnub` slug/version/classes/
  license/count; `library.properties precompiled=full`; OV767X `readFrame` no-swap;
  CMSIS-NN OPTIMIZED_KERNEL_DIR mechanism; #2331 link errors; speedup ranges.
- **UNVERIFIED (flagged, compute/confirm locally):** exact FoodSeg103 per-class
  image counts; Nutrition5k exact string forms of your 5 foods;
  `food-detection-fme3o` version number; UNIMIB2016 exact license string;
  NutritionVerse per-subset license.
- **Excluded by your constraints:** MetaFood3D (gated/password); UEC Food-100/256
  (bbox-only).
