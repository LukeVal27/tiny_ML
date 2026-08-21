# Research brief — TinyML food classifier, data + toolchain questions

Copy everything below the line into a Claude chat with web search enabled.

---

I need you to research and answer the questions below. This is for a university
TinyML project that is already built and deployed — I am not looking for general
advice, project ideas, or tutorials. I need specific, verifiable facts with
working links.

## Project context

A 5-class food classifier plus a 3-tier portion-size head, quantized to int8 and
running on an **Arduino Nano 33 BLE Sense (nRF52840, 256 KB SRAM, 1 MB flash)**
with an **OV7675** camera on the Arduino Tiny Machine Learning shield.

- Classes: **chicken, broccoli, rice, beef, potato**
- Portion tiers: small (<80 g), medium (80–180 g), large (>180 g)
- Input: 96×96 RGB. Model is 221,888 bytes int8, uses 113,516 bytes of tensor arena.
- Current accuracy: **0.5593 class accuracy / 0.5493 macro-F1** (5 classes, chance = 0.20).
- Deployment view: a plate shot from directly above, plate rim visible in frame,
  one food type per plate.

**The problem I am solving.** Training data came from Food-101 (restaurant dish
photos) and the ImageNet broccoli synset. Neither has bounding boxes or masks, so
portion labels had to be *synthesised*: food texture patches pasted inside drawn
elliptical blobs on drawn plates. Two consequences:

1. Class accuracy is low. A logistic regression on nothing but colour histograms
   scores 0.510 on the same source photos, i.e. a colour-only model gets within
   ~5 points of the CNN — so the CNN appears to extract little beyond colour.
2. The model has never seen a real photograph of a plate, only synthetic composites.

I want **real overhead plate images**, ideally with **segmentation masks** (which
give exact food-area fractions, replacing my synthetic portion labels) or at
minimum **bounding boxes**, and ideally with **real gram/mass labels**.

## Constraints on what is useful

- Must be **free**, including for academic use. A free account/login is fine.
  Anything paid, quota-limited to uselessness, or requiring institutional
  application is not useful.
- Must be **programmatically downloadable** (Python SDK, HTTP, git, or `kaggle`
  CLI). Not "email the authors for access."
- License must permit academic use; state the license explicitly.
- Prefer datasets that are food-on-a-plate, not raw ingredients on a white
  background and not food growing in fields.

---

# Questions

## Q1 — FoodSeg103 class coverage (highest priority)

The HuggingFace dataset `EduardoPacheco/FoodSeg103` exists and is ungated. Its
columns are `image`, `label` (a pixel segmentation mask), `classes_on_image`.

1. **List all 103 class names, with their integer indices.** I need the exact
   index→name mapping because the masks encode class as pixel value.
2. Which classes correspond to my five (**chicken, broccoli, rice, beef,
   potato**)? Quote the exact class strings.
3. Roughly how many images contain each of those classes?
4. What is the license, and what is the original source paper?
5. Are the masks per-ingredient (one plate → several labelled regions) or
   per-image single-label?

## Q2 — Nutrition5k

Google Research dataset of overhead plate images with per-ingredient mass.

1. Is it publicly downloadable **for free**, and from where exactly? Give the
   real URL (GitHub repo, GCS bucket, or `gsutil` command).
2. What does a sample actually contain — overhead RGB? depth? video frames?
   per-ingredient gram masses? total dish mass only?
3. Total download size, and can I fetch a **subset** without pulling everything?
4. License.
5. Does it contain dishes with my five foods, and are ingredients labelled by
   name?

## Q3 — Roboflow Universe (free with login is acceptable)

I need datasets covering chicken, broccoli, rice, beef and potato with
**bounding boxes or masks**.

For each dataset you recommend, give me **exactly these fields**, because the
Python SDK needs the precise slugs:

```
workspace slug : <string>
project slug   : <string>
version number : <int>
classes        : <which of my 5 it covers, exact label strings>
image count    : <n>
annotation     : bbox | polygon | mask
license        : <string>
URL            : https://universe.roboflow.com/<workspace>/<project>
```

so that this call works verbatim:

```python
rf.workspace("<workspace>").project("<project>").version(<n>).download("yolov8")
```

Also confirm:
6. Is Roboflow Universe download free with a personal account, and are there
   rate/quota limits on the free tier that would block downloading a few
   thousand images?
7. Do any of these datasets contain **overhead/top-down plate shots**
   specifically, as opposed to angled restaurant photos?

## Q4 — Any other candidates

Datasets with overhead food images and mask or mass annotations. Candidates I am
aware of but have not verified: **UEC Food-100 / UEC Food-256** (reported to have
bounding boxes), **UNIMIB2016** (canteen trays, segmentation), **FoodSeg103**,
**MetaFood3D**, **NutritionVerse**. For each that is real and free, give the
download URL, annotation type, size, and license. Ignore any that are
classification-only with no spatial annotation — I already have that.

## Q5 — OV7675 RGB565 byte order (blocking a hardware test)

My Arduino sketch converts RGB565 camera bytes to RGB. I currently assume
**big-endian** (high byte first):

```cpp
const uint16_t px = (uint16_t(hi) << 8) | lo;
r = ((px >> 11) & 0x1F) * 255 / 31;
g = ((px >> 5)  & 0x3F) * 255 / 63;
b = ( px        & 0x1F) * 255 / 31;
```

Using the **`Arduino_OV767X`** library (or the Harvard TinyMLx fork,
`Harvard_TinyMLx`, which provides `Camera.readFrame()` and
`Camera.begin(QCIF, RGB565, 5, OV7675)`):

1. What byte order does `readFrame()` actually deliver for RGB565 — high byte
   first or low byte first?
2. Is there a known red/blue channel swap issue with this driver, and what is the
   accepted fix?
3. Cite the library source file or a GitHub issue, not a blog guess.

## Q6 — Enabling CMSIS-NN in TFLite Micro on Arduino

My inference takes **1,076 ms**, which I believe is because reference kernels are
being used instead of CMSIS-NN optimised ones.

Facts I have already established myself — please do not re-derive these:
- The `Harvard_TinyMLx` library contains **no** `cmsis_nn` directory.
- `tensorflow/tflite-micro-arduino-examples` (commit `7948b85`, 2023-12-21) **does**
  ship `src/third_party/cmsis_nn` with optimised `conv`, `depthwise_conv`,
  `pooling`, `fully_connected`, `softmax` and `add`.
- Simply compiling against that library gave **1,092,437 µs — no speedup**, so the
  optimised kernels were evidently not selected by the build.

Questions:
1. What actually determines whether TFLM compiles the `kernels/cmsis_nn/*`
   implementations rather than `kernels/*` reference ones on an Arduino
   `arduino:mbed_nano:nano33ble` build? Preprocessor define, build flag,
   directory layout, or `library.properties` setting?
2. Give the concrete `arduino-cli` invocation or `--build-property` flags that
   enable them.
3. Are there known issues where the Arduino library ships CMSIS-NN sources but
   the Arduino build system never compiles them (e.g. because of the
   `precompiled=full` flag in `library.properties`)?
4. Realistically, what speedup should a ~157k-parameter depthwise-separable
   int8 CNN at 96×96 RGB expect on a Cortex-M4F @ 64 MHz with CMSIS-NN?
5. Cite TFLM docs, GitHub issues, or source — not blog posts.

---

# Output format

For every dataset: **name · direct URL · free? · login required? · license ·
annotation type · image count · exact download command**.

Rules:
- If you cannot verify something, say "unverified" — do **not** guess a URL, a
  class name, or a slug. A wrong Roboflow slug wastes a debugging cycle.
- Prefer primary sources: dataset cards, papers, GitHub repos, library source.
- Flag anything that looks abandoned or has broken download links.
- End with a short ranked recommendation: **which single dataset should I
  integrate first**, given that I need real overhead plate images with masks or
  masses covering chicken, broccoli, rice, beef and potato, and that I already
  have a working synthetic-composition pipeline I would rather adapt than replace.
