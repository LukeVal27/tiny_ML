#!/usr/bin/env python3
"""
Synthetic plate composition -> exact portion-tier ground truth.

Why this exists
---------------
The keyless public sources are classification-only, so there are no bounding
boxes and therefore no portion signal (handoff Section 4 derives the tier from
relative bbox area). Instead of inventing weak labels, we *construct* the
supervision: paste a food patch onto a synthetic plate at a known area fraction.
The label is then exact by construction rather than estimated.

This also matches the deployment setup described in Section 4 far better than
raw web photos do: fixed camera height, a plate rim in frame acting as the scale
fiducial, food occupying a measurable fraction of that rim.

The anti-shortcut detail that makes this work
---------------------------------------------
If the plate were always rendered at the same apparent size, "food pixel area"
alone would predict the tier and the network would never learn to use the rim.
So we jitter the plate's apparent diameter (PLATE_DIAM_JITTER, simulating real
camera-height variation) while defining the label from food area *relative to the
plate*. Absolute pixel area is then a genuinely ambiguous cue, and the only way
to score well is to measure the food against the rim -- which is exactly the
behaviour we need on-device.

Area fraction -> mass
---------------------
Section 4 wants explicit thresholds. We state the physical model rather than
hand-waving a number:

    mass_g ~= f * PLATE_AREA_CM2 * MEAN_FOOD_DEPTH_CM * MEAN_DENSITY_G_CM3

with f the fraction of the plate covered. With a 22 cm inner-diameter plate
(380 cm^2), ~1.5 cm mean serving depth and ~0.9 g/cm^3 mean density this gives
mass_g ~= f * 513, so the Section 4 cutoffs of 80 g / 180 g land at f = 0.156
and f = 0.351. Those two numbers are the tier boundaries below, and they are
constants at the top of this file precisely so Luke/Daniel can replace them with
measured values from the calibration captures without touching any other code.
"""

import argparse
import json
import math
import pathlib
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO = pathlib.Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
OUT = REPO / "data" / "composed"

CLASSES = ["chicken", "broccoli", "rice", "beef", "potato"]
PORTION_NAMES = ["small", "medium", "large"]

# ----------------------------------------------------------- physical model
PLATE_DIAM_CM = 22.0
PLATE_AREA_CM2 = math.pi * (PLATE_DIAM_CM / 2) ** 2      # ~380 cm^2
MEAN_FOOD_DEPTH_CM = 1.5
MEAN_DENSITY_G_CM3 = 0.9
GRAMS_PER_UNIT_FRACTION = PLATE_AREA_CM2 * MEAN_FOOD_DEPTH_CM * MEAN_DENSITY_G_CM3

MASS_SMALL_MAX_G = 80.0
MASS_MEDIUM_MAX_G = 180.0

F_SMALL_MAX = MASS_SMALL_MAX_G / GRAMS_PER_UNIT_FRACTION      # ~0.156
F_MEDIUM_MAX = MASS_MEDIUM_MAX_G / GRAMS_PER_UNIT_FRACTION    # ~0.351

# ------------------------------------------------------------ render params
IMG_PX = 96
PLATE_DIAM_JITTER = (0.62, 0.92)   # plate diameter as a fraction of frame width
F_RANGE = (0.04, 0.62)             # sampled food/plate area fractions


def fraction_to_tier(f):
    if f < F_SMALL_MAX:
        return 0
    if f < F_MEDIUM_MAX:
        return 1
    return 2


def fraction_to_mass(f):
    return f * GRAMS_PER_UNIT_FRACTION


# Plausible surfaces a plate actually sits on. Fully random RGB produced neon
# teal/magenta "tables" that no camera will ever see, which wastes model capacity
# on backgrounds instead of on food.
TABLE_TONES = [
    (150, 111, 74),   # light wood
    (110, 76, 48),    # medium wood
    (74, 50, 32),     # dark wood
    (196, 192, 185),  # light stone / laminate
    (140, 138, 134),  # grey counter
    (86, 88, 92),     # dark slate
    (222, 219, 212),  # white tablecloth
    (168, 160, 140),  # beige placemat
]


def make_table(px, rng):
    """A plausible surface under the plate: realistic base tone + smoothed noise."""
    base = np.array(TABLE_TONES[int(rng.integers(len(TABLE_TONES)))], dtype=np.float32)
    base = base * rng.uniform(0.82, 1.18)              # overall lightness
    tint = rng.uniform(-10, 10, size=3)                # mild colour cast only
    noise = rng.normal(0, 12, size=(px, px, 1))
    img = np.clip(base + tint + noise, 0, 255).astype(np.uint8)
    out = Image.fromarray(img)
    return out.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 2.2)))


def blob_mask(px, cx, cy, radius, rng, lobes=None):
    """
    Smooth irregular closed shape in polar coordinates.

    A plain circle would let the model key on shape rather than extent, so the
    radius is modulated by a few low-frequency sinusoids to look like a real
    serving spread on a plate.
    """
    lobes = lobes or rng.integers(3, 7)
    phases = rng.uniform(0, 2 * math.pi, size=int(lobes))
    amps = rng.uniform(0.06, 0.20, size=int(lobes))

    pts = []
    for i in range(180):
        th = 2 * math.pi * i / 180
        r = radius * (1.0 + sum(a * math.sin((k + 2) * th + p)
                                for k, (a, p) in enumerate(zip(amps, phases))))
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))

    m = Image.new("L", (px, px), 0)
    ImageDraw.Draw(m).polygon(pts, fill=255)
    return m.filter(ImageFilter.GaussianBlur(1.1))


def sample_fraction(rng):
    """
    Draw an area fraction with the three tiers equally likely.

    Sampling f uniformly over F_RANGE would give a 20/34/46 small/medium/large
    split simply because the "large" band is widest. That imbalance would show up
    as a lopsided confusion matrix that says more about the generator than about
    the model, so we pick the tier first and then the fraction within it.
    """
    tier = int(rng.integers(0, 3))
    lo, hi = [
        (F_RANGE[0], F_SMALL_MAX),
        (F_SMALL_MAX, F_MEDIUM_MAX),
        (F_MEDIUM_MAX, F_RANGE[1]),
    ][tier]
    # Nudge off the exact boundary so a float rounding tie cannot flip the label.
    return float(rng.uniform(lo + 1e-4, hi - 1e-4))


def crop_texture(src, patch_px, rng):
    """
    Take a food patch from the source photo at (roughly) 1:1 texture scale.

    The naive version resized a fixed ~0.5-of-image crop down to whatever the
    serving size demanded. That is physically wrong and it hurt small portions
    badly: shrinking a whole risotto photo to 30 px makes the individual grains
    sub-pixel, so a small serving of rice became an untextured pale smear that
    the classifier could not distinguish from potato.

    In reality a small portion shows FEWER grains, not SMALLER ones. So we size
    the source crop in proportion to the destination patch, holding the
    resampling factor near 1 and preserving real texture frequency at every
    portion size.

    Cropping stays centre-biased: Food-101 photos are centred on the dish, and
    edge crops pick up tablecloth, cutlery and hands, which would silently
    mislabel those pixels as food.
    """
    zoom = rng.uniform(0.90, 1.55)          # keeps resample factor in ~[0.65, 1.1]
    limit = min(src.width, src.height)
    s = int(np.clip(patch_px * zoom, 24, limit))
    span = limit - s

    # Try a few placements and keep the most food-like one. Even centre-biased
    # crops of Food-101 photos regularly land on bare plate, tablecloth or dark
    # background, and every one of those is a mislabelled training example that
    # caps achievable accuracy no matter how good the model is.
    best, best_score = None, -1.0
    for _ in range(6):
        if span > 0:
            ctr = span / 2.0
            x = int(np.clip(rng.normal(ctr, span * 0.18), 0, span))
            y = int(np.clip(rng.normal(ctr, span * 0.18), 0, span))
        else:
            x = y = 0
        cand = src.crop((x, y, x + s, y + s))
        sc = foodiness(cand)
        if sc > best_score:
            best, best_score = cand, sc
        if sc > 0.55:                       # good enough, stop early
            break

    return best.resize((patch_px, patch_px), Image.LANCZOS)


def foodiness(patch):
    """
    Cheap heuristic score for "this crop actually contains food".

    Food is colourful and textured; the things we want to reject are flat and
    either near-white (plate), near-grey (tablecloth) or near-black (shadow and
    background). We score on saturation and local variation and penalise the
    plate-white and very-dark corners of colour space.

    This is a heuristic, not a segmenter -- it does not need to be right every
    time, it just needs to shift the distribution away from obvious non-food.
    """
    a = np.asarray(patch.convert("RGB"), dtype=np.float32) / 255.0
    if a.size == 0:
        return 0.0

    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    val = mx

    mean_sat = float(sat.mean())
    mean_val = float(val.mean())
    texture = float(a.mean(axis=2).std())

    score = 0.6 * mean_sat + 0.4 * min(texture / 0.18, 1.0)
    if mean_val > 0.80 and mean_sat < 0.12:   # bare white plate
        score *= 0.25
    if mean_val < 0.14:                        # near-black background
        score *= 0.3
    return score


def draw_plate(px, rng):
    """
    Render table + plate. Returns (canvas, plate_cx, plate_cy, plate_radius).

    Shared by the synthetic-texture path (compose_one, below) and the real-cutout
    path (data/compose_real.py) so both produce an identical plate geometry --
    including PLATE_DIAM_JITTER, which is the anti-shortcut that forces the
    portion head to measure food against the rim rather than in raw pixels.
    """
    canvas = make_table(px, rng).convert("RGB")

    # --- plate: apparent diameter jitters, so absolute area is not the label
    plate_d = rng.uniform(*PLATE_DIAM_JITTER) * px
    pr = plate_d / 2
    pcx = px / 2 + rng.uniform(-0.05, 0.05) * px
    pcy = px / 2 + rng.uniform(-0.05, 0.05) * px

    plate_tone = int(rng.integers(185, 250))
    rim_tone = int(np.clip(plate_tone - rng.integers(25, 60), 0, 255))

    d = ImageDraw.Draw(canvas)
    d.ellipse([pcx - pr, pcy - pr, pcx + pr, pcy + pr],
              fill=(plate_tone,) * 3, outline=(rim_tone,) * 3,
              width=max(1, int(pr * 0.06)))
    # Inner rim line: the scale fiducial the portion head is meant to exploit.
    ir = pr * 0.86
    d.ellipse([pcx - ir, pcy - ir, pcx + ir, pcy + ir], outline=(rim_tone,) * 3,
              width=max(1, int(pr * 0.03)))
    return canvas.filter(ImageFilter.GaussianBlur(0.4)), pcx, pcy, pr


def compose_one(food_img, rng, px=IMG_PX):
    """Render one plate scene. Returns (PIL image, area fraction, tier)."""
    canvas, pcx, pcy, pr = draw_plate(px, rng)

    # --- food at a sampled fraction of PLATE area (not frame area)
    f = sample_fraction(rng)
    plate_area = math.pi * pr ** 2
    food_area = f * plate_area
    food_r = math.sqrt(food_area / math.pi)

    # Offset the serving within the plate, but keep it on the plate.
    max_off = max(0.0, pr * 0.80 - food_r)
    ang = rng.uniform(0, 2 * math.pi)
    off = rng.uniform(0, max_off)
    fcx, fcy = pcx + off * math.cos(ang), pcy + off * math.sin(ang)

    patch_px = max(8, int(food_r * 2.6))
    patch = crop_texture(food_img, patch_px, rng)
    if rng.random() < 0.5:
        patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
    patch = patch.rotate(rng.uniform(0, 360), resample=Image.BILINEAR)

    layer = Image.new("RGB", (px, px))
    layer.paste(patch, (int(fcx - patch_px / 2), int(fcy - patch_px / 2)))

    mask = blob_mask(px, fcx, fcy, food_r, rng)
    canvas = Image.composite(layer, canvas, mask)

    # --- lighting variation (Section 3 asks for multiple lighting conditions)
    arr = np.asarray(canvas, dtype=np.float32)
    gain = rng.uniform(0.72, 1.30)
    warm = rng.uniform(-16, 16)
    arr[..., 0] += warm
    arr[..., 2] -= warm
    # Soft directional falloff, like a single overhead light source.
    yy, xx = np.mgrid[0:px, 0:px]
    lx, ly = rng.uniform(0, px), rng.uniform(0, px)
    dist = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2) / px
    arr *= (gain * (1.0 - 0.28 * dist))[..., None]
    arr += rng.normal(0, rng.uniform(1.5, 7.0), size=arr.shape)
    canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    return canvas, f, fraction_to_tier(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class-train", type=int, default=3000)
    ap.add_argument("--per-class-test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output dir (default data/composed)")
    args = ap.parse_args()

    print("Portion tier thresholds derived from the physical model:")
    print(f"  grams per unit area-fraction : {GRAMS_PER_UNIT_FRACTION:.1f} g")
    print(f"  small  : f < {F_SMALL_MAX:.3f}   (< {MASS_SMALL_MAX_G:.0f} g)")
    print(f"  medium : {F_SMALL_MAX:.3f} <= f < {F_MEDIUM_MAX:.3f} "
          f"({MASS_SMALL_MAX_G:.0f}-{MASS_MEDIUM_MAX_G:.0f} g)")
    print(f"  large  : f >= {F_MEDIUM_MAX:.3f}  (> {MASS_MEDIUM_MAX_G:.0f} g)\n")

    global OUT
    if args.out:
        OUT = pathlib.Path(args.out)
        if not OUT.is_absolute():
            OUT = REPO / OUT
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    tier_counts = {s: [0, 0, 0] for s in ("train", "test")}

    for split, n_per in (("train", args.per_class_train),
                         ("test", args.per_class_test)):
        for cls in CLASSES:
            src_dir = RAW / split / cls
            srcs = sorted(src_dir.glob("*.jpg"))
            if not srcs:
                print(f"!! no source images in {src_dir}", file=sys.stderr)
                continue

            # Seed per (split, class) so runs are reproducible and the train and
            # test generators never share a random stream.
            rng = np.random.default_rng(abs(hash((args.seed, split, cls))) % 2**32)
            dst = OUT / split / cls
            dst.mkdir(parents=True, exist_ok=True)

            for i in range(n_per):
                src = srcs[int(rng.integers(len(srcs)))]
                with Image.open(src) as im:
                    # Hand the FULL source to compose_one: crop_texture sizes the
                    # crop from the serving size so texture scale is preserved.
                    img, f, tier = compose_one(im.convert("RGB"), rng)
                fn = dst / f"{cls}_{i:05d}_t{tier}.jpg"
                img.save(fn, "JPEG", quality=90)
                tier_counts[split][tier] += 1
                index.append({
                    "path": str(fn.relative_to(REPO)),
                    "split": split, "cls": cls, "cls_idx": CLASSES.index(cls),
                    "tier": tier, "area_frac": round(f, 4),
                    "mass_g_est": round(fraction_to_mass(f), 1),
                    "src": str(src.relative_to(REPO)),
                })
            print(f"  {split}/{cls}: {n_per} composed")

    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    meta = {
        "classes": CLASSES, "portion_names": PORTION_NAMES, "img_px": IMG_PX,
        "f_small_max": F_SMALL_MAX, "f_medium_max": F_MEDIUM_MAX,
        "mass_small_max_g": MASS_SMALL_MAX_G,
        "mass_medium_max_g": MASS_MEDIUM_MAX_G,
        "grams_per_unit_fraction": GRAMS_PER_UNIT_FRACTION,
        "plate_diam_cm": PLATE_DIAM_CM,
        "mean_food_depth_cm": MEAN_FOOD_DEPTH_CM,
        "mean_density_g_cm3": MEAN_DENSITY_G_CM3,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))

    print(f"\ntotal composed: {len(index)}")
    for split in ("train", "test"):
        c = tier_counts[split]
        print(f"  {split} tier balance small/medium/large: {c}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
