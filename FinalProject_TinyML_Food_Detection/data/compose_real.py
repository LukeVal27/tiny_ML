#!/usr/bin/env python3
"""
Composite REAL FoodSeg103 cutouts onto the synthetic plate at controlled area
fractions.

This is the fix for the class head. The synthetic pipeline gave the model real
food *texture* inside a drawn elliptical blob; a colour-histogram baseline scored
0.510 against the CNN's 0.559, i.e. almost no shape or structure signal was
available to learn. These cutouts carry the food's true outline and true internal
structure, extracted from real plated photographs via FoodSeg103's masks.

What is deliberately kept from the synthetic pipeline
-----------------------------------------------------
Everything that makes the PORTION head work (0.9307 accuracy, off-by-two 0.03%):

  * draw_plate()          identical table + plate + rim rendering
  * PLATE_DIAM_JITTER     plate apparent size varies, so raw food-pixel area is
                          an ambiguous cue and the model must measure against the
                          rim
  * sample_fraction()     tiers equally likely
  * F_SMALL_MAX / F_MEDIUM_MAX and the gram model

The food's opaque mask area is scaled to exactly the sampled fraction of plate
area, so the portion label stays exact by construction. Only the food's
appearance changes from synthetic-blob to real-outline.

Lighting: real photographs already carry real scene lighting. Applying the
synthetic composer's full exposure model on top is precisely the stacking bug
that once crushed a large share of every batch to near-black and destroyed the
colour cue. A gentle global grade is still applied so plate and food read as
lit by the same source, but at a fraction of the synthetic range.

Usage:
    python3 -m data.compose_real --per-class-train 3000 --per-class-test 600
"""

import argparse
import collections
import json
import math
import pathlib
import sys

import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.compose_portions import (  # noqa: E402
    CLASSES, F_MEDIUM_MAX, F_SMALL_MAX, GRAMS_PER_UNIT_FRACTION, IMG_PX,
    PORTION_NAMES, draw_plate, fraction_to_mass, fraction_to_tier,
    sample_fraction,
)

CUTOUTS = REPO / "data" / "foodseg103_cutouts"
OUT = REPO / "data" / "foodseg103_real"

# Minimum opaque pixels a scaled cutout must retain. Below this the food is a
# few pixels of mush and the class is unlearnable no matter how real the source.
MIN_OPAQUE_PX = 24


def load_cutout(path):
    """Load an RGBA cutout and return (image, opaque_pixel_count)."""
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im.split()[-1])
    return im, int((a > 127).sum())


def place_cutout(canvas, cut, opaque_px, pcx, pcy, pr, frac, rng):
    """
    Scale `cut` so its opaque area equals `frac` of the plate area, then paste.

    Returns True on success. The scale factor comes straight from the area
    identity: opaque area grows with s^2, so s = sqrt(target / current).
    """
    plate_area = math.pi * pr ** 2
    target = frac * plate_area
    if opaque_px <= 0:
        return False

    s = math.sqrt(target / float(opaque_px))
    nw, nh = max(1, int(round(cut.width * s))), max(1, int(round(cut.height * s)))
    if nw * nh <= 0:
        return False

    # Downscale keeps real texture; extreme upscale would invent detail that is
    # not there and reintroduce the blur problem the texture-scale fix removed.
    if s > 2.2:
        return False

    resample = Image.LANCZOS if s < 1.0 else Image.BICUBIC
    cut = cut.resize((nw, nh), resample)
    if rng.random() < 0.5:
        cut = cut.transpose(Image.FLIP_LEFT_RIGHT)
    cut = cut.rotate(rng.uniform(0, 360), resample=Image.BICUBIC, expand=True)

    alpha = np.asarray(cut.split()[-1])
    if int((alpha > 127).sum()) < MIN_OPAQUE_PX:
        return False

    # A cutout far larger than the frame would land mostly off-canvas and the
    # plate would render empty or nearly so.
    if max(cut.width, cut.height) > 1.5 * canvas.width:
        return False

    # Bound the offset by the cutout's ACTUAL half-extent, not by the radius of
    # an equivalent-area disc. Masks like broccoli florets are sparse, so their
    # bounding box is much wider than their opaque area implies -- using the
    # area-equivalent radius let large servings drift off the plate entirely,
    # which produced empty-looking plates and food clipped at the frame edge.
    half = max(cut.width, cut.height) / 2.0
    max_off = max(0.0, pr * 0.85 - half)
    ang = rng.uniform(0, 2 * math.pi)
    off = rng.uniform(0, max_off)          # 0 when the food already fills the plate
    cx, cy = pcx + off * math.cos(ang), pcy + off * math.sin(ang)

    canvas.paste(cut, (int(cx - cut.width / 2), int(cy - cut.height / 2)), cut)
    return True


def grade(img, rng):
    """Gentle global light grade over plate AND food together."""
    a = np.asarray(img, dtype=np.float32)
    px = a.shape[0]
    a *= rng.uniform(0.88, 1.12)                      # mild exposure
    warm = rng.uniform(-7, 7)                          # mild colour temperature
    a[..., 0] += warm
    a[..., 2] -= warm
    yy, xx = np.mgrid[0:px, 0:px]                      # soft directional falloff
    lx, ly = rng.uniform(0, px), rng.uniform(0, px)
    dist = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2) / px
    a *= (1.0 - 0.12 * dist)[..., None]
    a += rng.normal(0, rng.uniform(1.0, 3.5), size=a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class-train", type=int, default=3000)
    ap.add_argument("--per-class-test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out) if args.out else OUT
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"tier thresholds reused: small f<{F_SMALL_MAX:.3f}  "
          f"medium <{F_MEDIUM_MAX:.3f}  ({GRAMS_PER_UNIT_FRACTION:.1f} g per "
          f"unit fraction)\n")

    index = []
    tier_counts = {s: [0, 0, 0] for s in ("train", "test")}
    skipped = collections.Counter()

    for split, n_per in (("train", args.per_class_train),
                         ("test", args.per_class_test)):
        for cls in CLASSES:
            src_dir = CUTOUTS / split / cls
            srcs = sorted(src_dir.glob("*.png"))
            if not srcs:
                print(f"!! no cutouts in {src_dir}", file=sys.stderr)
                continue

            rng = np.random.default_rng(
                abs(hash((args.seed, "real", split, cls))) % 2**32)
            dst = out_dir / split / cls
            dst.mkdir(parents=True, exist_ok=True)

            # Cache decoded cutouts: each is reused many times and decoding a
            # PNG per composite would dominate runtime.
            cache = {}
            made = 0
            attempts = 0
            while made < n_per and attempts < n_per * 6:
                attempts += 1
                p = srcs[int(rng.integers(len(srcs)))]
                if p not in cache:
                    cache[p] = load_cutout(p)
                cut, opq = cache[p]

                f = sample_fraction(rng)
                canvas, pcx, pcy, pr = draw_plate(IMG_PX, rng)
                if not place_cutout(canvas, cut, opq, pcx, pcy, pr, f, rng):
                    skipped[cls] += 1
                    continue

                img = grade(canvas, rng)
                tier = fraction_to_tier(f)
                fn = dst / f"{cls}_{made:05d}_t{tier}.jpg"
                img.save(fn, "JPEG", quality=90)
                tier_counts[split][tier] += 1
                index.append({
                    "path": str(fn.relative_to(REPO)), "split": split,
                    "cls": cls, "cls_idx": CLASSES.index(cls), "tier": tier,
                    "area_frac": round(f, 4),
                    "mass_g_est": round(fraction_to_mass(f), 1),
                    "src": str(p.relative_to(REPO)),
                })
                made += 1
            print(f"  {split}/{cls}: {made} composed "
                  f"from {len(srcs)} cutouts (skipped {skipped[cls]})")

    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    (out_dir / "meta.json").write_text(json.dumps({
        "classes": CLASSES, "portion_names": PORTION_NAMES, "img_px": IMG_PX,
        "f_small_max": F_SMALL_MAX, "f_medium_max": F_MEDIUM_MAX,
        "grams_per_unit_fraction": GRAMS_PER_UNIT_FRACTION,
        "source": "FoodSeg103 real mask cutouts on synthetic plate",
    }, indent=1))

    print(f"\ntotal composed: {len(index)}")
    for s in ("train", "test"):
        print(f"  {s} tiers s/m/l: {tier_counts[s]}")
    print(f"written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
