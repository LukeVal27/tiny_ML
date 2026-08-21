#!/usr/bin/env python3
"""
FoodSeg103 -> real food cutouts, in ONE pass over the dataset.

Why cutouts and not portion labels straight from the masks
----------------------------------------------------------
NEXT_TASK.md A2 proposed portion fraction = (pixels of class) / (non-background
pixels). That cannot measure a serving. In FoodSeg103 the plate and the table are
*background*, so non-background IS the food: a plate holding only rice scores 1.0
whether it is a spoonful or a mountain. Dividing by whole-image area instead just
measures how the photographer framed the shot, and these are web photos with
wildly varying zoom.

So we take the other half of what the masks offer. The mask gives us the food's
REAL outline and REAL texture, which is what the class head is starving for
(macro-F1 0.5493, barely ahead of a colour-histogram baseline at 0.510). We cut
the food out along its true mask and hand it to compose_portions.py, which pastes
it onto the existing synthetic plate at a *controlled* area fraction. Portion
ground truth stays exact and untouched; only the food's appearance becomes real.

This script also records the area statistics that demonstrate the A2 problem
empirically, so the decision is evidenced rather than asserted.

Single pass, by design: counts, area diagnostics and cutout extraction all happen
in one iteration, and rows are rejected on `classes_on_image` BEFORE the image or
mask is decoded — that skips decoding roughly 90% of the corpus.

Usage:
    python3 -m data.foodseg_cutouts
    python3 -m data.foodseg_cutouts --per-class 900 --min-px 64
"""

import argparse
import collections
import json
import pathlib
import sys

import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "foodseg103_cutouts"
RESULTS = REPO / "results"

sys.path.insert(0, str(REPO))
from data.build_dataset import FOODSEG103_MAP, FOODSEG103_EXCLUDED  # noqa: E402

WANTED = set(FOODSEG103_MAP)


def bbox_of(mask):
    """Tight bounding box of a boolean mask, or None if empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(x0), int(y0), int(x1) + 1, int(y1) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=900,
                    help="cap on cutouts kept per class per split")
    ap.add_argument("--min-px", type=int, default=56,
                    help="reject cutouts whose bbox is smaller than this")
    ap.add_argument("--min-fill", type=float, default=0.12,
                    help="reject cutouts whose mask fills less than this "
                         "fraction of their own bounding box")
    args = ap.parse_args()

    from datasets import load_dataset

    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    counts = collections.Counter()        # images containing each class
    kept = collections.Counter()          # cutouts actually written
    rejected = collections.Counter()      # why cutouts were dropped
    # Diagnostic for the A2 correction: for each class, its pixel share of the
    # whole image vs its share of non-background pixels.
    area_stats = collections.defaultdict(lambda: {"of_image": [], "of_food": []})

    for hf_split, our_split in (("train", "train"), ("validation", "test")):
        ds = load_dataset("EduardoPacheco/FoodSeg103", split=hf_split)
        print(f"[{hf_split}] {len(ds)} rows", flush=True)

        # Reject on the label list before touching pixels. This is the whole
        # reason the pass is cheap.
        hits = [i for i, cls in enumerate(ds["classes_on_image"])
                if WANTED & set(cls)]
        print(f"[{hf_split}] {len(hits)} rows contain one of our 5 classes",
              flush=True)

        for n, i in enumerate(hits):
            row = ds[i]
            present = WANTED & set(row["classes_on_image"])
            for idx in present:
                counts[(our_split, FOODSEG103_MAP[idx])] += 1

            img = row["image"].convert("RGB")
            lab = np.array(row["label"])
            food_px = int(np.count_nonzero(lab))      # non-background
            img_px = lab.size

            for idx in present:
                cls = FOODSEG103_MAP[idx]
                m = lab == idx
                n_px = int(m.sum())
                if n_px == 0:
                    continue

                area_stats[cls]["of_image"].append(n_px / img_px)
                if food_px:
                    area_stats[cls]["of_food"].append(n_px / food_px)

                if kept[(our_split, cls)] >= args.per_class:
                    continue

                bb = bbox_of(m)
                if bb is None:
                    continue
                x0, y0, x1, y1 = bb
                w, h = x1 - x0, y1 - y0
                if min(w, h) < args.min_px:
                    rejected[f"{cls}:too_small"] += 1
                    continue
                # A sparse, scattered mask crops to a big box that is mostly
                # holes; pasted on a plate it reads as confetti, not a serving.
                if n_px / float(w * h) < args.min_fill:
                    rejected[f"{cls}:too_sparse"] += 1
                    continue

                # RGBA cutout: colour from the photo, alpha from the true mask.
                # compose_portions.py scales this by its own alpha area, so the
                # real outline survives all the way onto the plate.
                crop = np.array(img)[y0:y1, x0:x1]
                alpha = (m[y0:y1, x0:x1] * 255).astype(np.uint8)
                rgba = np.dstack([crop, alpha])

                d = OUT / our_split / cls
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgba, mode="RGBA").save(
                    d / f"fs103_{row['id']}_{idx}.png")
                kept[(our_split, cls)] += 1

            if n % 200 == 0:
                print(f"  {hf_split} {n}/{len(hits)} kept={dict(kept)}",
                      flush=True)

    # ------------------------------------------------------------- reporting
    classes = sorted(set(FOODSEG103_MAP.values()))
    summary = {
        "mapping": {str(k): v for k, v in FOODSEG103_MAP.items()},
        "excluded": {str(k): v for k, v in FOODSEG103_EXCLUDED.items()},
        "images_containing_class": {
            s: {c: counts[(s, c)] for c in classes} for s in ("train", "test")
        },
        "cutouts_written": {
            s: {c: kept[(s, c)] for c in classes} for s in ("train", "test")
        },
        "rejected": dict(rejected),
        "params": vars(args),
    }

    # The evidence for the A2 correction, in numbers.
    diag = {}
    for c in classes:
        oi = area_stats[c]["of_image"]
        of = area_stats[c]["of_food"]
        if oi:
            diag[c] = {
                "n": len(oi),
                "median_share_of_image": round(float(np.median(oi)), 4),
                "median_share_of_food": round(float(np.median(of)), 4),
                "frac_where_share_of_food_over_0.9": round(
                    float(np.mean(np.array(of) > 0.9)), 4),
            }
    summary["area_diagnostics"] = diag

    (RESULTS / "foodseg103_counts.json").write_text(json.dumps(summary, indent=1))
    (RESULTS / "foodseg103_mapping.json").write_text(json.dumps(
        {"mapping": summary["mapping"], "excluded": summary["excluded"],
         "note": "beef == steak (no beef class exists); chicken == chicken duck "
                 "(merged label). Deliberate, logged decisions."}, indent=1))

    print("\n" + "=" * 68)
    print("FOODSEG103 INGEST COMPLETE")
    print("=" * 68)
    print(f"{'class':<10}{'train img':>11}{'test img':>10}"
          f"{'train cut':>11}{'test cut':>10}")
    print("-" * 68)
    for c in classes:
        print(f"{c:<10}{counts[('train', c)]:>11}{counts[('test', c)]:>10}"
              f"{kept[('train', c)]:>11}{kept[('test', c)]:>10}")
    print("-" * 68)
    if rejected:
        print("rejected:", dict(rejected))
    print("\nArea diagnostics (does the A2 formula carry portion signal?)")
    for c, d in diag.items():
        print(f"  {c:<10} share_of_food median={d['median_share_of_food']:.3f} "
              f"| >0.9 in {d['frac_where_share_of_food_over_0.9'] * 100:.1f}% "
              f"of images | share_of_image median={d['median_share_of_image']:.3f}")
    print(f"\nwrote {RESULTS / 'foodseg103_counts.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
