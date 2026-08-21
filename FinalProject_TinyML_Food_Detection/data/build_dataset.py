#!/usr/bin/env python3
"""
Build the 5-class food dataset from keyless public sources.

Sources (no API key required):
  * ethz/food101            -> chicken, rice, beef, potato
  * mlnomad/imnet1k_broccoli -> broccoli (ImageNet-1k synset n07714990)

Food-101 is fine-grained (101 dish classes); we map several dish classes onto
each of our 5 coarse ingredient classes. Images are stored at STORE_PX so the
training pipeline still has room for random-resized-crop down to 96x96.

Split hygiene (handoff Section 3): we keep Food-101's official train/test
split, which is curated to avoid the same dish photo appearing on both sides,
and additionally run a perceptual-hash dedup pass across the whole corpus so
near-duplicate photos cannot straddle splits.

Usage:
    python3 data/build_dataset.py
    python3 data/build_dataset.py --per-class 1200
"""

import argparse
import collections
import io
import json
import pathlib
import sys

# Food-101 dish class -> our coarse class.
#
# CHOSEN FOR VISUAL COHERENCE, not for maximum image count. The first version of
# this map merged every plausible dish into each coarse class, and it failed
# badly: chicken and rice both scored F1 = 0.000 because the model never
# predicted them at all. The reason was that "potato" spanned french_fries
# (yellow), poutine (brown gravy) and gnocchi (pale), so it covered the entire
# colour range that chicken (curry/quesadilla/wings) and rice (fried/risotto/
# paella) also occupy. Once the blob mask removes plate-level context, colour and
# texture are nearly all that remain, so three mutually-overlapping colour
# distributions collapse onto whichever class has the widest one.
#
# Each coarse class below is now a single tight appearance mode:
#   chicken  golden-brown fried, lumpy, bone-in
#   rice     pale fine-grained
#   beef     dark red-brown seared muscle (raw carpaccio/tartare dropped)
#   potato   yellow cut sticks (poutine/gnocchi dropped)
FOOD101_MAP = {
    "chicken_wings": "chicken",
    "risotto": "rice",
    "fried_rice": "rice",
    "steak": "beef",
    "filet_mignon": "beef",
    "prime_rib": "beef",
    "french_fries": "potato",
}

# FoodSeg103 class index -> our coarse class.
#
# Mask pixel values in EduardoPacheco/FoodSeg103 ARE the class index, so these
# integers are what we compare pixels against. The mapping does not come out
# clean, and every compromise below is deliberate and must be stated in the
# writeup rather than discovered later:
#
#   87 broccoli      clean 1:1.
#   66 rice          clean 1:1.
#   70 potato        clean, but `french fries` (index 3) is a SEPARATE class and
#                    is deliberately excluded -- fries are visually distinct from
#                    potato and folding them in would rebuild exactly the
#                    colour-overlap problem that drove chicken and rice to
#                    F1 = 0.000 under the old Food-101 mapping.
#   48 chicken duck  MERGED class. There is no pure-chicken label, so our
#                    "chicken" becomes "chicken or duck".
#   46 steak         There is NO beef class in FoodSeg103. Our "beef" becomes
#                    "steak" -- a narrowing, not a rename, and the single biggest
#                    semantic change in this dataset switch.
FOODSEG103_MAP = {
    87: "broccoli",
    66: "rice",
    70: "potato",
    48: "chicken",
    46: "beef",
}

# Excluded on purpose; kept here so the decision is visible rather than implicit.
FOODSEG103_EXCLUDED = {3: "french fries (kept out of `potato`)"}

CLASSES = ["chicken", "broccoli", "rice", "beef", "potato"]
STORE_PX = 160  # stored size; training crops to 96
REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "raw"


def prep(img, px=STORE_PX):
    """Center-crop to square, resize to px, force RGB."""
    from PIL import Image

    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((px, px), Image.LANCZOS)


def phash(img, size=8):
    """Tiny DCT-free perceptual hash: 8x8 grayscale vs. its own mean."""
    import numpy as np
    from PIL import Image

    g = np.asarray(img.convert("L").resize((size, size), Image.LANCZOS), dtype=float)
    bits = (g > g.mean()).flatten()
    return "".join("1" if b else "0" for b in bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1400,
                    help="cap of TRAIN images kept per coarse class")
    args = ap.parse_args()

    from datasets import load_dataset

    OUT.mkdir(parents=True, exist_ok=True)
    seen_hashes = {}          # phash -> "split/class/idx" of first occurrence
    counts = collections.defaultdict(int)
    dupes = 0
    manifest = []

    def emit(img, cls, split, tag):
        """Store one image unless it perceptually duplicates an earlier one."""
        nonlocal dupes
        img = prep(img)
        h = phash(img)
        if h in seen_hashes:
            dupes += 1
            return False
        seen_hashes[h] = f"{split}/{cls}"
        d = OUT / split / cls
        d.mkdir(parents=True, exist_ok=True)
        fn = d / f"{tag}.jpg"
        img.save(fn, "JPEG", quality=92)
        counts[(split, cls)] += 1
        manifest.append({"path": str(fn.relative_to(REPO)), "cls": cls,
                         "split": split, "src": tag, "phash": h})
        return True

    # ------------------------------------------------------------- Food-101
    print("Loading ethz/food101 (~5 GB on first run, cached afterwards) ...")
    for hf_split, our_split in (("train", "train"), ("validation", "test")):
        ds = load_dataset("ethz/food101", split=hf_split)
        names = ds.features["label"].names
        keep_idx = {names.index(k): v for k, v in FOOD101_MAP.items()}
        print(f"  {hf_split}: {len(ds)} rows -> filtering to "
              f"{len(keep_idx)} dish classes")

        sel = ds.filter(lambda l: l in keep_idx, input_columns=["label"],
                        num_proc=4)
        print(f"  {hf_split}: {len(sel)} rows survive filter")

        per_cls_cap = args.per_class if our_split == "train" else args.per_class // 3
        kept = collections.defaultdict(int)
        for i, row in enumerate(sel):
            cls = keep_idx[row["label"]]
            if kept[cls] >= per_cls_cap:
                continue
            if emit(row["image"], cls, our_split, f"f101_{names[row['label']]}_{i}"):
                kept[cls] += 1
            if i % 1000 == 0:
                print(f"    {i}/{len(sel)}  {dict(kept)}", flush=True)

    # -------------------------------------------------------------- broccoli
    print("\nLoading mlnomad/imnet1k_broccoli (~181 MB) ...")
    bds = load_dataset("mlnomad/imnet1k_broccoli", split="train")
    print(f"  {len(bds)} broccoli rows")
    # ImageNet has no official split here, so carve our own 75/25 deterministically.
    for i, row in enumerate(bds):
        split = "test" if i % 4 == 3 else "train"
        cap = args.per_class if split == "train" else args.per_class // 3
        if counts[(split, "broccoli")] >= cap:
            continue
        emit(row["image"], "broccoli", split, f"imnet_broccoli_{i}")

    # --------------------------------------------------------------- report
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print("\n" + "=" * 60)
    print("DATASET BUILD COMPLETE")
    print("=" * 60)
    print(f"near-duplicate images rejected: {dupes}")
    print(f"{'class':<12}{'train':>8}{'test':>8}{'total':>8}")
    print("-" * 60)
    tot_tr = tot_te = 0
    for c in CLASSES:
        tr, te = counts[("train", c)], counts[("test", c)]
        tot_tr += tr
        tot_te += te
        print(f"{c:<12}{tr:>8}{te:>8}{tr + te:>8}")
    print("-" * 60)
    print(f"{'TOTAL':<12}{tot_tr:>8}{tot_te:>8}{tot_tr + tot_te:>8}")
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
