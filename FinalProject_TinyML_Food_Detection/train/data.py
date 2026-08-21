"""
tf.data pipelines over the composed plate dataset.

The composed set is the training domain: every image carries both a class label
and an exact portion tier, and it looks like what the camera will see. The raw
Food-101/ImageNet photos are used only as texture sources during composition,
not as training images -- feeding raw web photos to the classifier would train it
on a framing (full-bleed dish, no plate rim) that never occurs at deployment.
"""

import json
import pathlib

import numpy as np
import tensorflow as tf

from .augment import augment

REPO = pathlib.Path(__file__).resolve().parent.parent
COMPOSED = REPO / "data" / "composed"
REAL = REPO / "data" / "foodseg103_real"

CLASSES = ["chicken", "broccoli", "rice", "beef", "potato"]
NUM_CLASSES = len(CLASSES)
NUM_TIERS = 3

# Which corpus to train on.
#   synthetic  data/composed          -- Food-101 texture inside drawn blobs
#   real       data/foodseg103_real   -- FoodSeg103 mask cutouts on the same plate
#   both       concatenation of the two
SOURCES = ("synthetic", "real", "both")


def _dirs_for(source):
    if source == "synthetic":
        return [COMPOSED]
    if source == "real":
        return [REAL]
    if source == "both":
        return [COMPOSED, REAL]
    raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")


def load_index(source="synthetic"):
    """
    Index rows for the chosen corpus.

    Rows carry an absolute `path` already, so mixing corpora is just list
    concatenation. `src` (the originating photo/cutout) is preserved because the
    group-level split in make_dataset keys on it.
    """
    idx, meta = [], None
    for d in _dirs_for(source):
        rows = json.loads((d / "index.json").read_text())
        for r in rows:
            r.setdefault("corpus", d.name)
        idx += rows
        if meta is None:
            meta = json.loads((d / "meta.json").read_text())
    return idx, meta


def _split_arrays(idx, split):
    rows = [r for r in idx if r["split"] == split]
    paths = np.array([str(REPO / r["path"]) for r in rows])
    cls = np.array([r["cls_idx"] for r in rows], dtype=np.int32)
    tier = np.array([r["tier"] for r in rows], dtype=np.int32)
    return paths, cls, tier


def make_dataset(split, batch=64, img_px=96, channels=3, training=None,
                 shuffle_buf=4096, seed=0, val_frac=0.0, aug_level="mild",
                 source="synthetic"):
    """
    Build a tf.data.Dataset yielding (image, {"cls":onehot, "portion":onehot}).

    val_frac > 0 carves a validation slice out of the *train* split. We split by
    source photo, not by composed image: many composed images share one source
    photo, so an image-level split would put near-identical food textures on both
    sides and inflate validation scores (the split-hygiene point in Section 3).
    """
    idx, _ = load_index(source)
    training = (split == "train") if training is None else training

    if split in ("train", "val") and val_frac > 0:
        rows = [r for r in idx if r["split"] == "train"]
        # Deterministic group assignment keyed on the SOURCE photo.
        srcs = sorted({r["src"] for r in rows})
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(srcs))
        n_val = int(len(srcs) * val_frac)
        val_srcs = {srcs[i] for i in perm[:n_val]}
        want_val = split == "val"
        rows = [r for r in rows if (r["src"] in val_srcs) == want_val]
        paths = np.array([str(REPO / r["path"]) for r in rows])
        cls = np.array([r["cls_idx"] for r in rows], dtype=np.int32)
        tier = np.array([r["tier"] for r in rows], dtype=np.int32)
    else:
        paths, cls, tier = _split_arrays(idx, split)

    n = len(paths)
    ds = tf.data.Dataset.from_tensor_slices((paths, cls, tier))
    if training:
        ds = ds.shuffle(min(shuffle_buf, max(n, 1)), seed=seed,
                        reshuffle_each_iteration=True)

    def _load(path, c, t):
        raw = tf.io.read_file(path)
        img = tf.io.decode_jpeg(raw, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (img_px, img_px))
        img = augment(img, training=training, level=aug_level)
        if channels == 1:
            img = tf.image.rgb_to_grayscale(img)
        return img, {
            "cls": tf.one_hot(c, NUM_CLASSES),
            "portion": tf.one_hot(t, NUM_TIERS),
        }

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds, n


def class_weights(split="train", source="synthetic"):
    """Inverse-frequency weights over the chosen corpus."""
    idx, _ = load_index(source)
    cnt = np.zeros(NUM_CLASSES)
    for r in idx:
        if r["split"] == split:
            cnt[r["cls_idx"]] += 1
    w = cnt.sum() / (NUM_CLASSES * np.maximum(cnt, 1))
    return {i: float(w[i]) for i in range(NUM_CLASSES)}


def representative_dataset(n=300, img_px=96, channels=3, seed=0,
                           aug_level="mild", source="synthetic"):
    """
    Calibration generator for int8 conversion (handoff Section 6.1).

    Deliberately drawn from the TRAIN split with augmentation ON: the quantiser
    should see the same noisy, degraded, badly-white-balanced activations the
    device will actually produce, not clean images. Calibrating on clean data is
    a common way to lose several points of accuracy at int8.
    """
    ds, _ = make_dataset("train", batch=1, img_px=img_px, channels=channels,
                         training=True, seed=seed, aug_level=aug_level,
                         source=source)

    def gen():
        for i, (img, _lbl) in enumerate(ds.take(n)):
            yield [tf.cast(img, tf.float32)]

    return gen
