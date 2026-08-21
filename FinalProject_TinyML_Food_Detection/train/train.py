#!/usr/bin/env python3
"""
FP32 baseline training for the Tier 1 classifier + portion head.

Usage:
    python3 -m train.train                      # default alpha=0.75, RGB
    python3 -m train.train --alpha 1.0 --epochs 40
    python3 -m train.train --channels 1         # grayscale ablation
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import tensorflow as tf

from model.model import build_model, summarize
from .data import (CLASSES, NUM_CLASSES, NUM_TIERS, class_weights,
                   make_dataset)
from .losses import MacroF1, OrdinalError, ordinal_ce, weighted_ce

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
CKPT = REPO / "results" / "checkpoints"


def build_compiled(alpha, channels, expand, lr, portion_penalty, cls_w, por_w,
                   class_weights=None):
    model = build_model(input_shape=(96, 96, channels), alpha=alpha, expand=expand)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={
            "cls": (weighted_ce(class_weights) if class_weights else
                    tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)),
            "portion": ordinal_ce(NUM_TIERS, penalty=portion_penalty),
        },
        # The class head is the primary deliverable; the portion head is graded
        # but secondary, and it is the easier task, so it gets less weight.
        loss_weights={"cls": cls_w, "portion": por_w},
        metrics={
            "cls": [tf.keras.metrics.CategoricalAccuracy(name="acc"),
                    MacroF1(NUM_CLASSES)],
            "portion": [tf.keras.metrics.CategoricalAccuracy(name="acc"),
                        OrdinalError()],
        },
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--channels", type=int, default=3, choices=(1, 3))
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--portion-penalty", type=float, default=0.6)
    ap.add_argument("--cls-weight", type=float, default=1.0)
    ap.add_argument("--portion-weight", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--aug-level", default="mild",
                    choices=("none", "real", "mild", "full"))
    ap.add_argument("--source", default="synthetic",
                    choices=("synthetic", "real", "both"))
    ap.add_argument("--class-weights", default=None,
                    help="comma-separated per-class weights in CLASSES order; "
                         "counters a collapsed class (see losses.weighted_ce)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or f"a{args.alpha}_c{args.channels}"
    RESULTS.mkdir(exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(0)
    np.random.seed(0)

    train_ds, n_tr = make_dataset("train", batch=args.batch,
                                  channels=args.channels, val_frac=args.val_frac,
                                  aug_level=args.aug_level, source=args.source)
    val_ds, n_va = make_dataset("val", batch=args.batch, channels=args.channels,
                                training=False, val_frac=args.val_frac,
                                source=args.source)
    test_ds, n_te = make_dataset("test", batch=args.batch,
                                 channels=args.channels, training=False,
                                 source=args.source)

    print(f"train {n_tr} | val {n_va} | test {n_te}")
    print(f"source: {args.source} | class weights: "
          f"{class_weights(source=args.source)}")

    cw = None
    if args.class_weights:
        cw = [float(x) for x in args.class_weights.split(",")]
        assert len(cw) == NUM_CLASSES, f"need {NUM_CLASSES} weights, got {len(cw)}"
        print(f"class weights: {dict(zip(CLASSES, cw))}")
    model = build_compiled(args.alpha, args.channels, args.expand, args.lr,
                           args.portion_penalty, args.cls_weight,
                           args.portion_weight, class_weights=cw)
    info = summarize(model)

    ckpt_path = CKPT / f"{tag}.keras"
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_cls_macro_f1", mode="max",
            save_best_only=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_cls_macro_f1", mode="max", factor=0.4, patience=4,
            min_lr=1e-5, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_cls_macro_f1", mode="max", patience=10,
            restore_best_weights=True, verbose=1),
    ]

    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                     callbacks=cbs, verbose=2)
    train_s = time.time() - t0

    print("\nEvaluating FP32 baseline on the held-out test split ...")
    ev = model.evaluate(test_ds, verbose=0, return_dict=True)
    for k, v in ev.items():
        print(f"  {k:28s} {v:.4f}")

    model.save(ckpt_path)
    rec = {
        "tag": tag, "alpha": args.alpha, "channels": args.channels,
        "source": args.source, "aug_level": args.aug_level,
        "class_weights": args.class_weights,
        "expand": args.expand, "epochs_run": len(hist.history["loss"]),
        "train_seconds": round(train_s, 1),
        "params": info["params"], "peak_act_bytes": info["peak_act_bytes"],
        "n_train": n_tr, "n_val": n_va, "n_test": n_te,
        "test": {k: float(v) for k, v in ev.items()},
        "args": vars(args),
    }
    with (RESULTS / "train_runs.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\nsaved {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
