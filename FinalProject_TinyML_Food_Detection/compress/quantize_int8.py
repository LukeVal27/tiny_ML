#!/usr/bin/env python3
"""
Post-training int8 quantization (handoff Section 6, step 1).

Produces a fully-integer TFLite model: int8 weights, int8 activations, and int8
input/output tensors. The int8 *input* matters as much as the weights here --
it means the Arduino sketch can hand the camera's bytes almost straight to the
interpreter instead of building a float buffer, which would cost 96*96*3*4 =
110 KB of arena we do not have.

Usage:
    python3 -m compress.quantize_int8 --ckpt results/checkpoints/a075_rgb.keras
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import tensorflow as tf

from model.model import to_batch1
from train.data import CLASSES, make_dataset, representative_dataset
from eval.metrics import evaluate_tflite, evaluate_keras, report

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
MODELS = RESULTS / "models"


def convert(model, rep_gen, full_int8=True):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if full_int8:
        conv.representative_dataset = rep_gen
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


def convert_fp32(model):
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def convert_drq(model):
    """Dynamic-range: weights int8, activations float. Useful as a middle point."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    return conv.convert()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--calib", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--expand", type=int, default=2)
    # Calibration and evaluation must run on the corpus the model was trained
    # on: quantising against the wrong distribution mis-sets every activation
    # range and silently costs accuracy.
    ap.add_argument("--source", default="synthetic",
                    choices=("synthetic", "real", "both"))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    ckpt = pathlib.Path(args.ckpt)
    tag = args.tag or ckpt.stem
    MODELS.mkdir(parents=True, exist_ok=True)

    print(f"loading {ckpt}")
    model = tf.keras.models.load_model(ckpt, compile=False)

    # Convert from a batch-1 clone so the exported graph has fully static shapes.
    export_model = to_batch1(model, alpha=args.alpha, expand=args.expand)

    test_ds, n_te = make_dataset("test", batch=64, channels=args.channels,
                                 training=False, source=args.source)

    # ---------------------------------------------------------- FP32 baseline
    print("\nevaluating FP32 Keras baseline ...")
    base = evaluate_keras(model, test_ds)
    report("FP32 (Keras)", base)

    variants = {}

    fp32_tfl = convert_fp32(export_model)
    variants["fp32_tflite"] = fp32_tfl

    print("\nconverting dynamic-range int8 ...")
    variants["drq"] = convert_drq(export_model)

    print(f"converting full int8 (calibrating on {args.calib} augmented "
          f"training images) ...")
    aug = "real" if args.source == "real" else "mild"
    rep = representative_dataset(n=args.calib, channels=args.channels,
                                 aug_level=aug, source=args.source)
    variants["int8"] = convert(export_model, rep, full_int8=True)

    rows = []
    for name, blob in variants.items():
        path = MODELS / f"{tag}_{name}.tflite"
        path.write_bytes(blob)
        print(f"\nevaluating {name} ({len(blob):,} bytes) ...")
        m = evaluate_tflite(blob, test_ds)
        report(name, m)
        rows.append({
            "variant": name, "bytes": len(blob),
            "kb": round(len(blob) / 1024, 1),
            "path": str(path.relative_to(REPO)), **m,
        })

    # ------------------------------------------------------------- comparison
    print("\n" + "=" * 88)
    print(f"COMPRESSION COMPARISON  [{tag}]   test n={n_te}")
    print("=" * 88)
    print(f"{'variant':<14}{'size KB':>9}{'vs fp32':>9}{'cls F1':>9}"
          f"{'cls acc':>9}{'por acc':>9}{'ord err':>9}")
    print("-" * 88)
    base_kb = rows[0]["kb"]
    print(f"{'FP32 (keras)':<14}{'-':>9}{'-':>9}{base['macro_f1']:>9.4f}"
          f"{base['cls_acc']:>9.4f}{base['portion_acc']:>9.4f}"
          f"{base['ordinal_error']:>9.4f}")
    for r in rows:
        print(f"{r['variant']:<14}{r['kb']:>9.1f}"
              f"{r['kb'] / base_kb:>8.2f}x{r['macro_f1']:>9.4f}"
              f"{r['cls_acc']:>9.4f}{r['portion_acc']:>9.4f}"
              f"{r['ordinal_error']:>9.4f}")
    print("=" * 88)

    d_f1 = rows[-1]["macro_f1"] - base["macro_f1"]
    print(f"int8 macro-F1 delta vs FP32: {d_f1:+.4f} "
          f"({'within' if abs(d_f1) < 0.05 else 'EXCEEDS'} the 5% budget)")

    out = {"tag": tag, "n_test": n_te, "fp32_keras": base, "variants": rows}
    (RESULTS / f"quantization_{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RESULTS / f'quantization_{tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
