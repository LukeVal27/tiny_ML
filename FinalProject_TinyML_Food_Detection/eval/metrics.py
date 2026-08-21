"""
Metrics for Section 7: macro-F1, portion confusion matrix, ordinal error.

evaluate_tflite() runs the actual .tflite interpreter rather than the Keras
graph, so the numbers reported for the int8 model include real quantisation
effects (including the int8 input rescale the device will perform).
"""

import numpy as np
import tensorflow as tf

CLASSES = ["chicken", "broccoli", "rice", "beef", "potato"]
PORTIONS = ["small", "medium", "large"]


def _macro_f1(y_true, y_pred, n):
    f1s = []
    for c in range(n):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s)), [float(x) for x in f1s]


def confusion(y_true, y_pred, n):
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def _summarize(cls_t, cls_p, por_t, por_p):
    macro_f1, per_class = _macro_f1(cls_t, cls_p, len(CLASSES))
    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class,
        "cls_acc": float(np.mean(cls_t == cls_p)),
        "portion_acc": float(np.mean(por_t == por_p)),
        "ordinal_error": float(np.mean(np.abs(por_t - por_p))),
        # Off-by-two is the failure the ordinal loss exists to suppress, so we
        # report it separately rather than hiding it inside the mean.
        "portion_off_by_two": float(np.mean(np.abs(por_t - por_p) == 2)),
        "cls_confusion": confusion(cls_t, cls_p, len(CLASSES)).tolist(),
        "portion_confusion": confusion(por_t, por_p, len(PORTIONS)).tolist(),
    }


def evaluate_keras(model, ds):
    ct, cp, pt, pp = [], [], [], []
    for x, y in ds:
        out = model(x, training=False)
        ct.append(np.argmax(y["cls"].numpy(), 1))
        pt.append(np.argmax(y["portion"].numpy(), 1))
        cp.append(np.argmax(out["cls"].numpy(), 1))
        pp.append(np.argmax(out["portion"].numpy(), 1))
    return _summarize(np.concatenate(ct), np.concatenate(cp),
                      np.concatenate(pt), np.concatenate(pp))


def evaluate_tflite(blob, ds):
    interp = tf.lite.Interpreter(model_content=blob)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()

    # Keras dict outputs do not preserve name order through conversion, so bind
    # the heads by their tensor width (5 = classes, 3 = tiers) instead of by
    # index. Guessing the order here is a classic way to silently swap heads.
    by_units = {}
    for o in outs:
        by_units[int(o["shape"][-1])] = o
    o_cls, o_por = by_units[len(CLASSES)], by_units[len(PORTIONS)]

    ct, cp, pt, pp = [], [], [], []
    for x, y in ds:
        xb = x.numpy()
        for i in range(xb.shape[0]):
            img = xb[i : i + 1]
            if inp["dtype"] == np.int8:
                scale, zp = inp["quantization"]
                img = np.clip(np.round(img / scale + zp), -128, 127).astype(np.int8)
            interp.set_tensor(inp["index"], img)
            interp.invoke()
            cp.append(int(np.argmax(interp.get_tensor(o_cls["index"])[0])))
            pp.append(int(np.argmax(interp.get_tensor(o_por["index"])[0])))
        ct.append(np.argmax(y["cls"].numpy(), 1))
        pt.append(np.argmax(y["portion"].numpy(), 1))

    return _summarize(np.concatenate(ct), np.array(cp),
                      np.concatenate(pt), np.array(pp))


def report(name, m):
    print(f"  [{name}] macro-F1 {m['macro_f1']:.4f} | cls acc {m['cls_acc']:.4f} "
          f"| portion acc {m['portion_acc']:.4f} | ord err {m['ordinal_error']:.4f} "
          f"| off-by-2 {m['portion_off_by_two']:.4f}")


def print_confusion(m, labels, title):
    print(f"\n{title}")
    w = max(len(x) for x in labels) + 1
    print(" " * w + "".join(f"{l[:7]:>8}" for l in labels) + "     recall")
    for i, row in enumerate(m):
        tot = sum(row)
        rec = row[i] / tot if tot else 0.0
        print(f"{labels[i]:<{w}}" + "".join(f"{v:>8}" for v in row)
              + f"     {rec:.3f}")
