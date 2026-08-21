"""
Assembles backbone + heads into the deployable Tier 1 model, and reports the
size/arena numbers we design against.
"""

import tensorflow as tf
from tensorflow.keras import Model

from .backbone import build_backbone
from .heads import CLASS_NAMES, PORTION_NAMES, add_heads


def build_model(input_shape=(96, 96, 3), alpha=1.0, expand=2, dropout=0.3,
                name="food_tinyml", batch_size=None):
    inputs, feature_map = build_backbone(input_shape, alpha=alpha, expand=expand,
                                        batch_size=batch_size)
    cls_out, portion_out = add_heads(
        feature_map, len(CLASS_NAMES), len(PORTION_NAMES), dropout=dropout
    )
    return Model(inputs, {"cls": cls_out, "portion": portion_out}, name=name)


def to_batch1(model, alpha=1.0, expand=2):
    """
    Rebuild `model` with a fixed batch of 1 and copy its weights across.

    Used only for export. Pooling/reshape layers carry no parameters, so this is
    numerically lossless; it exists purely to make every tensor shape static.
    """
    shape = tuple(model.input_shape[1:])
    clone = build_model(shape, alpha=alpha, expand=expand, batch_size=1)
    src = {l.name: l for l in model.layers if l.get_weights()}
    for l in clone.layers:
        if l.get_weights() and l.name in src:
            l.set_weights(src[l.name].get_weights())
    return clone


def peak_activation_bytes(model):
    """
    Crude but useful lower bound on the TFLM arena.

    TFLM reuses one buffer pool, so the binding constraint is the largest pair of
    simultaneously-live tensors -- in a straight-line convnet that is a layer's
    input plus its output. We report the max over layers, int8 (1 byte/elem).
    The real number comes from arena_used_bytes() on-device; this just tells us
    early whether an architecture is obviously doomed.
    """
    peak = 0
    worst = None
    for layer in model.layers:
        try:
            ins = layer.input if isinstance(layer.input, list) else [layer.input]
            outs = layer.output if isinstance(layer.output, list) else [layer.output]
        except AttributeError:
            continue

        def nbytes(t):
            shape = t.shape
            if shape is None or len(shape) < 2:
                return 0
            n = 1
            for d in shape[1:]:  # skip batch
                if d is None:
                    return 0
                n *= int(d)
            return n

        total = sum(nbytes(t) for t in ins) + sum(nbytes(t) for t in outs)
        if total > peak:
            peak, worst = total, layer.name
    return peak, worst


def summarize(model):
    params = model.count_params()
    peak, worst = peak_activation_bytes(model)
    print(f"model            : {model.name}")
    print(f"input            : {model.input_shape[1:]}")
    print(f"parameters       : {params:,}")
    print(f"est. int8 weights: {params / 1024:.1f} KB")
    print(f"peak activations : {peak / 1024:.1f} KB  (at layer '{worst}')")
    print(f"est. arena floor : {peak / 1024:.1f} KB + TFLM overhead (~4-10 KB)")
    return {"params": params, "peak_act_bytes": peak, "peak_layer": worst}


if __name__ == "__main__":
    import sys

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    ch = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    m = build_model(input_shape=(96, 96, ch), alpha=alpha)
    summarize(m)
