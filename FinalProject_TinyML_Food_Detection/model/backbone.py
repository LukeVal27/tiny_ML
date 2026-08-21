"""
Depthwise-separable backbone for the Tier 1 food classifier.

Design constraints (handoff Sections 1 and 5):
  * input 96x96, RGB if the RAM gate allows it
  * every op must exist in TFLite Micro's builtin set, so: Conv2D, DepthwiseConv2D,
    ReLU6, BatchNorm (folds into the conv at convert time), GlobalAveragePooling,
    FullyConnected, Softmax. No squeeze-excite, no hard-swish, no FPN.
  * narrow channels and few stages -- we are targeting < 150 KB int8, not ImageNet.

The peak TFLM arena is driven by the two largest simultaneously-live tensors,
which here is the stem output plus its input. Keeping the stem stride-2 and
narrow is the single most effective arena lever, so STEM_CH is deliberately small.
"""

import tensorflow as tf
from tensorflow.keras import layers

# Keras defaults to 0.99. At ~200 steps/epoch the moving mean/variance lag far
# behind the live batch statistics, so train and inference modes disagree and
# validation metrics swing wildly epoch to epoch. 0.9 converges fast enough for
# a run this short.
BN_MOMENTUM = 0.9


def _conv_bn(x, filters, kernel, stride, name):
    """Standard conv -> BN -> ReLU6. BN folds into the conv during int8 convert."""
    x = layers.Conv2D(
        filters, kernel, strides=stride, padding="same", use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_bn")(x)
    return layers.ReLU(max_value=6.0, name=f"{name}_relu")(x)


def _ds_block(x, filters, stride, name, expand=1):
    """
    Depthwise-separable block: (optional 1x1 expand) -> 3x3 depthwise -> 1x1 project.

    expand=1 gives the plain MobileNetV1 block, which is the cheapest thing that
    still works. expand>1 gives a V2-style inverted bottleneck; we allow it because
    at these tiny widths a little expansion buys real accuracy for few parameters.
    """
    inp = x
    in_ch = x.shape[-1]

    if expand > 1:
        x = _conv_bn(x, in_ch * expand, 1, 1, f"{name}_expand")

    x = layers.DepthwiseConv2D(
        3, strides=stride, padding="same", use_bias=False, name=f"{name}_dw",
    )(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_dw_bn")(x)
    x = layers.ReLU(max_value=6.0, name=f"{name}_dw_relu")(x)

    # Linear projection (no activation) -- the V2 insight: ReLU on a narrow
    # tensor destroys information, and it costs us nothing to leave it off.
    x = layers.Conv2D(
        filters, 1, strides=1, padding="same", use_bias=False, name=f"{name}_pw",
    )(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_pw_bn")(x)

    # Residual only when shapes already match; avoids a projection shortcut.
    if stride == 1 and in_ch == filters:
        x = layers.Add(name=f"{name}_add")([inp, x])
    return x


def build_backbone(input_shape=(96, 96, 3), alpha=1.0, expand=2,
                   batch_size=None):
    """
    Returns (inputs, feature_map) where feature_map is the final 6x6 grid.

    alpha scales every channel width, so we can trade size for accuracy after
    seeing the first measured numbers rather than guessing up front.
    """
    def ch(n):
        # Round to a multiple of 8: keeps int8 kernels well-aligned and avoids
        # awkward tensor shapes that inflate the arena.
        return max(8, int(round(n * alpha / 8)) * 8)

    # batch_size=1 for export: with a None batch, Keras Reshape/Flatten emit
    # SHAPE/STRIDED_SLICE/PACK to compute sizes at runtime. Pinning the batch
    # makes every shape static, so the graph is pure conv/pool/FC -- which is
    # what TFLite Micro handles best.
    inputs = layers.Input(shape=input_shape, batch_size=batch_size,
                          name="image")

    # Stem: immediately halve spatial size. 96 -> 48.
    x = _conv_bn(inputs, ch(16), 3, 2, "stem")

    #  48 -> 24
    x = _ds_block(x, ch(32), 2, "b1", expand=1)  # no expand yet: input is tiny
    x = _ds_block(x, ch(32), 1, "b2", expand=expand)

    #  24 -> 12
    x = _ds_block(x, ch(64), 2, "b3", expand=expand)
    x = _ds_block(x, ch(64), 1, "b4", expand=expand)

    #  12 -> 6
    x = _ds_block(x, ch(128), 2, "b5", expand=expand)
    x = _ds_block(x, ch(128), 1, "b6", expand=expand)

    # 6 -> 6, final widening before pooling gives the heads a richer embedding
    # for very little compute (1x1 conv on a 6x6 grid).
    x = _conv_bn(x, ch(192), 1, 1, "head_conv")

    # Return the 6x6 feature MAP, not a pooled vector: the two heads need
    # different spatial reductions (see model/heads.py).
    return inputs, x
