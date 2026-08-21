"""
Task heads: 5-way food class + 3-way ordinal portion tier.

Both heads sit on the shared 6x6 backbone feature map (handoff Section 5, "two
heads off the shared backbone"). They are decoupled in the YOLO sense --
separate weights, no gradient path between them except through the backbone.

WHY THE TWO HEADS POOL DIFFERENTLY
-----------------------------------
This is the important design decision in the file, and it came out of a measured
failure. Originally both heads read from one shared global-average pool, and the
classifier stalled at ~0.34 validation accuracy while the portion head sat happily
above 0.90.

Global average pooling is exactly the right reduction for the portion head: the
tier is *defined* by what fraction of the plate the food covers, so averaging
activations over the whole grid is a direct measurement of extent.

It is the wrong reduction for the class head. Food covers only a minority of the
frame -- a "small" serving is well under a fifth of it -- so averaging over all
36 cells dilutes the food response with plate and tablecloth cells that carry no
class information at all. The signal the classifier needs is "is there a broccoli-
like response ANYWHERE on this plate", which is a max, not a mean.

So: class head -> global MAX pool (presence), portion head -> global AVG pool
(extent). Both ops are TFLite Micro builtins, so this costs nothing on-device.
"""

from tensorflow.keras import layers

CLASS_NAMES = ["chicken", "broccoli", "rice", "beef", "potato"]
PORTION_NAMES = ["small", "medium", "large"]


def add_heads(feature_map, num_classes=5, num_tiers=3, dropout=0.3):
    """
    Attach both heads to the shared feature map.

    KERNEL-CHOICE NOTE
    ------------------
    We use MaxPooling2D / AveragePooling2D sized to the full feature map rather
    than the GlobalMaxPooling2D / GlobalAveragePooling2D layers that express the
    same maths. The values are identical -- a pool window covering the whole grid
    IS the global reduction -- but the exported op is not:

        GlobalMaxPooling2D     -> REDUCE_MAX
        GlobalAveragePooling2D -> MEAN
        MaxPooling2D(6)        -> MAX_POOL_2D
        AveragePooling2D(6)    -> AVERAGE_POOL_2D

    The bundled TFLite Micro runtime (Harvard_TinyMLx, ~2021) is several years
    older than the converter producing these models. Its int8 REDUCE_MAX/MEAN
    kernels are far less exercised than the pooling kernels that every
    MobileNet-style reference model on this board uses, and on hardware they
    returned a *different* answer from the host interpreter for a byte-identical
    int8 input -- silently wrong, with Invoke() still reporting kTfLiteOk.
    Switching to the pooling ops made device and host agree exactly.
    """
    h, w = int(feature_map.shape[1]), int(feature_map.shape[2])
    ch = int(feature_map.shape[3])

    # --- class head: presence detection -> max over the grid
    c = layers.MaxPooling2D(pool_size=(h, w), name="cls_pool")(feature_map)
    # Reshape with a fully-specified target, NOT Flatten: Flatten emits
    # SHAPE/STRIDED_SLICE/PACK to compute the size at runtime, and dynamic-shape
    # ops are exactly what an old TFLM build handles worst. Reshape to a known
    # constant compiles to a single static RESHAPE.
    c = layers.Reshape((ch,), name="cls_flat")(c)
    if dropout > 0:
        c = layers.Dropout(dropout, name="cls_drop")(c)
    cls_out = layers.Dense(num_classes, activation="softmax", name="cls_out")(c)

    # --- portion head: extent measurement -> average over the grid
    p = layers.AveragePooling2D(pool_size=(h, w), name="portion_pool")(feature_map)
    p = layers.Reshape((ch,), name="portion_flat")(p)
    if dropout > 0:
        p = layers.Dropout(dropout, name="portion_drop")(p)
    portion_out = layers.Dense(num_tiers, activation="softmax",
                               name="portion_out")(p)

    # Softmax is baked into the graph so the int8 model emits calibrated
    # probabilities directly -- no exp() in fixed point on the MCU.
    return cls_out, portion_out
