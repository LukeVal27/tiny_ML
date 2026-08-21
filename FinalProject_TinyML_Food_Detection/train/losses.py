"""
Losses and metrics for the two heads.

The portion head is ordinal: small < medium < large. A plain softmax
cross-entropy treats those three as unrelated labels, so confusing small with
large costs exactly as much as confusing small with medium -- which is wrong for
a portion estimate, and is the thing the professor flagged.
"""

import tensorflow as tf


def ordinal_ce(num_tiers=3, penalty=0.6, name="ordinal_ce"):
    """
    Cross-entropy plus a distance-weighted term (handoff Section 4).

        L = -log p_y  +  penalty * E_{k~p}[ |k - y| ]

    The second term is the expected ordinal distance under the predicted
    distribution. It is fully differentiable in p (no argmax), and it pushes
    probability mass toward the true tier's *neighbourhood* rather than merely
    onto the true tier: putting mass on "large" when the truth is "small" costs
    twice what putting it on "medium" costs.

    penalty=0 recovers ordinary categorical cross-entropy, which makes the
    ablation in the writeup a one-line change.
    """
    k = tf.range(num_tiers, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

        y_idx = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)      # (B,)
        dist = tf.abs(k[None, :] - y_idx[:, None])                   # (B, K)
        expected_dist = tf.reduce_sum(y_pred * dist, axis=-1)        # (B,)

        return ce + penalty * expected_dist

    loss.__name__ = name
    return loss


def weighted_ce(class_weights, label_smoothing=0.05, name="weighted_ce"):
    """
    Categorical cross-entropy with a per-class weight on the TRUE class.

    Why this exists
    ---------------
    Training on real FoodSeg103 data, the chicken class collapsed: recall 0.063,
    with 315 of 600 chicken images predicted as beef and 204 as potato, and only
    59 of 3000 predictions being "chicken" at all. The composed corpus is
    perfectly balanced (3000 images per class), so this is NOT a frequency
    problem -- cooked chicken is simply golden-brown and sits between `steak`
    (dark brown, an easy 0.94 recall) and `potato` (golden). Faced with an
    ambiguous class and an easy neighbour, the network maximises accuracy by
    abandoning the hard one.

    Keras cannot apply `class_weight` to a multi-output model, so the weight is
    folded into the loss: each sample's cross-entropy is scaled by the weight of
    its true class, which raises the cost of surrendering a hard class.

    Weights are supplied by the caller (see train.py --class-weights), typically
    derived from a prior run's per-class recall so the pressure lands where the
    measured failure is.
    """
    w = tf.constant(class_weights, dtype=tf.float32)
    n = len(class_weights)

    def loss(y_true, y_pred):
        if label_smoothing > 0:
            y_true = y_true * (1.0 - label_smoothing) + label_smoothing / n
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        # Weight by the true class, recovered from the (possibly smoothed) target.
        sw = tf.reduce_sum(tf.one_hot(tf.argmax(y_true, axis=-1), n) * w, axis=-1)
        return ce * sw

    loss.__name__ = name
    return loss


class OrdinalError(tf.keras.metrics.Metric):
    """
    Mean |predicted tier - true tier|. Reported in Section 7 alongside the
    confusion matrix; 0 is perfect, and it separates "off by one tier" from
    "off by two" in a way plain accuracy cannot.
    """

    def __init__(self, name="ordinal_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        t = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
        p = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
        d = tf.abs(t - p)
        if sample_weight is not None:
            sw = tf.cast(tf.reshape(sample_weight, tf.shape(d)), tf.float32)
            self.total.assign_add(tf.reduce_sum(d * sw))
            self.count.assign_add(tf.reduce_sum(sw))
        else:
            self.total.assign_add(tf.reduce_sum(d))
            self.count.assign_add(tf.cast(tf.size(d), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class MacroF1(tf.keras.metrics.Metric):
    """Macro-averaged F1 (handoff Section 7's headline class metric)."""

    def __init__(self, num_classes=5, name="macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        t = tf.one_hot(tf.argmax(y_true, axis=-1), self.n)
        p = tf.one_hot(tf.argmax(y_pred, axis=-1), self.n)
        if sample_weight is not None:
            w = tf.reshape(tf.cast(sample_weight, tf.float32), (-1, 1))
            t, p = t * w, p * w
        self.tp.assign_add(tf.reduce_sum(t * p, axis=0))
        self.fp.assign_add(tf.reduce_sum((1 - t) * p, axis=0))
        self.fn.assign_add(tf.reduce_sum(t * (1 - p), axis=0))

    def result(self):
        f1 = tf.math.divide_no_nan(2 * self.tp, 2 * self.tp + self.fp + self.fn)
        return tf.reduce_mean(f1)

    def reset_state(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(tf.zeros((self.n,)))
