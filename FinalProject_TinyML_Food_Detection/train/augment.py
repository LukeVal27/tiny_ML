"""
Augmentation, with an explicit sim-to-real bias.

Two jobs here, and they are different:

1. Ordinary invariance augmentation (flips, rotations, mild colour jitter) so
   the classifier does not overfit 6.5k source photos.

2. *Camera degradation* -- making clean composed images look like they came off
   an OV7675, to close the sim-to-real gap the handoff wants quantified.

CALIBRATION NOTE (learned the hard way)
---------------------------------------
The first version of this file was far too aggressive and cost ~20 points of
macro-F1. The failure mode is worth recording because it is easy to repeat:
data/compose_portions.py ALREADY applies a per-image exposure gain, a warm/cool
colour shift and a directional light falloff. Stacking another multiplicative
brightness/gain/saturation/hue jitter on top pushed a large fraction of the batch
to near-black or blown-out, with extreme colour casts.

That is fatal here specifically because colour is the dominant class cue --
green broccoli vs. white rice vs. brown beef. Destroying colour left the portion
head healthy (area survives any colour transform) while the class head collapsed
to barely above chance. Symptom and cause lined up exactly.

So: the composer owns *scene* lighting, and this file owns only *sensor*
degradation, with the two kept deliberately disjoint and each mild on its own.

Geometric caution: we do NOT apply zoom or scale jitter. The portion label is
defined by food area relative to the plate rim, so any transform that rescales
food without rescaling the rim silently corrupts the label. Flips and 90-degree
rotations preserve the ratio; zooms do not.
"""

import tensorflow as tf

# Named levels so the ablation in the writeup is a single flag.
#   none  no augmentation
#   real  geometry + sensor degradation ONLY -- for FoodSeg103 cutouts, which
#         are real photographs and already carry real scene lighting and real
#         white balance. Adding photometric jitter on top is the same stacking
#         mistake documented below, just with a different first layer.
#   mild  real + gentle photometric jitter, for synthetic composites
#   full  mild with wider ranges
LEVELS = ("none", "real", "mild", "full")


def _rand(lo, hi):
    return tf.random.uniform([], lo, hi)


def camera_degrade(img, strength=1.0):
    """
    Simulate the OV7675 image path on a [0,1] float RGB tensor.

    Ordered to mirror the physical pipeline: optics blur -> sensor noise ->
    mild colour processing -> RGB565 quantisation. Gains here are intentionally
    small; the composer already varied exposure.
    """
    # --- soft plastic lens: approximate blur via down/up sampling.
    # Floor of 0.6 keeps ~58x58 detail at 96x96; the old 0.45 erased texture.
    if tf.random.uniform([]) < 0.45 * strength:
        h = tf.shape(img)[0]
        f = _rand(0.62, 0.92)
        small = tf.cast(tf.cast(h, tf.float32) * f, tf.int32)
        img = tf.image.resize(img, (small, small), method="area")
        img = tf.image.resize(img, (h, h), method="bilinear")

    # --- sensor read noise
    img = img + tf.random.normal(tf.shape(img), stddev=_rand(0.004, 0.022) * strength)

    # --- mild AWB error only. No exposure gain: the composer owns exposure.
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = img * tf.random.uniform([1, 1, 3], 1.0 - 0.06 * strength,
                                  1.0 + 0.06 * strength)

    # --- RGB565 on the wire: 5/6/5 bits. Cheap, realistic, colour-preserving.
    if tf.random.uniform([]) < 0.7:
        levels = tf.constant([31.0, 63.0, 31.0])
        img = tf.round(tf.clip_by_value(img, 0, 1) * levels) / levels

    return tf.clip_by_value(img, 0.0, 1.0)


def augment(img, training=True, level="mild"):
    """Training augmentation for one [0,1] float RGB image."""
    if not training or level == "none":
        return img

    full = level == "full"
    real = level == "real"

    # Geometry is free: it cannot damage the colour signal.
    img = tf.image.random_flip_left_right(img)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, tf.int32))

    if not real:
        # Photometric jitter kept deliberately gentle -- see CALIBRATION NOTE.
        img = tf.image.random_brightness(img, 0.10 if not full else 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.20)
        # Hue is the class signal itself. Touch it barely, if at all.
        img = tf.image.random_hue(img, 0.015 if not full else 0.025)
        img = tf.image.random_saturation(img, 0.85, 1.18)
        img = tf.clip_by_value(img, 0.0, 1.0)

    # Sensor degradation applies to every level: it models the OV7675 image
    # path, which sits downstream of whatever produced the picture.
    p = 0.55 if not full else 0.8
    if tf.random.uniform([]) < p:
        img = camera_degrade(img, strength=1.0 if not full else 1.4)

    return img
