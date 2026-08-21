#!/usr/bin/env python3
"""
Why does the portion head return `large` at 0.9961 on every camera frame?

46 live captures, two sessions, two framings, one lone broccoli floret held high
above the plate -- all `large`, and all at exactly 0.9961 = 255/256, the largest
value an int8 softmax can represent. Bit-identical output across wildly different
scenes is a pinned tensor, not a confident model.

Framing is already ruled out on the bench (the floret test). This runs the same
model on the host so the remaining question can be answered without a flash
cycle: does the saturation follow the *pixels*?

The test has three parts:

  CONTROL     training plates through the host interpreter. The portion head must
              vary here, or the harness itself is wrong and nothing else means
              anything.
  CAMERA      results/raw_frame.bin, decoded and downsampled byte-for-byte the
              way classifier_tier1.ino does it. If this pins at `large`, the
              fault is in the image, not the board.
  INTERVENTION candidate normalisations applied to that same frame. Whichever one
              unpins the portion head names the fix.

The mechanism under test comes straight from the design decision in CLAUDE.md:
the two heads pool differently. The class head max-pools, and a max is carried by
the peak food response, which survives a washed-out frame -- the class head is
in fact fine on camera input (0.850). The portion head average-pools, and an
average moves with any global offset. That asymmetry predicts exactly what we
see.

Usage:
    python3 harness/portion_probe.py
"""

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
MODEL = REPO / "results" / "models" / "real_cw_int8.tflite"
FRAME = REPO / "results" / "raw_frame.bin"
INDEX = REPO / "data" / "foodseg103_real" / "index.json"
OUT = REPO / "results" / "portion_probe.json"

CLASSES = ["chicken", "broccoli", "rice", "beef", "potato"]
PORTIONS = ["small", "medium", "large"]

# Camera + sketch geometry, mirrored from classifier_tier1.ino.
CAM_W, CAM_H = 176, 144
SQ = 144
SQ_X0, SQ_Y0 = (CAM_W - SQ) // 2, (CAM_H - SQ) // 2
IN_W = IN_H = 96


def decode565(raw):
    """
    RGB565 -> uint8 RGB, big-endian (hi<<8)|lo.

    Byte order is settled: verified twice on-device, against Camera.testPattern()
    and against a real scene. Do not "fix" this -- see CLAUDE.md.
    """
    b = np.frombuffer(raw, dtype=np.uint8).reshape(CAM_H, CAM_W, 2).astype(np.uint16)
    v = (b[:, :, 0] << 8) | b[:, :, 1]
    r = ((v >> 11) & 0x1F) << 3
    g = ((v >> 5) & 0x3F) << 2
    bl = (v & 0x1F) << 3
    return np.stack([r, g, bl], -1).astype(np.uint8)


def downsample(rgb):
    """
    The sketch's downsampleToInput(): 144x144 centre square -> 96x96 by a 2x2 box
    average taken at the 1.5x sample point. Reproduced rather than approximated,
    because the whole point is to feed the model the bytes the board feeds it.
    """
    out = np.zeros((IN_H, IN_W, 3), np.float32)
    for oy in range(IN_H):
        sy = SQ_Y0 + (oy * 3) // 2
        for ox in range(IN_W):
            sx = SQ_X0 + (ox * 3) // 2
            ys = [min(sy + d, CAM_H - 1) for d in (0, 1)]
            xs = [min(sx + d, CAM_W - 1) for d in (0, 1)]
            out[oy, ox] = rgb[np.ix_(ys, xs)].reshape(-1, 3).mean(0)
    return out / 255.0


class Model:
    def __init__(self, path):
        import tensorflow as tf

        self.interp = tf.lite.Interpreter(model_content=path.read_bytes())
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        # Bind heads by tensor width (5 = class, 3 = portion), never by index --
        # the converter makes no ordering promise. Same rule as eval/metrics.py.
        by_units = {int(o["shape"][-1]): o for o in self.interp.get_output_details()}
        self.o_cls, self.o_por = by_units[len(CLASSES)], by_units[len(PORTIONS)]

    def _get(self, det):
        """
        Dequantise: real = (q - zero_point) * scale.

        Skipping this is not harmless. The raw int8 softmax spans [-128, 127],
        so a three-way head can easily sum to a NEGATIVE number -- and dividing
        by it silently turns every argmax into an argmin. That inverted the
        control set to 0/6 before this was fixed.
        """
        q = self.interp.get_tensor(det["index"])[0].astype(np.float32)
        s, z = det["quantization"]
        return (q - z) * s if s else q

    def __call__(self, img01):
        x = img01[None].astype(np.float32)
        if self.inp["dtype"] == np.int8:
            s, z = self.inp["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        return self._get(self.o_cls), self._get(self.o_por)


def softmax_report(cls, por):
    # Already probabilities once dequantised -- do not renormalise.
    return {
        "cls": CLASSES[int(np.argmax(cls))],
        "cls_conf": round(float(cls.max()), 4),
        "portion": PORTIONS[int(np.argmax(por))],
        "portion_conf": round(float(por.max()), 4),
        "portion_vec": [round(float(v), 4) for v in por],
    }


def line(tag, r, extra=""):
    print(f"  {tag:<22} {r['cls']:<9} {r['cls_conf']:.3f}   "
          f"portion {r['portion']:<7} {r['portion_conf']:.4f}   "
          f"{r['portion_vec']} {extra}")


# ------------------------------------------------------------------ candidates
# Each returns a new [0,1] image. These are the plausible ways a washed-out frame
# could be brought back toward the training distribution, cheapest first -- all
# are a handful of int ops per pixel, i.e. affordable in downsampleToInput().

def ident(img, ref):
    return img


def grey_world(img, ref):
    """Per-channel gain so the channel means match. Classic white balance."""
    m = img.reshape(-1, 3).mean(0)
    return np.clip(img * (m.mean() / np.maximum(m, 1e-6)), 0, 1)


def stretch(img, ref):
    """Per-channel percentile contrast stretch -- fixes a squashed dynamic range."""
    out = np.empty_like(img)
    for c in range(3):
        lo, hi = np.percentile(img[..., c], (2, 98))
        out[..., c] = np.clip((img[..., c] - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def match_ref(img, ref):
    """Force per-channel mean and std onto the training corpus statistics."""
    out = np.empty_like(img)
    for c in range(3):
        s = img[..., c].std()
        out[..., c] = (img[..., c] - img[..., c].mean()) / max(s, 1e-6)
        out[..., c] = out[..., c] * ref["std"][c] + ref["mean"][c]
    return np.clip(out, 0, 1)


CANDIDATES = [("as shipped", ident), ("grey-world WB", grey_world),
              ("contrast stretch", stretch), ("match train mean/std", match_ref)]


def main():
    for p in (MODEL, FRAME, INDEX):
        if not p.exists():
            print(f"missing: {p}")
            return 1

    from PIL import Image

    model = Model(MODEL)
    rows = json.loads(INDEX.read_text())

    # ---------------------------------------------------------------- CONTROL
    # Six test plates, two per tier, so a pinned head is obvious immediately.
    print("\nCONTROL — training plates (the head must vary here)")
    # Shuffled, not the first six -- the index is ordered by class, so taking
    # from the top yields six chicken plates and tests one corner of the corpus.
    test = [r for r in rows if r["split"] == "test"]
    np.random.default_rng(0).shuffle(test)
    picks, seen = [], {}
    for r in test:
        if seen.get(r["tier"], 0) < 2:
            seen[r["tier"]] = seen.get(r["tier"], 0) + 1
            picks.append(r)
        if len(picks) == 6:
            break

    ctrl, imgs = [], []
    for r in picks:
        img = np.asarray(Image.open(REPO / r["path"]).convert("RGB"),
                         np.float32) / 255.0
        imgs.append(img)
        res = softmax_report(*model(img))
        res["truth"] = PORTIONS[r["tier"]]
        ctrl.append(res)
        line(f"{r['cls']}/{PORTIONS[r['tier']]}", res,
             "OK" if res["portion"] == PORTIONS[r["tier"]] else "MISS")

    # Gate on CORRECTNESS, not variety. An earlier version only checked that the
    # predictions differed from each other, and cheerfully passed a harness whose
    # argmax was inverted -- it was 0/6 and still called itself sane.
    n_ok = sum(1 for c in ctrl if c["portion"] == c["truth"])
    print(f"  -> {n_ok}/6 correct (model scores 0.9187 on the full test set)", end="  ")
    if n_ok < 4:
        print("\n     HARNESS BROKEN — fix this before reading anything below.")
        return 1
    print("(harness sane)")

    stack = np.stack(imgs).reshape(-1, 3)
    ref = {"mean": stack.mean(0).tolist(), "std": stack.std(0).tolist()}

    # ----------------------------------------------------------------- CAMERA
    print("\nCAMERA — results/raw_frame.bin, downsampled as the sketch does")
    cam = downsample(decode565(FRAME.read_bytes()))
    cam_mean = cam.reshape(-1, 3).mean(0)
    cam_std = cam.reshape(-1, 3).std(0)
    print(f"  train mean RGB {np.round(ref['mean'], 3)}  std {np.round(ref['std'], 3)}")
    print(f"  cam   mean RGB {np.round(cam_mean, 3)}  std {np.round(cam_std, 3)}")

    # ----------------------------------------------------------- INTERVENTION
    print("\nINTERVENTION — same frame, each normalisation")
    interv = {}
    for name, fn in CANDIDATES:
        res = softmax_report(*model(fn(cam, ref)))
        interv[name] = res
        line(name, res)

    unpinned = [k for k, v in interv.items()
                if not (v["portion"] == "large" and v["portion_conf"] > 0.99)]
    print()
    if unpinned:
        print(f"  -> UNPINNED by: {', '.join(unpinned)}")
        print("     The saturation follows the pixels. Fixable in preprocessing.")
    else:
        print("  -> still pinned under every normalisation.")
        print("     Not a simple photometric offset — report as a measured failure.")

    # ------------------------------------------------------------------ SCALE
    # Photometrics are exonerated, so geometry is what is left. The portion tier
    # is defined against the plate rim, and training only ever showed the plate
    # at 62-92% of frame width (PLATE_DIAM_JITTER). Both ways out of that band
    # are untested: too close and the rim leaves the frame, too far and the plate
    # is smaller than anything the model has seen. The bench saw `large` at both
    # extremes, so sweep a KNOWN-CORRECT plate through the whole range and find
    # where the head lets go.
    print("\nSCALE — a known-good 'small' plate, zoomed about its centre")
    small = next(r for r in picks if r["tier"] == 0)
    base = Image.open(REPO / small["path"]).convert("RGB")
    sweep = []
    for z in (0.45, 0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0):
        if z >= 1.0:                      # zoom in: plate grows past the rim
            c = int(round(IN_W / z))
            o = (IN_W - c) // 2
            img = base.crop((o, o, o + c, o + c)).resize((IN_W, IN_H), Image.BILINEAR)
        else:                             # zoom out: plate shrinks, pad the edge
            c = int(round(IN_W * z))
            canvas = Image.new("RGB", (IN_W, IN_H), tuple(
                np.asarray(base).reshape(-1, 3)[:IN_W].mean(0).astype(int)))
            canvas.paste(base.resize((c, c), Image.BILINEAR),
                         ((IN_W - c) // 2, (IN_H - c) // 2))
            img = canvas
        res = softmax_report(*model(np.asarray(img, np.float32) / 255.0))
        res["zoom"] = z
        sweep.append(res)
        band = "in training band" if 0.62 / 0.77 <= z <= 0.92 / 0.77 else ""
        line(f"zoom x{z:.2f}", res,
             ("OK" if res["portion"] == "small" else "-> " + res["portion"]) +
             f"  {band}")

    held = [s["zoom"] for s in sweep if s["portion"] == "small"]
    print(f"\n  -> holds 'small' for zoom {min(held):.2f}-{max(held):.2f}"
          if held else "\n  -> never holds 'small'")

    OUT.write_text(json.dumps(
        {"control": ctrl, "control_correct": n_ok,
         "scale_sweep": sweep,
         "train_stats": ref,
         "camera_stats": {"mean": cam_mean.tolist(), "std": cam_std.tolist()},
         "interventions": interv, "unpinned_by": unpinned}, indent=2))
    print(f"\nwritten: {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
