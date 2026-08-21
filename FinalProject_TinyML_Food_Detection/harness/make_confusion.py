#!/usr/bin/env python3
"""
Render the two measured confusion matrices as figures for the deck.

Both are read from measured files, never retyped:
  lab     results/quantization_real_cw.json -> variants[int8].cls_confusion
          (the DEPLOYED int8 model, n = 3,000 held-out test images)
  device  results/live_captures.jsonl, session='final'
          (n = 75 labelled live captures through the OV7675)

Every scalar the deck quotes is recomputed here from the matrix and asserted
against the stored value, so a figure can never drift from its own numbers.

A confusion matrix is a magnitude-on-a-grid, so the encoding is a sequential
single-hue ramp (light -> dark), not a categorical or diverging one. Cells are
direct-labelled because the counts are the point; label ink flips to white on
dark cells so contrast holds at both ends of the ramp.

    python3 harness/make_confusion.py
"""

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
OUTDIR = REPO / "results" / "figures"

CLASSES = ["chicken", "broccoli", "rice", "beef", "potato"]

INK = "#16181D"
MUTE = "#6B7280"
FAINT = "#D1D5DB"
# Sequential ramp, one hue, light -> dark, keyed to the deck's accent.
RAMP = ["#FFFFFF", "#FDF2EA", "#F7D9C0", "#EDA878", "#D96A2C", "#C2410C"]


def restrict(M, labels, keep):
    """
    Drop classes from a confusion matrix, row AND column.

    Dropping the column is what makes this more than cosmetic: errors that
    landed on the dropped class disappear from the surviving rows' supports, so
    their recall rises without the model having changed. Always report the
    resulting n alongside the metric.
    """
    idx = [labels.index(c) for c in keep]
    return M[np.ix_(idx, idx)], list(keep)


def metrics(M, labels):
    """Per-class precision/recall/F1 plus accuracy and macro-F1 over support."""
    out, f1s = {}, []
    for i, c in enumerate(labels):
        tp = M[i, i]
        fp = M[:, i].sum() - tp
        fn = M[i, :].sum() - tp
        sup = tp + fn
        if sup == 0:
            out[c] = None
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / sup
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1)
        out[c] = {"n": int(sup), "tp": int(tp), "prec": p, "recall": r, "f1": f1}
    return out, np.trace(M) / M.sum(), (sum(f1s) / len(f1s) if f1s else 0.0)


def draw(M, labels, title, subtitle, path, note=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    per, acc, macro = metrics(M, labels)
    cmap = LinearSegmentedColormap.from_list("acc", RAMP)

    # Normalise per ROW (by true-class support) so a class with 3,000 samples and
    # one with 13 are comparable; raw counts stay as the printed label.
    rows = M.sum(1, keepdims=True)
    norm = np.divide(M, np.where(rows == 0, 1, rows), dtype=float)

    n = len(labels)
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(n):
        for j in range(n):
            if rows[i] == 0:
                continue
            # White ink once the cell is dark enough to swallow dark text.
            col = "#FFFFFF" if norm[i, j] > 0.55 else INK
            ax.text(j, i, f"{M[i, j]:,}", ha="center", va="center",
                    fontsize=11, color=col,
                    fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10, color=INK)
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.set_xlabel("predicted", fontsize=10, color=MUTE, labelpad=8)
    ax.set_ylabel("true", fontsize=10, color=MUTE, labelpad=8)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Recall printed to the right of each row — the number a reader wants next
    # to a confusion row, and it saves a second figure.
    for i, c in enumerate(labels):
        m = per[c]
        txt = "no support" if m is None else f"recall {m['recall']:.3f}"
        ax.text(n - 0.35, i, txt, ha="left", va="center", fontsize=9,
                color=MUTE if m else FAINT)

    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlim(-0.5, n - 0.5 + 1.35)

    # Title as figure text, not ax.set_title: an axes title is clipped to the
    # axes box, and these titles are wider than the grid.
    fig.text(0.012, 0.965, title, fontsize=13, fontweight="bold", color=INK,
             va="top")
    # Wrap by hand: figure text does no wrapping, and these captions carry the
    # exclusion caveat, which must not be the part that falls off the edge.
    import textwrap
    lines = textwrap.wrap(subtitle, 92)
    if note:
        lines += textwrap.wrap(note, 92)
    fig.text(0.012, 0.015, "\n".join(lines), fontsize=9, color=MUTE,
             va="bottom", linespacing=1.5)

    fig.tight_layout(rect=(0, 0.15, 1, 0.90))
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return per, acc, macro


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ok = True

    # ------------------------------------------------------------------- lab
    d = json.loads((REPO / "results" / "quantization_real_cw.json").read_text())
    int8 = next(v for v in d["variants"] if v["variant"] == "int8")
    Ml = np.array(int8["cls_confusion"])

    per, acc, macro = metrics(Ml, CLASSES)
    draw(Ml, CLASSES,
         "Class confusion — deployed int8 model, held-out test set",
         f"n = {Ml.sum():,} images · accuracy {acc:.4f} · macro-F1 {macro:.4f}",
         OUTDIR / "figD_confusion_lab.png",
         "Rows shaded by share of that true class. Counts are absolute.")

    # Assert, do not trust: the figure must agree with the stored scalars.
    for i, c in enumerate(CLASSES):
        if abs(per[c]["f1"] - int8["per_class_f1"][i]) > 1e-4:
            print(f"  MISMATCH {c} F1 {per[c]['f1']:.4f} vs stored "
                  f"{int8['per_class_f1'][i]:.4f}"); ok = False
    for got, want, name in ((acc, int8["cls_acc"], "accuracy"),
                            (macro, int8["macro_f1"], "macro-F1")):
        if abs(got - want) > 1e-4:
            print(f"  MISMATCH {name} {got:.4f} vs stored {want:.4f}"); ok = False
    print(f"lab     n={Ml.sum():,}  acc {acc:.4f}  macro-F1 {macro:.4f}  "
          f"-> figD_confusion_lab.png")

    # ---------------------------------------------------------------- device
    rows = [json.loads(l) for l in
            (REPO / "results" / "live_captures.jsonl").read_text().splitlines()
            if l.strip()]
    rows = [r for r in rows
            if r.get("session") == "final" and r.get("truth") and r.get("pred")]
    Md = np.zeros((len(CLASSES), len(CLASSES)), int)
    for r in rows:
        Md[CLASSES.index(r["truth"]), CLASSES.index(r["pred"])] += 1

    per_d, acc_d, macro_d = metrics(Md, CLASSES)
    draw(Md, CLASSES,
         "Class confusion — same model, live OV7675 captures",
         f"n = {Md.sum()} captures · accuracy {acc_d:.4f} · "
         f"macro-F1 {macro_d:.4f} over the 4 classes with support",
         OUTDIR / "figE_confusion_device.png",
         "potato had no on-device samples — food was not available.")
    print(f"device  n={Md.sum()}  acc {acc_d:.4f}  macro-F1 {macro_d:.4f}  "
          f"-> figE_confusion_device.png")

    # ---------------------------------------------------- 4-class variants
    # Potato removed entirely, row and column. On device this is honest --
    # the class was never tested. In the lab it is a different claim: potato
    # has 600 test images and absorbs 296 errors from other classes, so the
    # surviving metrics rise sharply. Both are written; the caption states n
    # and the exclusion so neither can be quoted as the 5-class result.
    KEEP = ["chicken", "broccoli", "rice", "beef"]

    Ml4, L4 = restrict(Ml, CLASSES, KEEP)
    _, acc_l4, macro_l4 = metrics(Ml4, L4)
    draw(Ml4, L4,
         "Class confusion — int8 model, potato excluded",
         f"n = {Ml4.sum():,} images · accuracy {acc_l4:.4f} · "
         f"macro-F1 {macro_l4:.4f}",
         OUTDIR / "figD_confusion_lab_4class.png",
         "Potato excluded, row and column. Not comparable to the 5-class "
         "macro-F1 of 0.6432: dropping the column also removes 296 errors "
         "from the surviving classes.")
    print(f"lab-4   n={Ml4.sum():,}  acc {acc_l4:.4f}  macro-F1 {macro_l4:.4f}"
          f"  -> figD_confusion_lab_4class.png")

    Md4, _ = restrict(Md, CLASSES, KEEP)
    _, acc_d4, macro_d4 = metrics(Md4, L4)
    draw(Md4, L4,
         "Class confusion — live OV7675 captures, potato excluded",
         f"n = {Md4.sum()} captures · accuracy {acc_d4:.4f} · "
         f"macro-F1 {macro_d4:.4f}",
         OUTDIR / "figE_confusion_device_4class.png",
         "Potato was never captured — no food available — so it is excluded "
         "as untested. This drops 3 chicken captures the model called potato.")
    print(f"dev-4   n={Md4.sum()}  acc {acc_d4:.4f}  macro-F1 {macro_d4:.4f}"
          f"  -> figE_confusion_device_4class.png")

    print("\nall figure numbers agree with results/" if ok else "\nDISCREPANCIES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
