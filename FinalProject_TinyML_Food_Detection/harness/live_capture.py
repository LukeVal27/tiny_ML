#!/usr/bin/env python3
"""
Live-camera capture session: trigger, read, record, tally.

Point the camera at a plate, type the TRUE class, press Enter. The board
captures, classifies, and this logs prediction vs. truth so the session produces
a real confusion matrix instead of scrollback.

The comparison this is really for: the lab numbers (macro-F1 0.6432 on composited
plates) versus the same model on actual camera frames. The model has never seen a
camera frame, so the gap between those two is the measurement, not a failure.

Give the portion too -- "rice,small" -- and the portion head gets scored as well.
Without a portion truth label there is nothing to score it against, and the
device has so far returned `large` on every single capture.

Usage:
    python3 harness/live_capture.py --session final     # interactive
    python3 harness/live_capture.py --summary           # tally the whole log
    python3 harness/live_capture.py --summary --session final   # one session
"""

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
LOG = REPO / "results" / "live_captures.jsonl"
PORT_DEFAULT = "/dev/cu.usbmodem101"
CLASSES = ["chicken", "broccoli", "rice", "beef", "potato"]
PORTIONS = ["small", "medium", "large"]


def parse_kv(line):
    rec = {}
    for kv in line.split(",")[1:]:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            rec[k] = int(v)
        except ValueError:
            try:
                rec[k] = float(v)
            except ValueError:
                rec[k] = v
    return rec


def confusion(rows, labels, truth_key, pred_key, title):
    """
    Print one confusion matrix and return (correct, n).

    The `n` column is not decoration: the rubric wants the per-class sample
    count stated explicitly, so the table has to carry it rather than leaving it
    to be summed by eye off the row.
    """
    rows = [r for r in rows if r.get(truth_key) and r.get(pred_key)]
    if not rows:
        return 0, 0

    w = max(max(len(x) for x in labels), len("truth \\ pred")) + 2
    print(f"\n{title}")
    print("truth \\ pred".ljust(w) + "".join(f"{x[:8]:>9}" for x in labels) +
          f"{'n':>6}   recall")
    for t in labels:
        row = [sum(1 for r in rows if r[truth_key] == t and r[pred_key] == p)
               for p in labels]
        tot = sum(row)
        rec = row[labels.index(t)] / tot if tot else 0.0
        print(t.ljust(w) + "".join(f"{v:>9}" for v in row) + f"{tot:>6}" +
              (f"   {rec:.3f}" if tot else "      --"))

    return sum(1 for r in rows if r[truth_key] == r[pred_key]), len(rows)


def summary(session=None):
    if not LOG.exists():
        print("no captures logged yet")
        return 0
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    if session:
        rows = [r for r in rows if r.get("session") == session]
    rows = [r for r in rows if r.get("truth") and r.get("pred")]
    if not rows:
        print("no labelled captures yet" +
              (f" in session '{session}'" if session else ""))
        return 0

    tag = f"  [session '{session}']" if session else ""
    correct, n = confusion(rows, CLASSES, "truth", "pred",
                           f"CLASS{tag}")
    print(f"\nclass accuracy: {correct}/{n} = {correct / n:.3f}")

    # Portion is only scoreable where a portion truth was typed in. Say so when
    # it is missing rather than silently omitting the table -- every capture
    # before this flag existed is unscoreable, and that is the point.
    pcorrect, pn = confusion(rows, PORTIONS, "truth_portion", "portion",
                             f"PORTION{tag}")
    if pn:
        print(f"\nportion accuracy: {pcorrect}/{pn} = {pcorrect / pn:.3f}")
    else:
        print("\nportion: no ground truth recorded — capture as 'rice,small' "
              "to score the portion head")

    lat = [r["total_us"] for r in rows if r.get("total_us")]
    if lat:
        print(f"\ncapture-to-result latency: mean {sum(lat)/len(lat)/1000:.0f} ms "
              f"(n={len(lat)})")
    print(f"\nlab reference on composited plates: macro-F1 0.6432, acc 0.6433")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=PORT_DEFAULT)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--session", default="",
                    help="tag every row so one sitting can be tallied alone")
    args = ap.parse_args()

    if args.summary:
        return summary(args.session or None)

    import serial
    import subprocess

    # The IDE's serial-monitor helper grabs the port exclusively; without this
    # every read comes back empty and the board looks dead.
    subprocess.run(["pkill", "-f", "serial-monitor"], capture_output=True)
    time.sleep(0.3)

    ser = serial.Serial(args.port, 115200, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    LOG.parent.mkdir(exist_ok=True)

    print("=" * 62)
    print("LIVE CAPTURE  —  plate overhead, whole rim in frame, one food")
    print("  type  class[,portion]  then Enter to capture")
    print(f"  classes:  {', '.join(CLASSES)}")
    print(f"  portions: {', '.join(PORTIONS)}  (optional, e.g. 'rice,small')")
    if args.session:
        print(f"  session:  {args.session}")
    print("  's' = summary,  'q' = quit")
    print("=" * 62)

    try:
        while True:
            entry = input("\ntrue class[,portion] > ").strip().lower()
            if entry in ("q", "quit", "exit"):
                break
            if entry in ("s", "summary"):
                summary(args.session or None)
                continue

            truth, _, tp = entry.partition(",")
            truth, truth_portion = truth.strip(), tp.strip()
            if truth and truth not in CLASSES:
                # Allow unlabelled captures, but say so rather than guessing.
                print(f"  (not one of {CLASSES} — recording as unlabelled)")
                truth = ""
            if truth_portion and truth_portion not in PORTIONS:
                print(f"  (not one of {PORTIONS} — portion left unlabelled)")
                truth_portion = ""

            ser.reset_input_buffer()
            ser.write(b"c")
            infer = timing = None
            end = time.time() + 15
            while time.time() < end and (infer is None or timing is None):
                line = ser.readline().decode("utf-8", errors="replace").strip()
                if line.startswith("INFER,"):
                    infer = parse_kv(line)
                elif line.startswith("TIMING,"):
                    timing = parse_kv(line)

            if infer is None:
                print("  no response — is BENCH_MODE=0 flashed? is the shield seated?")
                continue

            rec = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "session": args.session,
                "truth": truth,
                "truth_portion": truth_portion,
                "pred": infer.get("cls"),
                "cls_conf": infer.get("cls_conf"),
                "portion": infer.get("portion"),
                "portion_conf": infer.get("portion_conf"),
                "mass_range": infer.get("mass_range"),
            }
            if timing:
                rec.update({k: timing.get(k) for k in
                            ("capture_us", "preprocess_us", "infer_us", "total_us")})
            with LOG.open("a") as f:
                f.write(json.dumps(rec) + "\n")

            mark = ""
            if truth:
                mark = "  ✓" if rec["pred"] == truth else f"  ✗ (said {rec['pred']})"
            pmark = ""
            if truth_portion:
                pmark = "✓" if rec["portion"] == truth_portion else \
                        f"✗ (truth {truth_portion})"
            print(f"  -> {rec['pred']} ({rec['cls_conf']}) | "
                  f"portion {rec['portion']} {rec['mass_range']} {pmark} | "
                  f"{(rec.get('total_us') or 0)/1000:.0f} ms{mark}")
    finally:
        ser.close()
        print()
        summary(args.session or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
