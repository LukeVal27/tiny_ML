#!/usr/bin/env python3
"""
Live view + classification, driving the classifier_tier1 sketch (BENCH_MODE=0).

Shows the 96x96 image the model actually sees -- the same input tensor it
classifies, not a parallel recomputation -- with a guide ring for plate framing
and the latest prediction overlaid.

Controls (in the window):
    c   capture and classify the current frame
    v   toggle the live stream on/off
    q   quit

The board streams frames as int8 with zero_point -128, so uint8 = value + 128.
No extra RAM is used on the device: the preview IS the model's input buffer.

Usage:
    python3 harness/camera_view.py
    python3 harness/camera_view.py --save 5     # also save PNGs
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
PORT_DEFAULT = "/dev/cu.usbmodem101"
W = H = 96
FRAME_BYTES = W * H * 3
MAGIC = b"FRM:"
# Half-resolution framing preview: 4x less serial data, same field of view.
HALF_W = HALF_H = 48
HALF_BYTES = HALF_W * HALF_H * 3
MAGIC_HALF = b"FRH:"
LOG = REPO / "results" / "live_captures.jsonl"


def parse_kv(line):
    rec = {}
    for kv in line.split(",")[1:]:
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                rec[k] = float(v) if "." in v else int(v)
            except ValueError:
                rec[k] = v
    return rec


class Reader:
    """Pulls frames and INFER lines out of one mixed binary/text stream."""

    def __init__(self, ser):
        self.ser = ser
        self.buf = bytearray()

    def poll(self):
        """Return (frame_or_None, list_of_text_lines)."""
        n = self.ser.in_waiting
        if n:
            self.buf += self.ser.read(n)

        frame, lines = None, []

        # Frames first: find the magic, take the fixed-size payload after it.
        while True:
            i_full = self.buf.find(MAGIC)
            i_half = self.buf.find(MAGIC_HALF)
            cands = [(i, n, w) for i, n, w in
                     ((i_full, FRAME_BYTES, W), (i_half, HALF_BYTES, HALF_W))
                     if i >= 0]
            if not cands:
                break
            i, nbytes, width = min(cands)
            if len(self.buf) < i + 4 + nbytes:
                break
            payload = bytes(self.buf[i + 4: i + 4 + nbytes])
            # Text before the marker is telemetry; keep it.
            head = bytes(self.buf[:i])
            for ln in head.split(b"\n"):
                t = ln.decode("utf-8", "replace").strip()
                if t:
                    lines.append(t)
            del self.buf[: i + 4 + nbytes]
            if width == W:
                # Full-res frames carry the model's int8 input tensor.
                arr = np.frombuffer(payload, dtype=np.int8).astype(np.int16) + 128
                frame = arr.astype(np.uint8).reshape(H, W, 3)
            else:
                # Half-res framing frames are already uint8.
                frame = np.frombuffer(payload, dtype=np.uint8).reshape(
                    HALF_H, HALF_W, 3)

        # Any complete text lines left over (no frame pending).
        if MAGIC not in self.buf and MAGIC_HALF not in self.buf:
            while b"\n" in self.buf:
                ln, _, rest = bytes(self.buf).partition(b"\n")
                del self.buf[: len(ln) + 1]
                t = ln.decode("utf-8", "replace").strip()
                if t:
                    lines.append(t)
        # Never let a stalled stream grow without bound.
        if len(self.buf) > 4 * FRAME_BYTES:
            del self.buf[: len(self.buf) - 2 * FRAME_BYTES]
        return frame, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=PORT_DEFAULT)
    ap.add_argument("--save", type=int, default=0)
    args = ap.parse_args()

    import serial
    import matplotlib.pyplot as plt

    subprocess.run(["pkill", "-f", "serial-monitor"], capture_output=True)
    time.sleep(0.3)
    ser = serial.Serial(args.port, 115200, timeout=0)
    time.sleep(2)
    ser.reset_input_buffer()
    ser.write(b"v")                      # start streaming
    rd = Reader(ser)

    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(np.zeros((H, W, 3), np.uint8), interpolation="nearest")
    ring = plt.Circle((W / 2 - 0.5, H / 2 - 0.5), W * 0.77 / 2, fill=False,
                      color="#00E5FF", lw=1.4, alpha=0.85)
    ax.add_patch(ring)
    ax.set_title("fit the plate rim near the ring\n"
                 "c = classify   v = stream on/off   q = quit", fontsize=10)
    txt = ax.text(0.5, -0.06, "streaming…", transform=ax.transAxes,
                  ha="center", va="top", fontsize=11, family="monospace")
    fig.canvas.manager.set_window_title("food-tinyml live view")

    state = {"n": 0, "saved": 0, "t0": time.time()}
    outdir = REPO / "results" / "camera_frames"
    if args.save:
        outdir.mkdir(parents=True, exist_ok=True)

    def on_key(ev):
        if ev.key in ("c", "v"):
            ser.write(ev.key.encode())
            ser.flush()
            print(f"  [sent '{ev.key}']")   # confirms the window had focus
            if ev.key == "c":
                txt.set_text("classifying… (~1.1 s)")
        elif ev.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout(); plt.ion(); plt.show()

    print("window open — click it, then press c / v / q")
    try:
        while plt.fignum_exists(fig.number):
            frame, lines = rd.poll()
            if frame is not None:
                h, w = frame.shape[:2]
                if im.get_array().shape[:2] != (h, w):
                    im.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
                    ring.set_center((w / 2 - 0.5, h / 2 - 0.5))
                    ring.set_radius(w * 0.77 / 2)
                im.set_data(frame)
                state["n"] += 1
                if state["saved"] < args.save:
                    from PIL import Image
                    Image.fromarray(frame).save(
                        outdir / f"frame_{state['saved']:03d}.png")
                    state["saved"] += 1
                fps = state["n"] / max(time.time() - state["t0"], 1e-6)
                ax.set_xlabel(f"{state['n']} frames · {fps:.1f} fps", fontsize=9)
            for ln in lines:
                if ln.startswith("INFER,"):
                    r = parse_kv(ln)
                    msg = (f"{r.get('cls')}  {r.get('cls_conf')}\n"
                           f"portion {r.get('portion')}  {r.get('mass_range')}")
                    txt.set_text(msg)
                    print(f"  {msg.splitlines()[0]} | {msg.splitlines()[1]}")
                    with LOG.open("a") as f:
                        f.write(json.dumps({
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "truth": "", "pred": r.get("cls"),
                            "cls_conf": r.get("cls_conf"),
                            "portion": r.get("portion"),
                            "mass_range": r.get("mass_range"),
                            "source": "camera_view"}) + "\n")
                elif ln.startswith(("BOOT", "TIMING", "#")):
                    print(f"  {ln[:120]}")
            fig.canvas.draw_idle()
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            # Must flush AND give the board time to act on it. Closing the port
            # straight after write() drops the byte, leaving the sketch
            # streaming and the blue LED blinking after the viewer has exited.
            ser.reset_input_buffer()
            ser.write(b"v")
            ser.flush()
            time.sleep(0.4)
            ser.close()
        except Exception:
            pass
        print(f"\nstopped after {state['n']} frames — stream toggled off")
    return 0


if __name__ == "__main__":
    sys.exit(main())
