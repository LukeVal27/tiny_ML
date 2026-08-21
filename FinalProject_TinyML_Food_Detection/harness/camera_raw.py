#!/usr/bin/env python3
"""
Viewer for the camera_raw sketch — the camera with nothing else in the way.

Live half-resolution stream for aiming, plus a one-shot full raw dump that
decodes the sensor's own RGB565 bytes four different ways side by side. That
comparison is the point: if all four look equally broken the fault is upstream
of the decode (optics, exposure, or the bit-banged read); if one looks right,
we have our answer.

    python3 harness/camera_raw.py            live view, press r for the raw test
    python3 harness/camera_raw.py --raw      go straight to the raw test
"""
import argparse, subprocess, sys, time
import numpy as np

PORT = "/dev/cu.usbmodem101"
CAM_W, CAM_H = 176, 144
OUT_W, OUT_H = CAM_W // 2, CAM_H // 2
HALF_BYTES = OUT_W * OUT_H * 3
RAW_BYTES = CAM_W * CAM_H * 2


def read_block(ser, magic, nbytes, timeout=20):
    buf = bytearray(); end = time.time() + timeout
    while time.time() < end:
        d = ser.read(4096)
        if d: buf += d
        i = buf.find(magic)
        if i >= 0 and len(buf) >= i + len(magic) + nbytes:
            return bytes(buf[i + len(magic): i + len(magic) + nbytes])
    return None


def decode(raw, mode):
    """Four candidate interpretations of the same RGB565 bytes."""
    a = np.frombuffer(raw, dtype=np.uint8).reshape(CAM_H, CAM_W, 2).astype(np.uint16)
    if mode.startswith("big"):
        v = (a[..., 0] << 8) | a[..., 1]
    else:
        v = (a[..., 1] << 8) | a[..., 0]
    r = ((v >> 11) & 0x1F) * 255 // 31
    g = ((v >> 5) & 0x3F) * 255 // 63
    b = (v & 0x1F) * 255 // 31
    if mode.endswith("swap"):
        r, b = b, r
    return np.dstack([r, g, b]).astype(np.uint8)


def raw_test(ser):
    from PIL import Image
    ser.reset_input_buffer(); ser.write(b"r"); ser.flush()
    raw = read_block(ser, b"RAW1:", RAW_BYTES, timeout=30)
    if raw is None:
        print("no raw frame received"); return
    modes = ["big", "big+swap", "little", "little+swap"]
    imgs = [decode(raw, m) for m in modes]
    Z = 2
    sheet = Image.new("RGB", (len(imgs) * CAM_W * Z, CAM_H * Z))
    for i, im in enumerate(imgs):
        sheet.paste(Image.fromarray(im).resize((CAM_W * Z, CAM_H * Z), Image.NEAREST),
                    (i * CAM_W * Z, 0))
    sheet.save("results/raw_decode_compare.png")
    print("saved results/raw_decode_compare.png   order: " + " | ".join(modes))
    for m, im in zip(modes, imgs):
        mu = im.reshape(-1, 3).mean(0)
        print(f"  {m:12s} R={mu[0]:6.1f} G={mu[1]:6.1f} B={mu[2]:6.1f}  G-R={mu[1]-mu[0]:+6.1f}")
    open("results/raw_frame.bin", "wb").write(raw)
    print("raw bytes saved to results/raw_frame.bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()
    import serial, matplotlib, matplotlib.pyplot as plt
    # Matplotlib binds s=save, c/v=back/forward, f=fullscreen, g=grid, etc.
    # Those collide with the sensor-control keys, so clear the whole default
    # keymap and let this window own every keystroke.
    for _k in [k for k in matplotlib.rcParams if k.startswith("keymap.")]:
        matplotlib.rcParams[_k] = []

    subprocess.run(["pkill", "-f", "serial-monitor"], capture_output=True)
    time.sleep(0.3)
    ser = serial.Serial(args.port, 115200, timeout=0.3)
    time.sleep(2); ser.reset_input_buffer()

    if args.raw:
        raw_test(ser); ser.close(); return 0

    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.set_xticks([]); ax.set_yticks([])
    im = ax.imshow(np.zeros((OUT_H, OUT_W, 3), np.uint8), interpolation="nearest")
    ax.set_title("raw camera — no model, no preprocessing\n"
                 "S/s saturation   C/c contrast   B/b brightness   0 reset   "
                 "r raw test   q quit", fontsize=9)
    fig.canvas.manager.set_window_title("camera_raw")

    def on_key(ev):
        if ev.key == "r":
            raw_test(ser)
        elif ev.key == "q":
            plt.close(fig)
        elif ev.key in "sScCbB0":
            # Image-control keys are forwarded straight to the sensor.
            ser.write(ev.key.encode()); ser.flush()
            print(f"  [sent '{ev.key}']")
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.ion(); plt.tight_layout(); plt.show()

    n, t0 = 0, time.time()
    buf = bytearray()
    try:
        while plt.fignum_exists(fig.number):
            d = ser.read(8192)
            if d: buf += d
            while True:
                i = buf.find(b"HALF:")
                if i < 0 or len(buf) < i + 5 + HALF_BYTES: break
                pay = bytes(buf[i + 5: i + 5 + HALF_BYTES]); del buf[: i + 5 + HALF_BYTES]
                fr = np.frombuffer(pay, np.uint8).reshape(OUT_H, OUT_W, 3)
                im.set_data(fr)
                n += 1
                # Saturation spread is the number that matters here: a measured
                # 3-5 means the frame is effectively greyscale. Training images
                # sit far higher.
                mu = fr.reshape(-1, 3).mean(0)
                sat = mu.max() - mu.min()
                ax.set_xlabel(f"{n} frames · {n/max(time.time()-t0,1e-6):.1f} fps   "
                              f"| saturation spread {sat:4.1f}   "
                              f"R{mu[0]:5.1f} G{mu[1]:5.1f} B{mu[2]:5.1f}", fontsize=9)
            if b"CTRL," in buf and b"HALF:" not in buf:
                j = buf.find(b"CTRL,"); k = buf.find(b"\n", j)
                if k > 0:
                    print("  " + bytes(buf[j:k]).decode("utf-8", "replace").strip())
                    del buf[: k + 1]
            if len(buf) > 4 * HALF_BYTES: del buf[: len(buf) - 2 * HALF_BYTES]
            fig.canvas.draw_idle(); plt.pause(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close(); print(f"\nstopped after {n} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
