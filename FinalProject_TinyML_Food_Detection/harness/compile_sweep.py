#!/usr/bin/env python3
"""
Section 5a GATE, static half: arena-size sweep with no board attached.

The tensor arena and camera frame buffer are both statically allocated arrays,
so arduino-cli's own compile-time RAM report is a truthful measurement of
whether they coexist. This lets us answer the Tier1-vs-Tier2 gate question
without hardware; harness/flash_and_measure.py confirms it on-device later.

Usage:
    python3 harness/compile_sweep.py                  # default sweep
    python3 harness/compile_sweep.py --quick          # 3 points, smoke only
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
SKETCH = REPO / "deploy" / "nano" / "smoke_test"
RESULTS = REPO / "results" / "compile_sweep.jsonl"

FQBN = "arduino:mbed_nano:nano33ble"
SRAM_TOTAL = 262144
FLASH_TOTAL = 983040

# nRF52840 needs real stack headroom at runtime: mbed OS main thread, the
# camera driver's tight PCLK sampling loop, and TFLM's own call depth. We
# refuse to call a build "viable" without this much slack left over.
STACK_HEADROOM_MIN = 24 * 1024

FLASH_RE = re.compile(r"Sketch uses (\d+) bytes")
RAM_RE = re.compile(r"Global variables use (\d+) bytes.*?leaving (\d+) bytes", re.S)


def compile_once(arena, cam_format, cam_res, verbose=False):
    """Compile one configuration and return its static flash/RAM footprint."""
    flags = f"-DARENA_SIZE={arena} -DCAM_FORMAT={cam_format} -DCAM_RES={cam_res}"
    cmd = [
        "arduino-cli", "compile",
        "--fqbn", FQBN,
        "--build-property", f"build.extra_flags={flags}",
        str(SKETCH),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    out = proc.stdout + proc.stderr

    rec = {
        "arena_req": arena,
        "mode": "gray" if cam_format == 0 else "rgb565",
        "res": "qcif" if cam_res == 0 else "qqvga",
        "frame_bytes": (176 * 144 if cam_res == 0 else 160 * 120)
        * (1 if cam_format == 0 else 2),
        "compile_s": round(elapsed, 1),
    }

    if proc.returncode != 0:
        rec["status"] = "COMPILE_FAIL"
        # The linker is the ground truth for "does not fit at all".
        rec["overflow"] = "region `RAM' overflowed" in out or "overflowed by" in out
        rec["error"] = out.strip().splitlines()[-1] if out.strip() else "unknown"
        if verbose:
            print(out)
        return rec

    fm = FLASH_RE.search(out)
    rm = RAM_RE.search(out)
    if not (fm and rm):
        rec["status"] = "PARSE_FAIL"
        rec["error"] = out[-500:]
        return rec

    flash = int(fm.group(1))
    globals_ram = int(rm.group(1))
    free_ram = int(rm.group(2))

    rec.update(
        {
            "flash_bytes": flash,
            "flash_pct": round(100 * flash / FLASH_TOTAL, 1),
            "globals_bytes": globals_ram,
            "free_for_locals": free_ram,
            "free_pct": round(100 * free_ram / SRAM_TOTAL, 1),
            # Everything that is not arena and not frame: mbed OS, TFLM
            # runtime, Serial, driver state.
            "overhead_bytes": globals_ram - arena - rec["frame_bytes"],
            "status": "OK" if free_ram >= STACK_HEADROOM_MIN else "TIGHT",
        }
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        configs = [(a, 0, 0) for a in (20_000, 100_000, 160_000)]
    else:
        configs = (
            # Grayscale QCIF - the primary candidate path.
            [(a, 0, 0) for a in (20_000, 60_000, 100_000, 140_000, 170_000, 190_000)]
            # RGB565 QCIF - does color fit at all?
            + [(a, 1, 0) for a in (60_000, 100_000, 140_000)]
            # Grayscale QQVGA - smaller frame fallback.
            + [(a, 0, 1) for a in (140_000, 170_000)]
        )

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    records = []
    print(f"Running {len(configs)} compile configurations "
          f"(~75 s each, ~{len(configs) * 75 // 60} min total)\n")

    for i, (arena, fmt, res) in enumerate(configs, 1):
        label = f"arena={arena:>7,} fmt={'gray' if fmt == 0 else 'rgb565':<6} " \
                f"res={'qcif' if res == 0 else 'qqvga'}"
        print(f"[{i}/{len(configs)}] {label} ... ", end="", flush=True)
        rec = compile_once(arena, fmt, res)
        records.append(rec)
        with RESULTS.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        if rec["status"] in ("OK", "TIGHT"):
            print(f"{rec['status']:5s} globals={rec['globals_bytes']:,} "
                  f"free={rec['free_for_locals']:,} flash={rec['flash_bytes']:,}")
        else:
            print(f"{rec['status']} ({rec.get('error', '')[:70]})")

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 78)
    print("SECTION 5a GATE - STATIC COMPILE SWEEP")
    print("=" * 78)
    print(f"{'mode':<8}{'res':<7}{'arena':>9}{'frame':>8}{'globals':>9}"
          f"{'free':>9}{'flash':>9}  status")
    print("-" * 78)
    for r in records:
        if r["status"] in ("OK", "TIGHT"):
            print(f"{r['mode']:<8}{r['res']:<7}{r['arena_req']:>9,}"
                  f"{r['frame_bytes']:>8,}{r['globals_bytes']:>9,}"
                  f"{r['free_for_locals']:>9,}{r['flash_bytes']:>9,}  {r['status']}")
        else:
            print(f"{r['mode']:<8}{r['res']:<7}{r['arena_req']:>9,}"
                  f"{r['frame_bytes']:>8,}{'-':>9}{'-':>9}{'-':>9}  {r['status']}")

    ok = [r for r in records if r["status"] == "OK"]
    if ok:
        overhead = round(sum(r["overhead_bytes"] for r in ok) / len(ok))
        print("-" * 78)
        print(f"Fixed overhead (mbed OS + TFLM + Serial): ~{overhead:,} bytes")
        for mode in ("gray", "rgb565"):
            sub = [r for r in ok if r["mode"] == mode and r["res"] == "qcif"]
            if sub:
                best = max(sub, key=lambda r: r["arena_req"])
                frame = best["frame_bytes"]
                ceiling = SRAM_TOTAL - overhead - frame - STACK_HEADROOM_MIN
                print(f"  {mode:<7} qcif: max viable arena ~= {ceiling:,} bytes "
                      f"(verified to {best['arena_req']:,})")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
