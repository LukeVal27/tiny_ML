#!/usr/bin/env python3
"""
Hardware-in-the-loop harness: compile -> upload -> read serial -> parse -> record.

Implements the Section 8a loop and its stop-rules. The rules matter more than the
happy path: an agent that retries an upload forever because the board is
unplugged is worse than one that stops and says so. So every failure mode here is
bounded -- one retry, then report and exit non-zero.

Usage:
    python3 harness/flash_and_measure.py --sketch deploy/nano/smoke_test \
        --define ARENA_SIZE=90000 --define CAM_FORMAT=1
    python3 harness/flash_and_measure.py --list-ports
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

FQBN = "arduino:mbed_nano:nano33ble"
BAUD = 115200
SRAM_TOTAL = 262144
FLASH_TOTAL = 983040

FLASH_RE = re.compile(r"Sketch uses (\d+) bytes")
RAM_RE = re.compile(r"Global variables use (\d+) bytes.*?leaving (\d+) bytes", re.S)
# Telemetry lines look like  TAG,key=value,key=value
TELEM_RE = re.compile(r"^(SMOKE|BOOT|BENCH|INFER|TIMING|CHECK|RAW),(.*)$")


def find_port():
    """Return (port, None) or (None, reason)."""
    out = subprocess.run(["arduino-cli", "board", "list", "--format", "json"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return None, f"arduino-cli board list failed: {out.stderr.strip()}"
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None, "could not parse arduino-cli board list output"

    ports = data.get("detected_ports", data) if isinstance(data, dict) else data
    for p in ports:
        info = p.get("port", p)
        addr = info.get("address", "")
        for b in (p.get("matching_boards") or []):
            if b.get("fqbn") == FQBN:
                return addr, None
        # Fall back on the USB address pattern if the board is in bootloader
        # mode, where it enumerates without advertising an FQBN.
        if "usbmodem" in addr:
            return addr, None
    return None, "no Nano 33 BLE found on any serial port"


def parse_telemetry(line):
    m = TELEM_RE.match(line.strip())
    if not m:
        return None
    tag, rest = m.group(1), m.group(2)
    rec = {"_tag": tag}
    for kv in rest.split(","):
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


def build_dir(sketch):
    """
    Dedicated build directory per sketch.

    We compile with an explicit --build-path and upload with --input-dir
    pointing at the same place. Without this, `upload` re-derives its own build
    directory and can flash a stale binary compiled with different -D flags,
    which during an arena sweep means the reported numbers quietly belong to the
    wrong configuration.
    """
    d = REPO / "results" / "build" / pathlib.Path(sketch).name
    d.mkdir(parents=True, exist_ok=True)
    return d


def compile_sketch(sketch, defines, verbose=False):
    cmd = ["arduino-cli", "compile", "--fqbn", FQBN,
           "--build-path", str(build_dir(sketch))]
    if defines:
        flags = " ".join(f"-D{d}" for d in defines)
        cmd += ["--build-property", f"build.extra_flags={flags}"]
    cmd.append(str(sketch))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        if verbose:
            print(out)
        return None, out

    fm, rm = FLASH_RE.search(out), RAM_RE.search(out)
    stats = {}
    if fm and rm:
        stats = {
            "flash_bytes": int(fm.group(1)),
            "flash_pct": round(100 * int(fm.group(1)) / FLASH_TOTAL, 1),
            "static_ram_bytes": int(rm.group(1)),
            "free_for_locals": int(rm.group(2)),
        }
    return stats, out


def release_port():
    """
    Kill any lingering arduino serial-monitor holding the port.

    If the Arduino IDE is open it keeps a `serial-monitor` helper attached to
    the board, and it respawns within seconds of being killed. That helper holds
    the port exclusively, which blocks the 1200-baud touch that puts the
    nRF52840 into its bootloader -- so every upload fails with the misleading
    "No device found on cu.usbmodemNNN". Killing it immediately before the touch
    wins the race reliably. Only the monitor helper is targeted; the IDE itself
    is left alone.
    """
    subprocess.run(["pkill", "-f", "serial-monitor"], capture_output=True)
    time.sleep(0.3)


def bootloader_touch(port):
    """Open at 1200 baud and drop DTR: the standard Arduino reset-to-bootloader."""
    try:
        import serial

        s = serial.Serial(port, 1200)
        s.setDTR(False)
        time.sleep(0.05)
        s.close()
        time.sleep(1.5)
        return True
    except Exception:
        # arduino-cli will still attempt its own touch; not fatal on its own.
        return False


def upload(sketch, port, defines):
    # `upload` takes no --build-property; it flashes an already-built artifact.
    # We point it at the exact directory compile_sketch() just wrote.
    release_port()
    bootloader_touch(port)
    cmd = ["arduino-cli", "upload", "-p", port, "--fqbn", FQBN,
           "--input-dir", str(build_dir(sketch)), str(sketch)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0, proc.stdout + proc.stderr


def read_serial(port, timeout=25, want_tags=("SMOKE", "BENCH", "INFER"),
                settle=2.0):
    # NOTE: BOOT is deliberately NOT a terminal tag. Seeing a terminal tag
    # shortens the read window to ~3 s, and BOOT prints immediately while BENCH
    # takes ~23 s (21 invokes at ~1.08 s each). Treating BOOT as terminal closed
    # the window long before the measurement arrived, so a run looked like it had
    # produced only boot telemetry and the latency figure was silently lost.
    """
    Open the port and collect telemetry lines until timeout.

    The nRF52840 re-enumerates its USB CDC after a reset, so the port can take a
    moment to reappear; we retry the open for a few seconds before giving up.
    """
    import serial

    # Same contention as during upload: the IDE's monitor grabs the port the
    # moment the board re-enumerates, and we would read nothing.
    release_port()

    ser = None
    deadline = time.time() + 10
    last_err = None
    while time.time() < deadline:
        try:
            ser = serial.Serial(port, BAUD, timeout=1)
            break
        except Exception as e:  # port not back yet
            last_err = e
            time.sleep(0.4)
    if ser is None:
        return [], [], f"could not open {port}: {last_err}"

    time.sleep(settle)
    lines, records = [], []
    end = time.time() + timeout
    try:
        while time.time() < end:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            lines.append(line)
            rec = parse_telemetry(line)
            if rec:
                records.append(rec)
                if rec["_tag"] in want_tags:
                    # Give any trailing lines (BENCH after BOOT) a moment.
                    end = min(end, time.time() + 3)
    finally:
        ser.close()
    return records, lines, None


def run_once(sketch, defines, port=None, timeout=25, verbose=False):
    sketch = pathlib.Path(sketch)
    label = ",".join(defines) if defines else "(defaults)"
    print(f"\n=== {sketch.name}  [{label}] ===")

    if port is None:
        port, err = find_port()
        if err:
            print(f"  ERROR: {err}")
            print("  ACTION NEEDED: plug in the Nano 33 BLE and re-run.")
            return None

    print(f"  port: {port}")
    print("  compiling ...", end=" ", flush=True)
    stats, out = compile_sketch(sketch, defines, verbose)
    if stats is None:
        print("FAILED")
        tail = [l for l in out.splitlines() if l.strip()][-6:]
        for l in tail:
            print(f"    {l}")
        return {"status": "COMPILE_FAIL", "defines": defines,
                "error": tail[-1] if tail else "unknown"}
    print(f"flash {stats['flash_bytes']:,} "
          f"({stats['flash_pct']}%)  static RAM {stats['static_ram_bytes']:,} "
          f"free {stats['free_for_locals']:,}")

    # Stop-rule: upload gets exactly one retry, then we ask for a human.
    print("  uploading ...", end=" ", flush=True)
    ok, uout = upload(sketch, port, defines)
    if not ok:
        print("failed, retrying once ...", end=" ", flush=True)
        time.sleep(2)
        ok, uout = upload(sketch, port, defines)
    if not ok:
        print("FAILED")
        for l in [l for l in uout.splitlines() if l.strip()][-6:]:
            print(f"    {l}")
        print("  ACTION NEEDED: double-tap the RESET button to enter the "
              "bootloader, then re-run.")
        return {"status": "UPLOAD_FAIL", "defines": defines, **stats}
    print("ok")

    print(f"  reading serial (timeout {timeout}s) ...", end=" ", flush=True)
    records, lines, err = read_serial(port, timeout=timeout)
    if err or not records:
        print("NO TELEMETRY")
        if lines:
            print("    raw serial saw:")
            for l in lines[-6:]:
                print(f"      {l}")
        print("  ACTION NEEDED: tap RESET once and re-run; if still silent, "
              "check the board is not held in the bootloader.")
        return {"status": "NO_TELEMETRY", "defines": defines, **stats}
    print(f"got {len(records)} telemetry line(s)")

    for r in records:
        kv = " ".join(f"{k}={v}" for k, v in r.items() if k != "_tag")
        print(f"    [{r['_tag']}] {kv}")

    merged = {"status": "OK", "defines": defines, **stats}
    for r in records:
        for k, v in r.items():
            if k != "_tag":
                merged[f"{r['_tag'].lower()}_{k}"] = v
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketch", default="deploy/nano/smoke_test")
    ap.add_argument("--define", action="append", default=[],
                    help="preprocessor define, e.g. ARENA_SIZE=90000")
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--port", default=None)
    ap.add_argument("--out", default="results/device_runs.jsonl")
    ap.add_argument("--list-ports", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.list_ports:
        port, err = find_port()
        print(f"port: {port}" if port else f"ERROR: {err}")
        return 0 if port else 1

    rec = run_once(REPO / args.sketch, args.define, args.port, args.timeout,
                   args.verbose)
    if rec is None:
        return 2

    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rec["sketch"] = args.sketch
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n  -> appended to {args.out}")
    return 0 if rec.get("status") == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
