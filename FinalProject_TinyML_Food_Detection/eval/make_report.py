#!/usr/bin/env python3
"""
Assemble the Section 7 metric table from everything measured so far.

Pulls together three independent sources -- host-side accuracy (quantization_*.json),
compile-time size and on-device telemetry (device_runs.jsonl), plus the Section 5a
gate sweep (compile_sweep.jsonl) -- into the baseline-vs-compressed table the
handoff asks for.

Usage:
    python3 -m eval.make_report
"""

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

CLASSES = ["chicken", "broccoli", "rice", "beef", "potato"]
PORTIONS = ["small", "medium", "large"]
SRAM_TOTAL = 262144
FLASH_TOTAL = 983040


def load_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def rule(w=92):
    return "-" * w


def confusion_block(m, labels, title):
    out = [f"\n{title}"]
    w = max(len(x) for x in labels) + 2
    out.append(" " * w + "".join(f"{l[:8]:>9}" for l in labels) + "    recall")
    for i, row in enumerate(m):
        tot = sum(row)
        rec = row[i] / tot if tot else 0.0
        out.append(f"{labels[i]:<{w}}" + "".join(f"{v:>9}" for v in row)
                   + f"    {rec:.3f}")
    return "\n".join(out)


def main():
    out = []
    out.append("=" * 92)
    out.append("TinyML FOOD DETECTION + PORTION ESTIMATION -- RESULTS")
    out.append("Arduino Nano 33 BLE Sense (nRF52840) + OV7675, 96x96 RGB")
    out.append("=" * 92)

    # ------------------------------------------------ Section 5a: the RAM gate
    sweep = load_jsonl(RESULTS / "compile_sweep.jsonl")
    if sweep:
        out.append("\n\n[1] SECTION 5a GATE -- RAM COEXISTENCE")
        out.append(rule())
        ok = [r for r in sweep if r.get("status") == "OK"]
        if ok:
            oh = round(sum(r["overhead_bytes"] for r in ok) / len(ok))
            out.append(f"Fixed overhead (mbed OS + TFLM + Serial): {oh:,} bytes")
        out.append(f"{'mode':<9}{'res':<8}{'arena':>10}{'frame':>9}"
                   f"{'globals':>10}{'free':>9}  status")
        for r in sweep:
            if r.get("status") in ("OK", "TIGHT"):
                out.append(f"{r['mode']:<9}{r['res']:<8}{r['arena_req']:>10,}"
                           f"{r['frame_bytes']:>9,}{r['globals_bytes']:>10,}"
                           f"{r['free_for_locals']:>9,}  {r['status']}")
            else:
                out.append(f"{r['mode']:<9}{r['res']:<8}{r['arena_req']:>10,}"
                           f"{r['frame_bytes']:>9,}{'-':>10}{'-':>9}  "
                           f"{r.get('status')}")

    # ------------------------------------------- on-device gate confirmation
    dev = load_jsonl(RESULTS / "device_runs.jsonl")
    smoke = [d for d in dev if d.get("smoke_status")]
    if smoke:
        s = smoke[-1]
        out.append("\nOn-device confirmation (real hardware):")
        out.append(f"  camera init        : {s.get('smoke_cam')}")
        out.append(f"  frame buffer       : {s.get('smoke_frame_bytes'):,} bytes "
                   f"({s.get('smoke_mode')} {s.get('smoke_res')})")
        out.append(f"  arena requested    : {s.get('smoke_arena_req'):,} bytes")
        out.append(f"  largest free block : {s.get('smoke_free_sram'):,} bytes")
        out.append(f"  status             : {s.get('smoke_status')}")

    # ------------------------------- Sections 6+7: compression vs. accuracy
    # Default to the tag that is actually deployed, not merely the last one
    # alphabetically -- a10_v3 used the REDUCE_MAX/MEAN graph that never ran
    # correctly on device, so reporting it would misdescribe the shipped system.
    tag = sys.argv[1] if len(sys.argv) > 1 else "a10_pool"
    qfiles = [RESULTS / f"quantization_{tag}.json"]
    qfiles = [p for p in qfiles if p.exists()] or sorted(
        RESULTS.glob("quantization_*.json"))
    if qfiles:
        q = json.loads(qfiles[-1].read_text())
        base = q["fp32_keras"]
        out.append(f"\n\n[2] COMPRESSION vs ACCURACY  [{q['tag']}]  "
                   f"test n={q['n_test']}")
        out.append(rule())
        out.append(f"{'variant':<16}{'size KB':>10}{'ratio':>8}"
                   f"{'macro-F1':>10}{'cls acc':>10}{'portion acc':>13}"
                   f"{'ord err':>10}{'off-by-2':>10}")
        out.append(f"{'FP32 (Keras)':<16}{'-':>10}{'-':>8}"
                   f"{base['macro_f1']:>10.4f}{base['cls_acc']:>10.4f}"
                   f"{base['portion_acc']:>13.4f}{base['ordinal_error']:>10.4f}"
                   f"{base['portion_off_by_two']:>10.4f}")
        fp32_kb = next((v["kb"] for v in q["variants"]
                        if v["variant"] == "fp32_tflite"), None)
        for v in q["variants"]:
            ratio = f"{v['kb'] / fp32_kb:.2f}x" if fp32_kb else "-"
            out.append(f"{v['variant']:<16}{v['kb']:>10.1f}{ratio:>8}"
                       f"{v['macro_f1']:>10.4f}{v['cls_acc']:>10.4f}"
                       f"{v['portion_acc']:>13.4f}{v['ordinal_error']:>10.4f}"
                       f"{v['portion_off_by_two']:>10.4f}")

        int8 = next((v for v in q["variants"] if v["variant"] == "int8"), None)
        if int8:
            out.append(f"\nint8 vs FP32 macro-F1 delta: "
                       f"{int8['macro_f1'] - base['macro_f1']:+.4f}")
            out.append("\nPer-class F1 (int8):")
            for c, f1 in zip(CLASSES, int8["per_class_f1"]):
                out.append(f"  {c:<10} {f1:.4f}")
            out.append(confusion_block(int8["cls_confusion"], CLASSES,
                                       "Class confusion (int8, rows = truth):"))
            out.append(confusion_block(int8["portion_confusion"], PORTIONS,
                                       "Portion confusion (int8, rows = truth):"))

    # ------------------------------------------- Section 7: device telemetry
    bench = [d for d in dev if d.get("bench_mean_us")]
    if bench:
        b = bench[-1]
        out.append("\n\n[3] ON-DEVICE MEASUREMENTS (nRF52840 @ 64 MHz)")
        out.append(rule())
        out.append(f"  model size (flash)   : {b.get('bench_model_bytes', 0):,} bytes")
        out.append(f"  tensor arena used    : {b.get('bench_arena_used', 0):,} bytes")
        # The BOOT line is not always captured (the reader can attach after it
        # prints), so fall back to the ARENA_SIZE the build was compiled with.
        arena_req = b.get("boot_arena_req")
        if not arena_req:
            for d in b.get("defines", []):
                if d.startswith("ARENA_SIZE="):
                    arena_req = int(d.split("=", 1)[1])
        out.append(f"  arena requested      : {arena_req or 0:,} bytes")
        out.append(f"  total sketch flash   : {b.get('flash_bytes', 0):,} bytes "
                   f"({b.get('flash_pct', 0)}% of {FLASH_TOTAL:,})")
        out.append(f"  static RAM           : {b.get('static_ram_bytes', 0):,} bytes "
                   f"of {SRAM_TOTAL:,}")
        out.append(f"  largest free block   : {b.get('bench_free_sram', 0):,} bytes")
        out.append(f"  inference latency    : {b.get('bench_mean_us', 0) / 1000:.1f} ms "
                   f"mean, {b.get('bench_max_us', 0) / 1000:.1f} ms max "
                   f"({b.get('bench_runs', 0)} runs)")
        out.append(f"  device/host agreement: {b.get('bench_match')} "
                   f"(pred cls={b.get('bench_pred_cls')} "
                   f"exp={b.get('bench_exp_cls')}, "
                   f"portion={b.get('bench_pred_portion')}/"
                   f"{b.get('bench_exp_portion')})")

    live = [d for d in dev if d.get("timing_total_us")]
    if live:
        t = live[-1]
        out.append(f"\n  capture-to-result latency (live camera):")
        out.append(f"    capture    : {t.get('timing_capture_us', 0) / 1000:.1f} ms")
        out.append(f"    preprocess : {t.get('timing_preprocess_us', 0) / 1000:.1f} ms")
        out.append(f"    inference  : {t.get('timing_infer_us', 0) / 1000:.1f} ms")
        out.append(f"    TOTAL      : {t.get('timing_total_us', 0) / 1000:.1f} ms")

    out.append("\n" + "=" * 92)
    text = "\n".join(out)
    print(text)
    (RESULTS / "REPORT.txt").write_text(text)
    print(f"\nwrote {RESULTS / 'REPORT.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
