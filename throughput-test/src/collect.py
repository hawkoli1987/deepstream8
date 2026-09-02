#!/usr/bin/env python3
"""collect.py — merge per-N results + sm.csv into summary.{csv,json}.

summary.json shape (consumed by build_html.py -> report.html __DATA__):
    {"meta": {...}, "points": [{N, status, fps_total, ...}, ...]}
"""
import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

COLUMNS = ["N", "status", "fps_total", "fps_per_stream", "ms_per_batch",
           "sm_pct_mean", "power_w_mean", "mem_mb_max", "engine_build_s",
           "exit_code", "note"]


def load_sm(path):
    """sm.csv -> list of (epoch, sm_pct, power_w, mem_mb) floats (None on NA)."""
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            def _f(key):
                v = (r.get(key) or "").strip()
                try:
                    return float(v)
                except ValueError:
                    return None
            rows.append((_f("epoch"), _f("sm_pct"), _f("power_w"), _f("mem_mb")))
    return rows


def window_stats(rows, w0, w1):
    """Mean sm/power + max mem over [w0, w1]."""
    sm_v = sm_n = pw_v = pw_n = 0.0
    mem = None
    for e, s, p, m in rows:
        if e is None or not (w0 <= e <= w1):
            continue
        if s is not None:
            sm_v += s
            sm_n += 1
        if p is not None:
            pw_v += p
            pw_n += 1
        if m is not None and (mem is None or m > mem):
            mem = m
    return (sm_v / sm_n if sm_n else None,
            pw_v / pw_n if pw_n else None,
            mem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag")
    args = ap.parse_args()
    if not args.tag:
        args.tag = os.path.basename(os.path.normpath(args.run_dir))
    sm = load_sm(os.path.join(args.run_dir, "sm.csv"))
    points = []
    for name in sorted(os.listdir(args.run_dir)):
        if not name.startswith("N"):
            continue
        try:
            n = int(name[1:])
        except ValueError:
            continue
        ndir = os.path.join(args.run_dir, name)
        if not os.path.isdir(ndir):
            continue
        res = fail = None
        rp = os.path.join(ndir, "result.json")
        fp = os.path.join(ndir, "failure.json")
        if os.path.isfile(rp):
            with open(rp) as f:
                res = json.load(f)
        elif os.path.isfile(fp):
            with open(fp) as f:
                fail = json.load(f)
        else:
            continue
        win = ((res.get("counter", {}).get("window") or {})
               if res else {})
        if res and win.get("start_epoch") and win.get("end_epoch"):
            s, p, m = window_stats(sm, win["start_epoch"], win["end_epoch"])
        else:
            s = p = m = None
        row = {
            "N": n,
            "status": (res or fail or {}).get("status", "?"),
            "exit_code": (res or fail or {}).get("exit_code"),
            "fps_total": res.get("fps_total") if res else None,
            "fps_per_stream": res.get("fps_per_stream") if res else None,
            "ms_per_batch": (res.get("counter", {}).get("ms_per_batch")
                             if res else None),
            "sm_pct_mean": s, "power_w_mean": p, "mem_mb_max": m,
            "engine_build_s": res.get("engine_build_s") if res else None,
        }
        row["note"] = ((res.get("config", {}).get("engine_note") if res else None)
                       or (fail or {}).get("error")
                       or (fail or {}).get("status") or "")
        points.append(row)
    points.sort(key=lambda r: r["N"])
    gpu_name = driver = ds_ver = None
    cfg_note = warm = meas = None
    for pt in points:
        if pt["status"] != "ok":
            continue
        ndir = os.path.join(args.run_dir, "N%d" % pt["N"])
        with open(os.path.join(ndir, "result.json")) as f:
            res = json.load(f)
        gpu_name = (res.get("gpu", {}) or {}).get("name") or gpu_name
        driver = (res.get("gpu", {}) or {}).get("driver") or driver
        cfg = res.get("config", {}) or {}
        cfg_note = cfg.get("engine_note") or cfg_note
        warm = cfg.get("warmup_s") if cfg.get("warmup_s") is not None else warm
        meas = cfg.get("measure_s") if cfg.get("measure_s") is not None else meas

    meta = {
        "tag": args.tag,
        "gpu": gpu_name or "?",
        "driver": driver or "?",
        "ds": "8.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "warmup_s": warm,
        "measure_s": meas,
        "pgie_note": cfg_note,
        "sources_note": "in-container sample_720p.mp4, file-loop",
        "notes": [],
        "runs": [
            {"N": pt["N"], "status": pt["status"],
             "dir": "N%d" % pt["N"]} for pt in points
        ],
    }
    out_json = os.path.join(args.run_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump({"meta": meta, "points": points}, f, indent=2)
        f.write("\n")
    out_csv = os.path.join(args.run_dir, "summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for pt in points:
            w.writerow(pt)
    print("summary -> %s" % out_json)
    print("summary -> %s" % out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())