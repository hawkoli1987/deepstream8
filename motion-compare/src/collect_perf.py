#!/usr/bin/env python3
"""
collect_perf.py — merge bench.json + sm.csv (+ optional ncu-*.csv) → perf.json.

Numbers the HTML perf section shows, per algorithm:
  e2e_ms_per_frame   median full-clip pass wall / 80 frames
  compute_ms_median  what the algorithm's math alone costs per frame:
                     cpu/mog2 = e2e; gpu_absdiff = the preloaded nox variant;
                     farneback = median per-frame t_calc (D2H excluded);
                     nvof     = e2e − decode-only baseline (estimate)
  sm_pct_mean        mean nvidia-smi utilization.gpu over the algo's window
  ncu                per-kernel Duration / Occupancy / SM throughput (if any)

usage: collect_perf.py <outdir>
"""
import sys, os, json, csv, statistics

ROWS = ["cpu_absdiff", "mog2", "gpu_absdiff", "farneback", "nvof"]


def sm_window(sm_rows, t0, t1):
    """Mean utilization.gpu over [t0, t1] + sample count."""
    vals = [float(r["util_gpu"]) for r in sm_rows
            if t0 <= float(r["epoch"]) <= t1]
    if not vals:
        return None, 0
    return round(sum(vals) / len(vals), 1), len(vals)


def load_ncu(path):
    """Parse an `ncu --csv` details file into per-kernel aggregates (best effort)."""
    kernels, seen = {}, set()
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                k = row.get("Kernel Name")
                m = row.get("Metric Name")
                v = row.get("Metric Value")
                if not k or not m or v is None:
                    continue
                d = kernels.setdefault(k, {"launches": 0, "dur": [],
                                           "occ": [], "sm": []})
                if row.get("ID") is not None:
                    key = (row.get("ID"), k)
                    if key not in seen:
                        seen.add(key)
                        d["launches"] += 1
                try:
                    v = float(str(v).replace(",", ""))
                except ValueError:
                    continue
                if m == "Duration":
                    d["dur"].append(v)
                elif m == "Achieved Occupancy":
                    d["occ"].append(v)
                elif m == "Compute (SM) Throughput":
                    d["sm"].append(v)
    except OSError:
        return None
    return {k: {"launches": d["launches"],
                "duration_us": round(sum(d["dur"]) / len(d["dur"]), 1) if d["dur"] else None,
                "occupancy_pct": round(sum(d["occ"]) / len(d["occ"]), 1) if d["occ"] else None,
                "sm_throughput_pct": round(sum(d["sm"]) / len(d["sm"]), 1) if d["sm"] else None}
            for k, d in kernels.items()}


def main():
    outdir = sys.argv[1]
    bench = json.load(open(os.path.join(outdir, "bench.json")))
    sm = list(csv.DictReader(open(os.path.join(outdir, "sm.csv"))))
    algos_b = bench["algos"]

    perf = {"meta": dict(bench["meta"]),
            "algos": {}}

    decode_e2e = algos_b["decode"]["e2e_ms_per_frame"]
    for aid in ROWS:
        b = algos_b[aid]
        sm_mean, sm_n = sm_window(sm, b["t_start"], b["t_end"])
        e2e = b["e2e_ms_per_frame"]
        if aid == "gpu_absdiff":
            nox = algos_b["gpu_absdiff_nox"]
            comp = nox["e2e_ms_per_frame"]
            note = ("compute = cupy loop with arrays preloaded on device "
                    f"({nox['passes']} passes, H2D excluded)")
            nox_sm, nox_n = sm_window(sm, nox["t_start"], nox["t_end"])
        elif aid == "farneback":
            comp = b["t_calc_median_ms"]
            note = "compute = median per-frame t_calc (D2H download excluded)"
        elif aid == "nvof":
            comp = round(e2e - decode_e2e, 3)
            note = "compute = e2e − decode-only baseline (estimate)"
        else:
            comp = e2e
            note = ""
        perf["algos"][aid] = {
            "passes": b["passes"], "window_s": round(b["t_end"] - b["t_start"], 1),
            "e2e_ms_per_frame": e2e, "compute_ms_median": comp,
            "sm_pct_mean": sm_mean, "sm_n": sm_n,
            "note": note}
    if "gpu_absdiff_nox" in algos_b:
        nox = algos_b["gpu_absdiff_" + "nox"]
        perf["algos"]["gpu_absdiff"]["sm_pct_mean_nox"] = nox_sm
        perf["algos"]["gpu_absdiff"]["sm_n_nox"] = nox_n

    # ---- ncu (best effort) ------------------------------------------------
    ncu = {"available": False, "kernels": [], "note": ""}
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True)
        perf["meta"]["driver"] = r.stdout.strip()
    except Exception:
        pass
    for aid, fname in [("gpu_absdiff", "ncu_gpu.csv"),
                       ("farneback", "ncu_far.csv"),
                       ("nvof", "ncu_nvof.csv")]:
        p = os.path.join(outdir, fname)
        if not os.path.exists(p):
            continue
        parsed = load_ncu(p)
        if parsed:
            ncu["available"] = ncu["available"] or bool(parsed)
            for k, d in parsed.items():
                ncu["kernels"].append({"algo": aid, "kernel": k,
                                       **d})
    perf["ncu"] = ncu

    # frame count from the run's own cv metadata (v1 cam6: 80; v2 clip2: 127)
    F_txt = str(json.load(open(os.path.join(outdir, "cv_meta.json")))["F"])
    perf["method"] = [
        f"Same {F_txt}-frame clip, one algorithm at a time, strictly sequential, "
        "one untimed warm-up pass then repeated full-clip passes to ≥30 s of GPU time per algorithm.",
        f"e2e = median full-clip pass wall / {F_txt}. compute = see each row's note "
        "(nvof cannot be isolated inside its pipeline: e2e minus the decode-only baseline, an estimate).",
        "SM% = mean of 1 Hz nvidia-smi utilization.gpu over each algorithm's timed window "
        "(a global counter; windows do not overlap because algorithms run sequentially).",
        "Driver/CUDA: see meta. Nsight Compute kernel detail was attempted for "
        "gpu_absdiff / farneback / nvof but blocked: the container does not have "
        "GPU performance-counter permission (ERR_NVGPUCTRPERM), so the "
        "occupancy dimension here is the sampled SM-busy% column only.",
    ]
    json.dump(perf, open(os.path.join(outdir, "perf.json"), "w"), indent=2)
    print("wrote", os.path.join(outdir, "perf.json"))


if __name__ == "__main__":
    main()
