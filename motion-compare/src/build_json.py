#!/usr/bin/env python3
"""
build_json.py — fuse the four algorithms' raw per-frame scores into the single
JSON the HTML consumes: percentile normalisation, Otsu thresholding (with a
robust fallback), per-frame motion/static labels, and base64 pooled heat grids.

Inputs  (in <outdir>):
  cv_scores.npz         from extract_cv.py
  cv_meta.json          "
  of_mv.raw, of_index.csv   from nvof_extract (C)
Output:
  <outdir>/motion_scores.json
"""
import sys, os, json, base64
import numpy as np

FPS = 8.0   # nominal display rate of the de-duplicated clip


def otsu_threshold(x, bins=256):
    """Otsu on samples x in [0,1]; returns (thr, separability eta in [0,1])."""
    x = x[np.isfinite(x)]
    hist, edges = np.histogram(x, bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * w0 - mu) ** 2 / (w0 * w1)
    sigma_b = np.nan_to_num(sigma_b)
    k = int(np.argmax(sigma_b))
    total_var = float(np.sum(p * (centres - mu_t) ** 2))
    eta = float(sigma_b[k] / total_var) if total_var > 0 else 0.0
    return float(centres[k]), eta


def kmeans2(x, iters=25):
    x = x[np.isfinite(x)]
    c = np.array([np.percentile(x, 15), np.percentile(x, 85)], np.float64)
    for _ in range(iters):
        d = np.abs(x[:, None] - c[None, :])
        lab = d.argmin(1)
        for j in (0, 1):
            if np.any(lab == j):
                c[j] = x[lab == j].mean()
    return float(c.mean())


def robust_fallback(x):
    x = x[np.isfinite(x)]
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(np.clip(med + 3 * 1.4826 * mad, 0.05, 0.95))


def pick_threshold(norm, valid):
    x = norm[valid]
    t_otsu, eta = otsu_threshold(x)
    t_km = kmeans2(x)
    t_fb = robust_fallback(x)
    if eta >= 0.55:
        return t_otsu, "otsu", eta, dict(otsu=t_otsu, kmeans2=t_km, median_3mad=t_fb)
    return t_fb, "median_3mad", eta, dict(otsu=t_otsu, kmeans2=t_km, median_3mad=t_fb)


def pool_to_grid(mapf, gw, gh):
    import cv2
    return cv2.resize(mapf.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)


def heat_b64(heat_f_gh_gw):
    """Per-algorithm p99 scale, quantise to uint8, base64. Returns (b64, p99)."""
    h = np.asarray(heat_f_gh_gw, np.float64)
    p99 = float(np.percentile(h, 99)) or 1.0
    q = np.clip(h / p99, 0, 1) * 255.0
    return base64.b64encode(q.astype(np.uint8).tobytes()).decode(), p99


def norm_series(raw, valid, lo=None, hi=None):
    if lo is None:
        v = raw[valid]
        lo, hi = np.percentile(v, [1, 99])
    if hi <= lo:
        hi = lo + 1e-12
    n = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
    n[~valid] = 0.0
    return n, float(lo), float(hi)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "out"
    z = np.load(os.path.join(outdir, "cv_scores.npz"))
    cvm = json.load(open(os.path.join(outdir, "cv_meta.json")))
    W, H, F = cvm["W"], cvm["H"], cvm["F"]
    GW, GH = int(z["grid_w"]), int(z["grid_h"])
    WARMUP = int(z["warmup"])

    raw_cpu = z["raw_cpu"].copy()
    raw_gpu = z["raw_gpu"].copy()
    raw_mog = z["raw_mog"].copy()
    shadow = z["shadow"].copy()
    heat_diff = z["heat_diff"]
    heat_mog = z["heat_mog"]

    # ---- NvOF: read MV grids, magnitude, pool -----------------------------
    import csv, cv2
    idx = list(csv.DictReader(open(os.path.join(outdir, "of_index.csv"))))
    mvbuf = np.fromfile(os.path.join(outdir, "of_mv.raw"), dtype=np.int16)
    diag = float(np.hypot(W, H))
    roi_full = z["roi_mask"]
    raw_nvof = np.zeros(F, np.float64)
    heat_nvof = np.zeros((F, GH, GW), np.float32)
    of_rows = of_cols = 0
    of_bmask = None
    for r in idx:
        fn = int(r["frame_num"]); rows = int(r["rows"]); cols = int(r["cols"])
        off = int(r["byte_offset"]) // 2; n = int(r["byte_len"]) // 2
        of_rows, of_cols = rows, cols
        if of_bmask is None:                          # block-level fisheye mask
            of_bmask = cv2.resize(roi_full.astype(np.float32), (cols, rows),
                                  interpolation=cv2.INTER_AREA) >= 0.5
        g = mvbuf[off:off + n].reshape(rows, cols, 2).astype(np.float64) / 32.0
        mag = np.hypot(g[..., 0], g[..., 1])          # pixels of displacement
        i = fn - 1                                    # frame_num 2..120 -> idx 1..119
        if 0 <= i < F:
            raw_nvof[i] = mag[of_bmask].mean() / diag  # resolution-independent
            heat_nvof[i] = pool_to_grid(np.where(of_bmask, mag, 0.0), GW, GH)

    # ---- validity masks --------------------------------------------------
    v_diff = np.zeros(F, bool); v_diff[1:] = True
    v_nvof = np.zeros(F, bool); v_nvof[1:] = True
    v_mog = np.zeros(F, bool); v_mog[WARMUP:] = True

    # ---- normalisation (CPU & GPU share constants) ----------------------
    both_valid = v_diff
    lo_d, hi_d = np.percentile(np.concatenate([raw_cpu[both_valid], raw_gpu[both_valid]]),
                               [1, 99])
    n_cpu, lo_c, hi_c = norm_series(raw_cpu, v_diff, lo_d, hi_d)
    n_gpu, lo_g, hi_g = norm_series(raw_gpu, v_diff, lo_d, hi_d)
    n_nvof, lo_n, hi_n = norm_series(raw_nvof, v_nvof)
    n_mog, lo_m, hi_m = norm_series(raw_mog, v_mog)

    def block(id_, label, device, unit, raw, norm, valid, lo, hi, heat, extra=None):
        thr, meth, eta, cands = pick_threshold(norm, valid)
        motion = (norm >= thr) & valid
        b64, p99 = heat_b64(heat)
        d = dict(
            id=id_, label=label, device=device, raw_unit=unit,
            first_valid_frame=int(np.argmax(valid)),
            valid=[bool(x) for x in valid],
            raw=[None if not valid[i] else float(raw[i]) for i in range(F)],
            norm=[None if not valid[i] else float(norm[i]) for i in range(F)],
            norm_constants=dict(lo=lo, hi=hi, method="percentile_1_99"),
            threshold=float(thr), threshold_method=meth,
            threshold_separability=float(eta),
            threshold_candidates={k: float(x) for k, x in cands.items()},
            threshold_override=None,
            motion=[None if not valid[i] else bool(motion[i]) for i in range(F)],
            heat_b64=b64, heat_p99=p99,
        )
        if extra:
            d["extra"] = extra
        return d

    algos = [
        block("cpu_absdiff", "Greyscale absdiff", "CPU  numpy",
              "mean |dY| over valid px, /255", raw_cpu, n_cpu, v_diff, lo_c, hi_c,
              heat_diff, dict(runtime_s=float(cvm["rt_cpu_s"]))),
        block("gpu_absdiff", "Greyscale absdiff", "GPU  CuPy (Tesla T4)",
              "mean |dY| over valid px, /255", raw_gpu, n_gpu, v_diff, lo_g, hi_g,
              heat_diff, dict(runtime_s=float(cvm["rt_gpu_s"]),
                              runtime_s_excl_transfer=float(cvm["rt_gpu_s_excl_transfer"]),
                              heat_shared_with="cpu_absdiff")),
        block("nvof", "NVIDIA Optical Flow Accelerator", "GPU  NvOFA (Tesla T4)",
              "mean |flow| px / frame-diagonal", raw_nvof, n_nvof, v_nvof, lo_n, hi_n,
              heat_nvof, dict(of_rows=of_rows, of_cols=of_cols,
                              block_size=int(round(H / max(of_rows, 1))),
                              raw_px=[None if not v_nvof[i] else float(raw_nvof[i] * diag)
                                      for i in range(F)])),
        block("mog2", "MOG2 background subtraction", "CPU  OpenCV",
              "foreground px fraction", raw_mog, n_mog, v_mog, lo_m, hi_m,
              heat_mog, dict(runtime_s=float(cvm["rt_mog_s"]), warmup_frames=WARMUP,
                             history=20, var_threshold=16,
                             shadows_counted_as_motion=False,
                             shadow_fraction=[float(s) for s in shadow])),
    ]

    keep = json.load(open(os.path.join(outdir, "keep.json")))
    doc = dict(
        schema_version=1,
        video=dict(file="sample_cam6.mp4", width=W, height=H, fps=FPS,
                   frame_count=F, duration_s=F / FPS, codec="h264",
                   diagonal_px=diag,
                   note="360-degree fisheye over a parking area; aisle traffic vs parked cars",
                   source_note=(f"sample_cam6.mp4 ships with a broken 3-frame cadence "
                                f"(every 3rd frame a duplicate). De-duplicated to {F} unique "
                                f"frames from the original 120; shown here at {FPS:g} fps."),
                   kept_source_frames=keep["keep"]),
        grid=dict(cols=GW, rows=GH),
        normalization=dict(method="percentile_robust", p_lo=1, p_hi=99,
                           shared_constants_groups=[["cpu_absdiff", "gpu_absdiff"]],
                           caveat=("raw[] is intensive and resolution-independent -> comparable "
                                   "across clips. norm[] is clip-relative -> NOT comparable "
                                   "across clips.")),
        parity=dict(pair=["cpu_absdiff", "gpu_absdiff"],
                    diff_images_bit_identical=bool(cvm["parity_bit_identical"]),
                    max_abs_delta_raw=float(cvm["max_abs_delta_raw"]),
                    identical_after_norm=bool(np.allclose(n_cpu, n_gpu, atol=1e-12)),
                    cpu_runtime_s=float(cvm["rt_cpu_s"]),
                    gpu_runtime_s=float(cvm["rt_gpu_s"]),
                    gpu_runtime_s_excl_transfer=float(cvm["rt_gpu_s_excl_transfer"])),
        run_log=dict(gpu=cvm["gpu_name"], driver="595.91.07", cuda="12.8",
                     deepstream="8.0.0"),
        algorithms=algos,
    )
    p = os.path.join(outdir, "motion_scores.json")
    json.dump(doc, open(p, "w"))
    kb = os.path.getsize(p) / 1024
    print(f"wrote {p}  ({kb:.0f} KB)")
    for a in algos:
        nm = sum(1 for x in a["motion"] if x)
        nv = sum(1 for x in a["valid"] if x)
        print(f"  {a['id']:12s} thr={a['threshold']:.3f} ({a['threshold_method']}, "
              f"eta={a['threshold_separability']:.2f})  motion {nm}/{nv} valid frames")


if __name__ == "__main__":
    main()
