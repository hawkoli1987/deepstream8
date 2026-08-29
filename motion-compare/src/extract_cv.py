#!/usr/bin/env python3
"""
extract_cv.py — per-frame motion scores for three algorithms that run on the
luma (Y) plane: greyscale absdiff on CPU (numpy), the same on GPU (CuPy), and
MOG2 background subtraction (OpenCV, CPU).

Input : a raw yuv420p dump of the clip (all frames), + geometry on the CLI.
Output: <out>/cv_scores.npz  with, per algorithm:
          raw[F]        float64   intensive score, resolution-independent
          heat[F,GH,GW] uint8     pooled activity grid (0..255), pre-normalisation
        plus:
          roi_mask[H,W] bool      valid (non-corner) pixels
          valid_pixel_fraction    float
          runtime_s per algo, and the CPU/GPU parity numbers.

All three algorithms see the identical uint8 Y bytes.
"""
import sys, os, json, time
import numpy as np

GRID_W, GRID_H = 46, 43            # pooled overlay grid; ~32 px cells at 1472x1384


def pool(mapf: np.ndarray) -> np.ndarray:
    """Mean-pool a float H x W map to GRID_H x GRID_W via area resampling."""
    import cv2
    return cv2.resize(mapf.astype(np.float32), (GRID_W, GRID_H),
                      interpolation=cv2.INTER_AREA)


def main():
    if len(sys.argv) != 6:
        sys.exit("usage: extract_cv.py <cam6.yuv> <W> <H> <F> <outdir>")
    yuv_path, W, H, F, outdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), \
        int(sys.argv[4]), sys.argv[5]
    import cv2

    frame_bytes = W * H * 3 // 2
    expect = frame_bytes * F
    actual = os.path.getsize(yuv_path)
    assert actual == expect, f"yuv size {actual} != {expect} (W*H*1.5*F)"

    buf = np.memmap(yuv_path, dtype=np.uint8, mode="r")
    Y = buf.reshape(F, frame_bytes)[:, : W * H].reshape(F, H, W)   # luma only

    # ---- ROI mask: keep only the illuminated fisheye disc -------------------
    # The clip is a circular fisheye inscribed in the frame height; corners are
    # dead (compression-noisy) black. Estimate the disc from a per-column /
    # per-row profile of the time-mean luma, then take the inscribed circle.
    tmean = Y.mean(axis=0)
    col_on = np.where(tmean.mean(axis=0) > 16)[0]
    row_on = np.where(tmean.mean(axis=1) > 16)[0]
    cx = 0.5 * (col_on[0] + col_on[-1])
    cy = 0.5 * (row_on[0] + row_on[-1])
    rad = 0.5 * min(col_on[-1] - col_on[0], row_on[-1] - row_on[0])
    roi = np.zeros((H, W), np.uint8)
    cv2.circle(roi, (int(round(cx)), int(round(cy))), int(round(rad * 0.97)), 1, -1)
    roi = roi.astype(bool)
    print(f"fisheye disc: centre=({cx:.0f},{cy:.0f}) r={rad:.0f}")
    n_valid = int(roi.sum())
    valid_frac = n_valid / (W * H)
    print(f"ROI: {n_valid}/{W*H} px valid ({valid_frac:.3f})")

    raw_cpu = np.zeros(F, np.float64)
    raw_gpu = np.zeros(F, np.float64)
    raw_mog = np.zeros(F, np.float64)
    shadow = np.zeros(F, np.float64)
    heat_diff = np.zeros((F, GRID_H, GRID_W), np.float32)   # shared by cpu & gpu
    heat_mog = np.zeros((F, GRID_H, GRID_W), np.float32)

    # ---- 1 & 2: greyscale absdiff, CPU then GPU -----------------------------
    t0 = time.time()
    for t in range(1, F):
        d = np.abs(Y[t].astype(np.int16) - Y[t - 1].astype(np.int16))
        raw_cpu[t] = d[roi].astype(np.float64).mean() / 255.0
        heat_diff[t] = pool(np.where(roi, d, 0.0))
    rt_cpu = time.time() - t0

    parity_bit_identical = True
    max_abs_delta = 0.0
    try:
        import cupy as cp
        roi_g = cp.asarray(roi)
        # warm-up JIT
        _ = cp.abs(cp.asarray(Y[1]).astype(cp.int16) - cp.asarray(Y[0]).astype(cp.int16))
        cp.cuda.Stream.null.synchronize()

        t0 = time.time()
        for t in range(1, F):
            a = cp.asarray(Y[t]).astype(cp.int16)
            b = cp.asarray(Y[t - 1]).astype(cp.int16)
            dg = cp.abs(a - b)
            raw_gpu[t] = float(dg[roi_g].astype(cp.float64).mean()) / 255.0
        cp.cuda.Stream.null.synchronize()
        rt_gpu = time.time() - t0

        # GPU compute-only (exclude H2D): preload device arrays first
        dev = [cp.asarray(Y[t]).astype(cp.int16) for t in range(F)]
        cp.cuda.Stream.null.synchronize()
        t0 = time.time()
        for t in range(1, F):
            dg = cp.abs(dev[t] - dev[t - 1])
            _ = float(dg[roi_g].astype(cp.float64).mean())
        cp.cuda.Stream.null.synchronize()
        rt_gpu_nox = time.time() - t0
        del dev

        # parity: difference images must be bit-identical
        for t in range(1, F):
            d = np.abs(Y[t].astype(np.int16) - Y[t - 1].astype(np.int16))
            a = cp.asarray(Y[t]).astype(cp.int16)
            b = cp.asarray(Y[t - 1]).astype(cp.int16)
            if not np.array_equal(d, cp.asnumpy(cp.abs(a - b))):
                parity_bit_identical = False
        max_abs_delta = float(np.max(np.abs(raw_cpu - raw_gpu)))
        gpu_name = str(cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    except Exception as e:
        print(f"CuPy path failed: {e!r} — GPU series = CPU series")
        raw_gpu = raw_cpu.copy()
        rt_gpu = rt_gpu_nox = float("nan")
        gpu_name = "unavailable"

    # ---- 4: MOG2 ----------------------------------------------------------
    bg = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=16,
                                            detectShadows=True)
    t0 = time.time()
    for t in range(F):
        fg = bg.apply(Y[t])
        raw_mog[t] = float((fg[roi] == 255).mean())
        shadow[t] = float((fg[roi] == 127).mean())
        heat_mog[t] = pool(np.where(roi, (fg == 255).astype(np.float32), 0.0))
    rt_mog = time.time() - t0

    WARMUP = 10
    raw_mog[:WARMUP] = 0.0
    shadow[:WARMUP] = 0.0
    heat_mog[:WARMUP] = 0.0

    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "cv_scores.npz"),
             raw_cpu=raw_cpu, raw_gpu=raw_gpu, raw_mog=raw_mog, shadow=shadow,
             heat_diff=heat_diff, heat_mog=heat_mog,
             roi_mask=roi, grid_w=GRID_W, grid_h=GRID_H,
             valid_pixel_fraction=valid_frac,
             warmup=WARMUP,
             rt_cpu=rt_cpu, rt_gpu=rt_gpu, rt_gpu_nox=rt_gpu_nox, rt_mog=rt_mog)

    meta = dict(
        W=W, H=H, F=F, grid_w=GRID_W, grid_h=GRID_H,
        valid_pixel_fraction=valid_frac,
        parity_bit_identical=bool(parity_bit_identical),
        max_abs_delta_raw=max_abs_delta,
        rt_cpu_s=rt_cpu, rt_gpu_s=rt_gpu, rt_gpu_s_excl_transfer=rt_gpu_nox,
        rt_mog_s=rt_mog, gpu_name=gpu_name,
    )
    with open(os.path.join(outdir, "cv_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
