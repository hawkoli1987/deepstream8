#!/usr/bin/env python3
"""
extract_farneback.py — 5th lane: OpenCV CUDA Farnebäck dense optical flow.

Per frame t>=1: upload uint8 Y, run cv2.cuda.FarnebackOpticalFlow (5-level
half-res pyramid, 3 warping iterations/level, 21x21 window, 5-tap polynomial
expansion sigma=1.1; flags=0 — the CUDA impl has no Gaussian variant), hypot of
the CV_32FC2 output, mean |flow| over the fisheye ROI / frame diagonal
(identical raw unit to the nvof lane).

Scores and per-frame timings both come from ONE pass (the score pass is the
timed pass); bench.py imports farneback_pass() rather than duplicating the
loop, so timed loops and score loops cannot drift apart.

usage: extract_farneback.py <cam6d.yuv> <W> <H> <F> <outdir>
"""
import sys, os, json, time
import numpy as np

GRID_W, GRID_H = 46, 43            # pooled overlay grid; matches extract_cv.py


def pool(mapf):
    """Mean-pool a float H x W map to GRID_H x GRID_W via area resampling."""
    import cv2
    return cv2.resize(mapf.astype(np.float32), (GRID_W, GRID_H),
                      interpolation=cv2.INTER_AREA)


def make_far():
    """The one Farnebäck parameter set used by both the score run and bench."""
    import cv2
    return cv2.cuda.FarnebackOpticalFlow.create(
        numLevels=5, pyrScale=0.5, fastPyramids=False, winSize=21,
        numIters=3, polyN=5, polySigma=1.1, flags=0)


def farneback_pass(Y, roi, far, prev, nxt, timings=None):
    """
    One full-clip pass: scores + per-frame timings in a single loop.

    Y    : uint8 np.memmap view [F,H,W] (luma)
    roi  : bool [H,W]
    far  : cv2.cuda.FarnebackOpticalFlow
    prev/nxt : reusable cv2.cuda.GpuMat
    timings: dict of float64[F] arrays (t_upload/t_calc/t_download/t_reduce)
             or None to skip timing arrays
    returns (raw[F] float64, heat[F,GH,GW] float32)
    """
    import cv2
    F, H, W = Y.shape
    diag = float(np.hypot(W, H))
    raw = np.zeros(F, np.float64)
    heat = np.zeros((F, GRID_H, GRID_W), np.float32)
    stream = cv2.cuda.Stream()
    for t in range(1, F):
        u0 = time.perf_counter()
        prev.upload(Y[t - 1])
        nxt.upload(Y[t])
        stream.waitForCompletion()
        t_upload = time.perf_counter() - u0
        c0 = time.perf_counter()
        flow = far.calc(prev, nxt, None)
        stream.waitForCompletion()
        t_calc = time.perf_counter() - c0
        d0 = time.perf_counter()
        f_host = flow.download()
        stream.waitForCompletion()
        t_download = time.perf_counter() - d0
        r0 = time.perf_counter()
        mag = np.hypot(f_host[..., 0], f_host[..., 1])
        raw[t] = float(mag[roi].astype(np.float64).mean()) / diag
        heat[t] = pool(np.where(roi, mag, 0.0))
        t_reduce = time.perf_counter() - r0
        if timings is not None:
            timings["t_upload"][t] = t_upload
            timings["t_calc"][t] = t_calc
            timings["t_download"][t] = t_download
            timings["t_reduce"][t] = t_reduce
    return raw, heat


def main():
    if len(sys.argv) != 6:
        sys.exit("usage: extract_farneback.py <cam6d.yuv> <W> <H> <F> <outdir>")
    yuv_path, W, H, F, outdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), \
        int(sys.argv[4]), sys.argv[5]
    import cv2

    frame_bytes = W * H * 3 // 2
    expect = frame_bytes * F
    actual = os.path.getsize(yuv_path)
    assert actual == expect, f"yuv size {actual} != {expect} (W*H*1.5*F)"

    buf = np.memmap(yuv_path, dtype=np.uint8, mode="r")
    Y = buf.reshape(F, frame_bytes)[:, : W * H].reshape(F, H, W)

    # ROI: reuse the exact mask from the cv lane — no recomputation drift.
    npz = np.load(os.path.join(outdir, "cv_scores.npz"))
    roi = npz["roi_mask"]
    assert roi.shape == (H, W), roi.shape
    print(f"ROI from cv_scores.npz: {int(roi.sum())} px")

    far = make_far()
    prev, nxt = cv2.cuda.GpuMat(), cv2.cuda.GpuMat()
    timings = {k: np.zeros(F, np.float64)
               for k in ("t_upload", "t_calc", "t_download", "t_reduce")}
    t0 = time.time()
    raw, heat = farneback_pass(Y, roi, far, prev, nxt, timings)
    rt = time.time() - t0
    raw[0] = 0.0   # no previous frame — masked invalid downstream
    heat[0] = 0.0

    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, "fb_scores.npz"),
             raw_far=raw, heat_far=heat,
             t_upload=timings["t_upload"], t_calc=timings["t_calc"],
             t_download=timings["t_download"], t_reduce=timings["t_reduce"],
             n_passes=1, rt_pass_s=rt)

    meta = dict(W=W, H=H, F=F, rt_pass_s=rt,
                t_calc_sum_s=float(timings["t_calc"].sum()),
                t_calc_median_ms=1000.0 * float(np.median(timings["t_calc"][1:])),
                params=dict(numLevels=5, pyrScale=0.5, fastPyramids=False,
                            winSize=21, numIters=3, polyN=5, polySigma=1.1,
                            flags=0),
                cv2_version=cv2.__version__)
    with open(os.path.join(outdir, "fb_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"pass wall {rt:.2f}s | median t_calc {meta['t_calc_median_ms']:.1f} ms/frame")


if __name__ == "__main__":
    main()
