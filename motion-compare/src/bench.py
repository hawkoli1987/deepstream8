#!/usr/bin/env python3
"""
bench.py — one T4, five algorithms, strictly sequential, same loops as the
score extractors. For each algorithm: one untimed warm-up pass, then repeated
full-clip passes until the timed window reaches --min-window seconds (and at
least 2 passes). Pass walls are epoch-bracketed (t_start/t_end) so the 1 Hz
nvidia-smi poller (bench_sm.py) can attribute SM% per algorithm afterwards.

  cpu_absdiff / mog2      : extract_cv.py loops, byte for byte (incl. pool)
  gpu_absdiff             : cupy loop with H2D inside the timing
  gpu_absdiff_nox         : cupy loop with device arrays preloaded
                            (compute-only; H2D excluded) -> separate window
  farneback               : extract_farneback.farneback_pass (imported, not
                            duplicated) — e2e pass wall + per-frame t_calc
  nvof / decode           : full pipelines run through the Gst API (same
                            wiring as nvof_extract.c); nvof compute is
                            NOT separable in-pipeline -> collect_perf.py
                            reports e2e minus the decode-only baseline

usage: bench.py <cam6d.yuv> <W> <H> <F> <outdir> [--min-window 30] [--video /work/cam6d.mp4]
"""
import sys, os, json, time, statistics
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def bench_loop(name, pass_fn, min_window_s, nframes):
    """One untimed warm-up, then passes until >= min_window_s (>=2 passes)."""
    pass_fn()                                   # warm-up (untimed)
    pass_walls, extras = [], []
    t_start = time.time()
    while True:
        w, extra = pass_fn()
        pass_walls.append(w)
        extras.append(extra)
        if sum(pass_walls) >= min_window_s and len(pass_walls) >= 2:
            break
    t_end = time.time()
    per_frame_ms = [1000.0 * w / nframes for w in pass_walls]
    return dict(passes=len(pass_walls),
                per_frame_ms=[round(x, 3) for x in per_frame_ms],
                pass_ms=[round(1000 * w, 1) for w in pass_walls],
                e2e_ms_per_frame=round(statistics.median(per_frame_ms), 3),
                t_start=round(t_start, 3), t_end=round(t_end, 3)), extras


# ---- gst runs (Gst API — parse-launch cannot link decoder -> nvstreammux) -
def gst_run(kind, video, W, H):
    """One full pipeline run (set PLAYING -> EOS), timed. Built through the
    Gst API with the wiring of nvof_extract.c: parse-launch's link-time caps
    check refuses nvv4l2decoder -> nvstreammux in DS8 even with explicit NVMM
    caps, while a requested sink_0 pad links fine (that is how the C tool
    links it)."""
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)

    pipeline = Gst.Pipeline()
    src = Gst.ElementFactory.make("filesrc", "src")
    demux = Gst.ElementFactory.make("qtdemux", "demux")
    parse = Gst.ElementFactory.make("h264parse", "parse")
    dec = Gst.ElementFactory.make("nvv4l2decoder", "dec")
    sink = Gst.ElementFactory.make("fakesink", "sink")
    if None in (src, demux, parse, dec, sink):
        raise RuntimeError("gst element creation failed")
    src.set_property("location", video)
    sink.set_property("sync", False)
    sink.set_property("async", False)

    pipeline.add(src, demux, parse, dec, sink)
    if not src.link(demux):
        raise RuntimeError("filesrc -> qtdemux link failed")
    if not parse.link(dec):
        raise RuntimeError("h264parse -> nvv4l2decoder link failed")

    if kind == "nvof":
        mux = Gst.ElementFactory.make("nvstreammux", "mux")
        nvof = Gst.ElementFactory.make("nvof", "nvof")
        if None in (mux, nvof):
            raise RuntimeError("gst element creation failed (mux/nvof)")
        mux.set_property("batch-size", 1)
        mux.set_property("width", W)
        mux.set_property("height", H)
        mux.set_property("live-source", 0)
        mux.set_property("batched-push-timeout", 4000000)
        nvof.set_property("preset-level", 2)
        pipeline.add(mux, nvof)
        msink = mux.request_pad_simple("sink_0")
        dsrc = dec.get_static_pad("src")
        if msink is None or dsrc is None or \
                dsrc.link(msink) != Gst.PadLinkReturn.OK:
            raise RuntimeError("nvv4l2decoder -> streammux link failed")
        if not mux.link(nvof) or not nvof.link(sink):
            raise RuntimeError("streammux -> nvof -> sink link failed")
    else:
        if not dec.link(sink):
            raise RuntimeError("nvv4l2decoder -> fakesink link failed")

    def on_demux_pad(_el, pad):
        pad.link(parse.get_static_pad("sink"))
    demux.connect("pad-added", on_demux_pad)

    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        raise RuntimeError("pipeline failed to go PLAYING")
    t0 = time.perf_counter()
    bus = pipeline.get_bus()
    while True:
        msg = bus.timed_pop_filtered(500 * Gst.MSECOND,
                                     Gst.MessageType.EOS | Gst.MessageType.ERROR)
        if msg is None:
            continue
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"gst run failed: {err.message} | {dbg}")
        break
    dt = time.perf_counter() - t0
    pipeline.set_state(Gst.State.NULL)
    return dt


def bench_entry_from_runs(runs, nframes, t_start, t_end):
    per_frame_ms = [1000.0 * w / nframes for w in runs]
    return dict(passes=len(runs),
                per_frame_ms=[round(x, 3) for x in per_frame_ms],
                pass_ms=[round(1000 * w, 1) for w in runs],
                e2e_ms_per_frame=round(statistics.median(per_frame_ms), 3),
                t_start=round(t_start, 3), t_end=round(t_end, 3))


def main():
    if len(sys.argv) < 6:
        sys.exit("usage: bench.py <cam6d.yuv> <W> <H> <F> <outdir> "
                 "[--min-window 30] [--video /work/cam6d.mp4]")
    yuv, W, H, F, outdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), \
        int(sys.argv[4]), sys.argv[5]
    args = sys.argv[6:]
    min_window = 30.0
    video = "/work/cam6d.mp4"
    if "--min-window" in args:
        min_window = float(args[args.index("--min-window") + 1])
    if "--video" in args:
        video = args[args.index("--video") + 1]

    import cv2
    import cupy as cp
    sys.path.insert(0, HERE)
    from extract_farneback import pool, make_far, farneback_pass

    frame_bytes = W * H * 3 // 2
    assert os.path.getsize(yuv) == frame_bytes * F, "yuv size mismatch"
    buf = np.memmap(yuv, dtype=np.uint8, mode="r")
    Y = buf.reshape(F, frame_bytes)[:, : W * H].reshape(F, H, W)

    npz = np.load(os.path.join(outdir, "cv_scores.npz"))
    roi = npz["roi_mask"]
    roi_g = cp.asarray(roi)

    algos = {}

    # ---- CPU pair (identical loops to extract_cv.py) --------------------
    def cpu_pass():
        t0 = time.perf_counter()
        for t in range(1, F):
            d = np.abs(Y[t].astype(np.int16) - Y[t - 1].astype(np.int16))
            _ = d[roi].astype(np.float64).mean() / 255.0
            _ = pool(np.where(roi, d, 0.0))
        return time.perf_counter() - t0, None

    def mog_pass():
        t0 = time.perf_counter()
        bg = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=16,
                                                detectShadows=True)
        for t in range(F):
            fg = bg.apply(Y[t])
            _ = float((fg[roi] == 255).mean())
            _ = float((fg[roi] == 127).mean())
            _ = pool(np.where(roi, (fg == 255).astype(np.float32), 0.0))
        return time.perf_counter() - t0, None

    # ---- GPU quartet (strictly after the CPU pair) ----------------------
    def gpu_pass():
        t0 = time.perf_counter()
        for t in range(1, F):
            a = cp.asarray(Y[t]).astype(cp.int16)
            b = cp.asarray(Y[t - 1]).astype(cp.int16)
            dg = cp.abs(a - b)
            _ = float(dg[roi_g].astype(cp.float64).mean()) / 255.0
        cp.cuda.Stream.null.synchronize()
        return time.perf_counter() - t0, None

    def gpu_nox_pass():
        dev = [cp.asarray(Y[t]).astype(cp.int16) for t in range(F)]
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for t in range(1, F):
            dg = cp.abs(dev[t] - dev[t - 1])
            _ = float(dg[roi_g].astype(cp.float64).mean()) / 255.0
        cp.cuda.Stream.null.synchronize()
        return time.perf_counter() - t0, None

    def fb_pass():
        timings = {k: np.zeros(F, np.float64)
                   for k in ("t_upload", "t_calc", "t_download", "t_reduce")}
        t0 = time.perf_counter()
        _, _ = farneback_pass(Y, roi, far, prev, nxt, timings)
        w = time.perf_counter() - t0
        return w, {"t_calc_median_ms":
                   round(1000.0 * float(np.median(timings["t_calc"][1:])), 3)}

    # ---- instantiate GPU objects once ------------------------------------
    far = make_far()
    prev, nxt = cv2.cuda.GpuMat(), cv2.cuda.GpuMat()

    # ---- run: CPU pair, GPU quartet, then the two gst pipelines ----------
    algos["cpu_absdiff"], extras = bench_loop("cpu_absdiff", cpu_pass,
                                              min_window, F)
    print(f"cpu_absdiff  e2e {algos['cpu_absdiff']['e2e_ms_per_frame']:7.2f} ms/frame",
          flush=True)
    algos["mog2"], extras = bench_loop("mog2", mog_pass, min_window, F)
    print(f"mog2         e2e {algos['mog2']['e2e_ms_per_frame']:7.2f} ms/frame",
          flush=True)
    algos["gpu_absdiff"], extras = bench_loop("gpu_absdiff", gpu_pass,
                                              min_window, F)
    print(f"gpu_absdiff  e2e {algos['gpu_absdiff']['e2e_ms_per_frame']:7.2f} ms/frame",
          flush=True)
    algos["gpu_absdiff_nox"], extras = bench_loop("gpu_absdiff_nox",
                                                  gpu_nox_pass, min_window, F)
    print(f"gpu nox      e2e {algos['gpu_absdiff_nox']['e2e_ms_per_frame']:7.2f} ms/frame",
          flush=True)
    algos["farneback"], extras = bench_loop("farneback", fb_pass,
                                            min_window, F)
    fb_extra = [e for e in extras if e]
    algos["farneback"]["t_calc_median_ms"] = \
        round(statistics.median([e["t_calc_median_ms"] for e in fb_extra]), 3)
    print(f"farneback    e2e {algos['farneback']['e2e_ms_per_frame']:7.2f} "
          f"ms/frame  t_calc {algos['farneback']['t_calc_median_ms']:.2f} ms",
          flush=True)
    runs = []
    t_start = time.time()
    runs.append(gst_run("nvof", video, W, H))
    while (sum(runs) < min_window and len(runs) < 60):
        runs.append(gst_run("nvof", video, W, H))
    t_end = time.time()
    algos["nvof"] = bench_entry_from_runs(runs, F, t_start, t_end)
    print(f"nvof         e2e {algos['nvof']['e2e_ms_per_frame']:7.2f} ms/frame", flush=True)

    t_start = time.time()
    runs = [gst_run("decode", video, W, H)]
    while (sum(runs) < min_window and len(runs) < 60):
        runs.append(gst_run("decode", video, W, H))
    t_end = time.time()
    algos["decode"] = bench_entry_from_runs(runs, F, t_start, t_end)
    print(f"decode       e2e {algos['decode']['e2e_ms_per_frame']:7.2f} ms/frame", flush=True)

    doc = {"meta": {"gpu": str(cp.cuda.runtime.getDeviceProperties(0)["name"].decode()),
                    "W": W, "H": H, "F": F,
                    "min_window_s": min_window,
                    "video": video,
                    "farneback_params": dict(numLevels=5, pyrScale=0.5,
                                             fastPyramids=False, winSize=21,
                                             numIters=3, polyN=5, polySigma=1.1,
                                             flags=0)},
           "algos": algos}
    with open(os.path.join(outdir, "bench.json"), "w") as f:
        json.dump(doc, f, indent=2)
    print("wrote", os.path.join(outdir, "bench.json"))


if __name__ == "__main__":
    main()
