#!/usr/bin/env python3
"""app_throughput.py — DS8 throughput-only perception pipeline, one batch-size point.

N x nvurisrcbin -> nvstreammux -> nvinfer(YOLO26m) -> vehicle-filter probe -> nvtracker
-> fakesink (Nullwriter: nothing rendered, nothing emitted).

Built through the Gst API (never parse-launch: DS8 parse-launch refuses
decoder->nvstreammux links; a requested sink_%d pad links fine).
Each mux batch buffer carries N frames; a fakesink probe counts batches over a
steady-state window and the app derives fps.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

import pyds  # noqa: N811

# exit codes
EX_OK = 0
EX_EXCEPTION = 1
EX_NOPLAY = 2
EX_GST_ERROR = 3
EX_NO_BUFFERS = 4
EX_NO_WINDOW = 5

DEFAULT_URI = "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4"
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
SCHEMA = "throughput-test/v1"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10,
                    help="num sources == streammux batch size (default 10)")
    ap.add_argument("--uri", action="append", default=None,
                    help="file:// or rtsp:// source; repeatable, cycles to fill N; "
                         "default %s" % DEFAULT_URI)
    ap.add_argument("--pgie-config", required=True)
    ap.add_argument("--tracker-config", required=True)
    ap.add_argument("--pgie-batch", type=int, default=None,
                    help="nvinfer batch size; default = N")
    ap.add_argument("--mux-width", type=int, default=1920)
    ap.add_argument("--mux-height", type=int, default=1080)
    ap.add_argument("--mux-timeout-us", type=int, default=45000)
    ap.add_argument("--warmup", type=float, default=20.0)
    ap.add_argument("--measure", type=float, default=30.0)
    ap.add_argument("--class-filter", default="2,3,5,7")
    ap.add_argument("--probe-sample-sec", type=float, default=5.0)
    ap.add_argument("--source-bin", choices=["nvurisrcbin", "custom"], default="nvurisrcbin")
    ap.add_argument("--engine-note", default="")
    ap.add_argument("--out", default=None, help="write result JSON here")
    ap.add_argument("--log", default=None, help="tee app log here")
    return ap.parse_args()


def set_props(element, props):
    """Set GObject properties, warning (not crashing) on unknown names."""
    for key, val in props.items():
        try:
            element.set_property(key, val)
        except TypeError as e:
            print("WARN: %s: %s" % (element.get_name(), e), file=sys.stderr)


def gpu_fingerprint():
    """(name, driver) via nvidia-smi; ('?', '?') if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            timeout=10).decode().strip().splitlines()[0]
        parts = [p.strip() for p in out.split(",")]
        return parts[0], parts[1] if len(parts) > 1 else "?"
    except Exception:
        return "?", "?"


def tee(log_path):
    """Log sink writing to stderr and optionally to --log."""
    class _Tee:
        def write(self, s):
            sys.__stderr__.write(s)
            if log_path:
                with open(log_path, "a") as f:
                    f.write(s)

        def flush(self):
            sys.__stderr__.flush()

    return _Tee()


def build_pipeline(a):
    """Create mux/pgie/tracker/sink and wire the static chain."""
    n = a.n
    pipeline = Gst.Pipeline()
    mux = Gst.ElementFactory.make("nvstreammux", "mux")
    pgie = Gst.ElementFactory.make("nvinfer", "pgie")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    sink = Gst.ElementFactory.make("fakesink", "sink")
    if None in (mux, pgie, tracker, sink):
        raise RuntimeError("element creation failed (mux/pgie/tracker/sink)")

    set_props(mux, {
        "batch-size": n,
        "width": a.mux_width,
        "height": a.mux_height,
        "live-source": 0,
        "batched-push-timeout": a.mux_timeout_us,
        "attach-sys-ts": 1,
    })
    set_props(pgie, {
        "config-file-path": a.pgie_config,
        "batch-size": a.pgie_batch or n,
        "interval": 0,
    })
    set_props(tracker, {
        "tracker-width": 960,
        "tracker-height": 544,
        "ll-lib-file": TRACKER_LIB,
        "ll-config-file": a.tracker_config,
        "compute-hw": 1,
    })
    set_props(sink, {"sync": False, "async": False})
    pipeline.add(mux, pgie, tracker, sink)
    if not mux.link(pgie) or not pgie.link(tracker) or not tracker.link(sink):
        raise RuntimeError("mux -> pgie -> tracker -> sink link failed")
    return pipeline, mux


def add_sources(pipeline, mux, uris, source_bin="nvurisrcbin"):
    """Add N sources; request all mux sink pads eagerly, link in pad-added."""
    n = len(uris)
    mux_pads = []
    for i in range(n):
        pad = mux.request_pad_simple("sink_%d" % i)
        if pad is None:
            raise RuntimeError("mux.request_pad_simple(sink_%d) failed" % i)
        mux_pads.append(pad)

    for i, uri in enumerate(uris):
        if source_bin == "custom":
            src = custom_source_bin("src_%d" % i, uri)
        else:
            src = Gst.ElementFactory.make("nvurisrcbin", "src_%d" % i)
            set_props(src, {
                "uri": uri,
                "file-loop": 1,
                "num-extra-surfaces": 32,
                "gpu-id": 0,
            })
        pipeline.add(src)

        spad = src.get_static_pad("src")
        if spad is not None:
            if spad.link(mux_pads[i]) != Gst.PadLinkReturn.OK:
                raise RuntimeError("source %d: direct link failed" % i)
        else:
            # nvurisrcbin exposes its ghost pad only after internal decode
            # starts; link there. mux_pads[i] was already requested above.
            src.connect("pad-added", on_src_pad, mux_pads[i])
    return n


def on_src_pad(element, pad, mux_pad):
    """Link a source's late src pad to its pre-requested mux sink pad."""
    name = element.get_name()
    ret = pad.link(mux_pad)
    if ret != Gst.PadLinkReturn.OK:
        print("ERROR: %s pad-added link failed: %s" % (name, ret), file=sys.stderr)


def custom_source_bin(name, uri):
    """Fallback bin: uridecodebin -> nvv4l2decoder, exposed as a src GhostPad."""
    ret = Gst.Bin.new(name)
    decodebin = Gst.ElementFactory.make("uridecodebin", None)
    if decodebin is None:
        raise RuntimeError("uridecodebin creation failed")
    decodebin.set_property("uri", uri)
    ret.add(decodebin)

    def _on_pad(_el, pad):
        ghost = ret.get_static_pad("src")
        if ghost is not None:
            ghost.set_target(pad)
            return
        ghost = Gst.GhostPad.new_no_target_from_template("src", pad)
        ghost.set_target(pad)
        ret.add_pad(ghost)

    decodebin.connect("pad-added", _on_pad)
    return ret


def make_class_filter_probe(keep_ids, sample_sec):
    """PGIE src-pad probe: keep COCO keep_ids, remap survivors to class_id=0 ("VOI")
    before the tracker. Object walking only during the first sample_sec of flow;
    afterwards the probe is a cheap frame counter."""
    stats = {"frames": 0, "kept": 0, "removed": 0, "t_first": None}
    deadline = None

    def probe(pad, info):
        nonlocal deadline
        try:
            buf = info.get_buffer()
            if buf is None:
                return Gst.PadProbeReturn.OK
            bmeta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            if bmeta is None:
                return Gst.PadProbeReturn.OK
            if stats["t_first"] is None:
                stats["t_first"] = time.monotonic()
            sampling = (time.monotonic() - stats["t_first"]) < sample_sec
            l_frame = bmeta.frame_meta_list
            while l_frame is not None:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                stats["frames"] += 1
                if sampling:
                    l_obj = frame_meta.obj_meta_list
                    while l_obj is not None:
                        obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                        l_next = l_obj.next  # capture before any removal
                        if obj.class_id in keep_ids:
                            obj.class_id = 0  # remap to VOI
                            stats["kept"] += 1
                        else:
                            pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj)
                            stats["removed"] += 1
                        l_obj = l_next
                l_frame = l_frame.next
        except Exception as exc:  # never break the pipeline from a probe
            print("WARN: class-filter probe: %s" % exc, file=sys.stderr)
        return Gst.PadProbeReturn.OK

    return probe, stats


def make_sink_counter_probe(counter):
    """fakesink sink-pad probe counting mux batch buffers (each batch = N frames)."""
    def probe(pad, info):
        buf = info.get_buffer()
        if buf is not None:
            counter["batches"] += 1
        return Gst.PadProbeReturn.OK

    return probe


def run_measurement(pipeline, sink, n, a, probe_stats):
    """PLAYING -> wait first batch -> warmup -> measure window -> EOS -> NULL.
    Returns (ok, counters) where ok is False if the run never produced buffers."""
    counter = {"batches": 0}
    sink.get_static_pad("sink").add_probe(
        Gst.PadProbeType.BUFFER, make_sink_counter_probe(counter))

    bus = pipeline.get_bus()
    t0 = time.monotonic()
    if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
        print("ERROR: set_state(PLAYING) FAILED", file=sys.stderr)
        return "noplay", counter
    pipeline.get_state(5 * Gst.SECOND)

    # wait for the first batch at fakesink (engine load + decode spin-up)
    b_prev = counter["batches"]
    deadline = time.monotonic() + 180.0
    while counter["batches"] == b_prev and time.monotonic() < deadline:
        msg = bus.timed_pop_filtered(100 * Gst.MSECOND, Gst.MessageType.ERROR)
        if msg is not None:
            err, dbg = msg.parse_error()
            print("GST-ERROR: %s (%s)" % (err.message, dbg), file=sys.stderr)
            return "gst_error", counter
    if counter["batches"] == b_prev:
        print("ERROR: no batch reached fakesink within 5 minute deadline", file=sys.stderr)
        return "no_buffers", counter

    time.sleep(a.warmup)
    t_start = time.monotonic()
    w_start = time.time()
    b_start = counter["batches"]
    time.sleep(a.measure)
    t_end = time.monotonic()
    w_end = time.time()
    b_end = counter["batches"]

    pipeline.send_event(Gst.Event.new_eos())
    pipeline.get_state(2 * Gst.SECOND)
    pipeline.set_state(Gst.State.NULL)

    batches = b_end - b_start
    dt = t_end - t_start
    out = {
        "batches": batches,
        "frames": batches * n,
        "dt_s": dt,
        "ms_per_batch": (dt * 1000.0 / batches) if batches else None,
        "window": {"start_epoch": w_start, "end_epoch": w_end},
    }
    return "ok", {"counter": out, "probe": dict(probe_stats)}


def cycle_uris(uri_args, n):
    """Repeat the --uri list to fill N sources."""
    base = uri_args or [DEFAULT_URI]
    return [base[i % len(base)] for i in range(n)]


def write_result(path, payload):
    """Atomic JSON write (tmp + os.replace)."""
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    print("result -> %s" % path)


def main():
    a = parse_args()
    if a.log:
        sys.stderr = tee(a.log)
    Gst.init(None)
    n = a.n
    keep = set(int(x) for x in a.class_filter.split(",") if x.strip())

    pipeline, mux = build_pipeline(a)
    pgie = pipeline.get_by_name("pgie")
    sink = pipeline.get_by_name("sink")
    probe, probe_stats = make_class_filter_probe(keep, a.probe_sample_sec)
    pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe)

    add_sources(pipeline, mux, cycle_uris(a.uri, n), a.source_bin)

    reason, meas = run_measurement(pipeline, sink, n, a, probe_stats)
    code = EX_OK
    if reason != "ok":
        code = {"noplay": EX_NOPLAY, "gst_error": EX_GST_ERROR,
                "no_buffers": EX_NO_BUFFERS}.get(reason, EX_EXCEPTION)

    res = {
        "schema": SCHEMA,
        "N": n,
        "status": reason,
        "exit_code": code,
        "uris": cycle_uris(a.uri, n)[:1] + ["..."] if n > 1 else cycle_uris(a.uri, n),
        "gpu": dict(zip(("name", "driver"), gpu_fingerprint())),
        "counter": (meas or {}).get("counter") or {},
        "probe": (meas or {}).get("probe") or probe_stats,
    }
    cnt = res["counter"]
    res["fps_total"] = (cnt["frames"] / cnt["dt_s"]) if cnt.get("dt_s") else None
    res["fps_per_stream"] = (res["fps_total"] / n) if res["fps_total"] else None
    res["config"] = {
        "pgie_config": a.pgie_config,
        "tracker_config": a.tracker_config,
        "mux": "legacy nvstreammux",
        "mux_wxh": "%dx%d" % (a.mux_width, a.mux_height),
        "mux_timeout_us": a.mux_timeout_us,
        "warmup_s": a.warmup,
        "measure_s": a.measure,
        "class_filter": sorted(keep),
        "source_bin": a.source_bin,
        "engine_note": a.engine_note,
    }
    write_result(a.out, res)
    return code


if __name__ == "__main__":
    sys.exit(main())
