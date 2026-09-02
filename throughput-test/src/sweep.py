#!/usr/bin/env python3
"""sweep.py — in-container driver: N-list sweep with per-N isolation.

For each N: ensure engine (trtexec, shared timing cache), render the per-N
pgie config, run app_throughput.py in an isolated subprocess with a timeout;
on failure write failure.json and continue. One 1 Hz sm_poll.py spans the
whole sweep; collect.py later epoch-filters it per-N.

Artifacts expected under --work (created by setup_env.sh):
    $WORK/onnx/yolo26m.onnx        source ONNX (e2e export)
    $WORK/env/onnx_io.json         input name + Plan A/B record
    $WORK/cfg/labels.txt           COCO-80 labels
    $WORK/cfg/tracker.yml          NvDCF ll-config
    $WORK/lib/*.so                 custom parser lib
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "app_throughput.py")
MAKE_CFG = os.path.join(HERE, "make_pgie_config.py")

CAT = {
    0: "ok",
    1: "exception",
    2: "noplay",
    3: "gst_error",
    4: "no_buffers",
    124: "timeout",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-list", default="1,2,4,8,16,24,32,48,64,96,128,160")
    ap.add_argument("--warmup", type=float, default=20.0)
    ap.add_argument("--measure", type=float, default=30.0)
    ap.add_argument("--app-timeout", type=float, default=None,
                    help="per-N subprocess timeout; default warmup+measure+420s")
    ap.add_argument("--uri", action="append", default=None)
    ap.add_argument("--engine-mode", choices=["per-n", "single"], default="per-n")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-runtime-sec", type=float, default=None)
    ap.add_argument("--work", default="/work/throughput")
    ap.add_argument("--tracker-config", default="/work/throughput/cfg/tracker.yml")
    ap.add_argument("--source-bin", choices=["nvurisrcbin", "custom"], default="nvurisrcbin")
    ap.add_argument("--sm-interval", type=float, default=1.0)
    return ap.parse_args()


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def check_output(cmd, timeout=None):
    """subprocess.run wrapper returning (rc, stdout)."""
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "").strip()


def vram_mb():
    rc, out = check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    if rc != 0 or not out:
        return None
    return int(out.splitlines()[0].strip())


def gpu_sm_tag():
    """GPU compute-capability tag for engine-cache keys, e.g. 'sm75'."""
    try:
        rc, out = check_output(
            ["nvidia-smi", "--query-gpu=compute_cap",
             "--format=csv,noheader,nounits"])
        if rc == 0 and out:
            major, minor = out.splitlines()[0].strip().split(".")
            return "sm%s%s" % (major, minor)
    except Exception:
        pass
    return "gpu0"


def decode_trt_banner(tok):
    """'100900' -> '10.9' (TRT banner encoding MMmmpp)."""
    if not tok or len(tok) < 4:
        return None
    return "%d.%d" % (int(tok[:2]), int(tok[2:4]))


def trt_tag():
    """TensorRT version tag for engine-cache keys, e.g. 'trt10.9'."""
    try:
        import tensorrt
        return "trt%s" % str(tensorrt.__version__)
    except Exception:
        pass
    try:
        p = subprocess.run(["trtexec", "--help"], capture_output=True,
                           text=True, timeout=60)
        m = re.search(r"v(\d{4,6})", (p.stdout or "") + (p.stderr or ""))
        if m:
            v = decode_trt_banner(m.group(1))
            if v:
                return "trt%s" % v
    except Exception:
        pass
    return "trt-unknown"


def engine_fp():
    """Engine-cache fingerprint: arch + TRT (engines are not portable)."""
    return "%s_%s" % (gpu_sm_tag(), trt_tag())


def ensure_engine(a, n, onnx, io_info, run_dir, fp):
    """Return (engine_path, build_seconds). Builds via trtexec if absent.

    The engine name embeds fp (arch+TRT) so a cache copied across machine
    types, or a TRT upgrade, can never silently reuse an incompatible engine.
    """
    stem = os.path.splitext(os.path.basename(onnx))[0]
    mdir = os.path.join(a.work, "models")
    os.makedirs(mdir, exist_ok=True)
    if a.engine_mode == "single":
        eng = os.path.join(mdir, "%s_bflex_%s_fp16.engine" % (stem, fp))
    else:
        eng = os.path.join(mdir, "%s_b%d_%s_fp16.engine" % (stem, n, fp))
    if os.path.isfile(eng):
        return eng, 0.0
    inp = (io_info or {}).get("input_name") or "images"
    min_n = 1
    opt_n = 160 if a.engine_mode == "single" else n
    max_n = 160 if a.engine_mode == "single" else n
    cmd = [
        "trtexec",
        "--onnx=" + onnx,
        "--saveEngine=" + eng,
        "--minShapes=%s:%dx3x640x640" % (inp, min_n),
        "--optShapes=%s:%dx3x640x640" % (inp, opt_n),
        "--maxShapes=%s:%dx3x640x640" % (inp, max_n),
        "--fp16",
        "--skipInference",
        "--memPoolSize=workspace:16384M",
        "--timingCacheFile=" + os.path.join(mdir, "timing_%s.cache" % fp),
    ]
    os.makedirs(run_dir, exist_ok=True)
    elog = os.path.join(run_dir, "engine_b%d.log" % n)
    log("engine build N=%d -> %s" % (n, os.path.basename(eng)))
    t0 = time.time()
    with open(elog, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if rc != 0 or not os.path.isfile(eng):
        raise RuntimeError("trtexec failed rc=%d (log %s)" % (rc, elog))
    return eng, dt


def render_pgie(a, n, onnx, engine, run_dir, io_info):
    """Render cfg/pgie_b<N>.txt; Plan A -> Yolo26E2e parser + cluster-mode 4,
    Plan B -> stock YoloV11 parser + cluster-mode 2."""
    cfg_dir = os.path.join(run_dir, "cfg")
    os.makedirs(cfg_dir, exist_ok=True)
    out = os.path.join(cfg_dir, "pgie_b%d.txt" % n)
    if (io_info or {}).get("plan") == "B":
        parse_func = "NvDsInferParseYoloV11"
        cluster_mode = 2
        parser_lib = os.path.join(a.work, "lib", "libnvdsinfer_custom_impl_Yolo.so")
    else:
        parse_func = "NvDsInferParseYolo26E2e"
        cluster_mode = 4
        parser_lib = os.path.join(a.work, "lib", "libnvdsinfer_custom_impl_Yolo26E2e.so")
    rc = subprocess.call([
        sys.executable, MAKE_CFG,
        "--out", out,
        "--onnx", onnx,
        "--engine", engine,
        "--batch", str(n),
        "--parse-func", parse_func,
        "--parser-lib", parser_lib,
        "--cluster-mode", str(cluster_mode),
    ])
    if rc != 0:
        raise RuntimeError("make_pgie_config failed rc=%d" % rc)
    return out


def run_app(a, n, cfg, run_dir, note):
    """Isolated app subprocess; returns (exit_code, result_dict_or_None)."""
    ndir = os.path.join(run_dir, "N%d" % n)
    os.makedirs(ndir, exist_ok=True)
    logp = os.path.join(ndir, "app.log")
    outp = os.path.join(ndir, "result.json")
    cmd = [
        sys.executable, APP,
        "--n", str(n),
        "--pgie-config", cfg,
        "--tracker-config", a.tracker_config,
        "--warmup", str(a.warmup),
        "--measure", str(a.measure),
        "--source-bin", a.source_bin,
        "--engine-note", note,
        "--out", outp,
        "--log", logp,
    ]
    if a.uri:
        for u in a.uri:
            cmd += ["--uri", u]
    timeout = a.app_timeout or (a.warmup + a.measure + 420.0)
    t0 = time.time()
    rc = 0
    try:
        with open(logp, "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired:
        rc = 124
        log("TIMEOUT N=%d after %.0fs" % (n, timeout))
    dt = time.time() - t0
    log("app N=%d rc=%d in %.0fs" % (n, rc, dt))
    res = None
    if os.path.isfile(outp):
        try:
            with open(outp) as f:
                res = json.load(f)
        except Exception:
            pass
    return rc, res


def write_failure(run_dir, n, rc, extra=None):
    """failure.json: status, exit code, VRAM after, app.log tail."""
    ndir = os.path.join(run_dir, "N%d" % n)
    os.makedirs(ndir, exist_ok=True)
    tail = []
    lp = os.path.join(ndir, "app.log")
    if os.path.isfile(lp):
        with open(lp) as f:
            tail = f.read().splitlines()[-40:]
    payload = {
        "N": n,
        "status": CAT.get(rc, "rc%d" % rc),
        "exit_code": rc,
        "vram_mb_after": vram_mb(),
        "stderr_tail": tail,
    }
    if extra:
        payload.update(extra)
    with open(os.path.join(ndir, "failure.json"), "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    log("FAIL N=%d status=%s" % (n, payload["status"]))


def main():
    a = parse_args()
    run_dir = os.path.normpath(os.path.join(HERE, "..", "runs", a.tag))
    ns = [int(x) for x in a.n_list.split(",") if x.strip()]
    if a.dry_run:
        print("dry-run tag=%s run_dir=%s" % (a.tag, run_dir))
        for n in ns:
            eng = "bflex" if a.engine_mode == "single" else "b%d" % n
            print("  N=%-4d engine=%s cfg=cfg/pgie_b%d.txt app_timeout=%s" % (
                n, eng, n, a.app_timeout or (a.warmup + a.measure + 420.0)))
        return 0

    os.makedirs(run_dir, exist_ok=True)
    onnx = os.path.join(a.work, "onnx", "yolo26m.onnx")
    io_path = os.path.join(a.work, "env", "onnx_io.json")
    missing = [p for p in (onnx, io_path, a.tracker_config) if not os.path.isfile(p)]
    if missing:
        print("ERROR missing setup artifacts: %s" % missing, file=sys.stderr)
        return 2
    with open(io_path) as f:
        io_info = json.load(f)

    fp = engine_fp()
    log("engine cache fingerprint: %s" % fp)

    poller = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "sm_poll.py"),
         "--out", os.path.join(run_dir, "sm.csv"),
         "--interval", str(a.sm_interval)])
    log("sm poller pid=%d -> %s" % (poller.pid, os.path.join(run_dir, "sm.csv")))

    deadline = (time.time() + a.max_runtime_sec) if a.max_runtime_sec else None
    note = "%s / engine=%s" % (
        (io_info or {}).get("plan", "A"), a.engine_mode)
    ok_count = 0
    fail_count = 0
    for n in ns:
        if deadline and time.time() > deadline:
            log("max runtime reached; stop before N=%d" % n)
            break
        if a.resume and os.path.isfile(os.path.join(run_dir, "N%d" % n, "result.json")):
            log("resume: skip N=%d (result.json exists)" % n)
            continue
        log("=== N=%d ===" % n)
        try:
            engine, build_s = ensure_engine(a, n, onnx, io_info, run_dir, fp)
            cfg = render_pgie(a, n, onnx, engine, run_dir, io_info)
            rc, res = run_app(a, n, cfg, run_dir, note)
            if rc == 0 and res is not None:
                res["engine"] = engine
                res["engine_build_s"] = build_s
                with open(os.path.join(run_dir, "N%d" % n, "result.json"), "w") as f:
                    json.dump(res, f, indent=2)
                    f.write("\n")
                ok_count += 1
                log("OK   N=%d fps_total=%s" % (n, res.get("fps_total")))
            else:
                if rc == 0:
                    rc = -1  # rc 0 but no result.json -> treat as failure
                write_failure(run_dir, n, rc)
                fail_count += 1
        except Exception as exc:
            write_failure(run_dir, n, -1,
                          extra={"status": "setup_error", "error": str(exc)})
            fail_count += 1

    poller.send_signal(signal.SIGTERM)
    try:
        poller.wait(timeout=10)
    except subprocess.TimeoutExpired:
        poller.kill()

    log("sweep done ok=%d fail=%d -> %s" % (ok_count, fail_count, run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
