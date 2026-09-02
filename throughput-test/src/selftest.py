#!/usr/bin/env python3
"""selftest.py — Mac-local static checks for throughput-test (V0 gate).

Checks: py_compile every src/*.py, bash -n every src/*.sh, corruption-token
scan, template placeholder checks, generator + collector smoke tests with a
synthetic run dir. Exits non-zero on the first hard failure.
"""
import glob
import json
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)

PY_FILES = ["app_throughput.py", "sweep.py", "sm_poll.py", "collect.py",
            "build_html.py", "make_pgie_config.py", "make_tracker_yml.py",
            "selftest.py"]
SH_FILES = ["setup_env.sh", "remote.sh", "runner.sh"]
BAD_TOKENS = ["marker", "corrupted-tokens", "Tracer0", "throughwork",
              "get buffer\n", "no-more-pads", "placeholder-"]
TEMPLATE_TOKENS = {"pgie_yolo26m.txt.tmpl": ["@ONNX@", "@ENGINE@", "@BATCH@"],
                   "report.html": ["__DATA__"]}


def check(name, ok, detail=""):
    print("  %-42s %s%s" % (name, "OK" if ok else "FAIL",
                            (" — " + detail) if detail and not ok else ""))
    return ok


def main():
    fails = 0
    print("== python compile ==")
    for f in PY_FILES:
        p = os.path.join(HERE, f)
        if not os.path.isfile(p):
            fails += check("py %s" % f, False, "missing")
            continue
        try:
            py_compile.compile(p, doraise=True)
            fails += not check("py %s" % f, True)
        except py_compile.PyCompileError as e:
            fails += not check("py %s" % f, False, str(e)[:120])

    print("== bash syntax ==")
    for f in SH_FILES:
        p = os.path.join(HERE, f)
        try:
            r = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
            fails += not check("sh %s" % f, r.returncode == 0, r.stderr[:120])
        except Exception as e:
            fails += not check("sh %s" % f, False, str(e)[:120])

    print("== corruption-token scan ==")
    for f in PY_FILES + SH_FILES:
        if f == "selftest.py":
            continue  # selftest contains the token strings by design
        p = os.path.join(HERE, f)
        if not os.path.isfile(p):
            continue
        with open(p) as fh:
            txt = fh.read()
        hits = [t for t in BAD_TOKENS if t in txt]
        fails += not check("scan %s" % f, not hits, ",".join(hits))

    print("== templates ==")
    for fname, toks in TEMPLATE_TOKENS.items():
        p = os.path.join(WS, "src", "templates", fname)
        with open(p) as fh:
            txt = fh.read()
        for t in toks:
            fails += not check("tmpl %s has %s" % (fname, t), t in txt)

    print("== labels ==")
    with open(os.path.join(WS, "src/templates/coco80.txt")) as fh:
        names = [l for l in fh.read().splitlines() if l.strip()]
    ok = (len(names) == 80 and len(set(names)) == 80
          and names[0] == "person" and names[-1] == "toothbrush")
    fails += not check("coco80.txt canonical", ok, "lines=%d" % len(names))

    print("== smoke: generators + collector ==")
    tmp = tempfile.mkdtemp(prefix="throughput-selftest-")
    try:
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "make_pgie_config.py"),
             "--out", os.path.join(tmp, "pgie.txt"),
             "--onnx", "/x/yolo26m.onnx",
             "--engine", "/x/yolo26m_b10_gpu0_fp16.engine",
             "--batch", "10"],
            capture_output=True, text=True)
        ok = r.returncode == 0 and "batch-size=10" in open(os.path.join(tmp, "pgie.txt")).read()
        fails += not check("make_pgie_config render", ok, r.stderr[:120])

        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "make_tracker_yml.py"),
             "--out", os.path.join(tmp, "tracker.yml")],
            capture_output=True, text=True)
        ok = r.returncode == 0
        fails += not check("make_tracker_yml fallback", ok, r.stderr[:120])

        fake = os.path.join(tmp, "run")
        os.makedirs(os.path.join(fake, "N4"))
        with open(os.path.join(fake, "N4", "result.json"), "w") as f:
            json.dump({
                "schema": "throughput-test/v1", "N": 4, "status": "ok", "exit_code": 0,
                "counter": {"batches": 900, "frames": 3600, "dt_s": 30.0,
                            "ms_per_batch": 33.3,
                            "window": {"start_epoch": 1000.0, "end_epoch": 1030.0}},
                "fps_total": 120.0, "fps_per_stream": 30.0,
                "gpu": {"name": "Tesla T4", "driver": "595"},
                "config": {"engine_note": "A / engine=per-n",
                           "warmup_s": 20.0, "measure_s": 30.0},
            }, f)
        with open(os.path.join(fake, "sm.csv"), "w") as f:
            f.write("epoch,sm_pct,power_w,mem_mb\n")
            for i in range(995, 1036):
                f.write("%d.5,85,55.2,9000\n" % i)
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "collect.py"),
             "--run-dir", fake],
            capture_output=True, text=True)
        ok = r.returncode == 0
        if ok:
            with open(os.path.join(fake, "summary.csv")) as f:
                body = f.read()
            ok = "4,ok,120.0" in body and "85.0" in body
        fails += not check("collect.py synthetic run", ok, r.stderr[:160])

        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "build_html.py"),
             "--run-dir", fake,
             "--out", os.path.join(tmp, "report.html")],
            capture_output=True, text=True)
        ok = r.returncode == 0
        if ok:
            with open(os.path.join(tmp, "report.html")) as f:
                body = f.read()
            ok = "__DATA__" not in body and '"points"' in body
        fails += not check("build_html.py token replace", ok, r.stderr[:160])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("== smoke: engine fingerprint helpers ==")
    sys.path.insert(0, HERE)
    import sweep  # noqa: E402
    ok = (sweep.decode_trt_banner("100900") == "10.9"
          and sweep.decode_trt_banner("") is None)
    fails += not check("decode_trt_banner", ok)
    fp = "%s_%s" % (sweep.gpu_sm_tag(), sweep.trt_tag())
    key = "y26m_b16_%s_fp16.engine" % fp
    ok = ok and "/" not in fp and " " not in fp
    fails += not check("engine cache key sane (%s)" % key, ok)

    print("")
    if fails:
        print("SELFTEST FAIL: %d check(s) failed" % fails)
        return 1
    print("SELFTEST OK (all checks passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())