# throughput-test

DeepStream 8 "Nullwriter" perception throughput harness: N × `nvurisrcbin` →
`nvstreammux` → `nvinfer` (YOLO26m, COCO-80) → vehicle-class filter → `nvtracker`
(NvDCF) → `fakesink`. Nothing is rendered, encoded, or emitted — the pipeline
exists to answer one question: **how does realtime perception throughput scale
with the number of streams N on a given GPU?**

House rule: `batch_size == num_streams` — mux batch-size = nvinfer batch-size =
N (sweep overrides the standalone app's hardcoded default of 10).

## Layout

```
src/
├── app_throughput.py   the pipeline app: one N, warmup+measure window, result.json
├── sweep.py            N-list driver: engine prebuild, per-N isolation, failure.json
├── sm_poll.py          1 Hz nvidia-smi poller (whole sweep -> sm.csv)
├── collect.py          merge results + sm.csv -> summary.{csv,json}
├── build_html.py       inline summary.json -> throughput-test-v1.html
├── make_pgie_config.py per-N nvinfer config from src/templates/pgie_yolo26m.txt.tmpl
└── make_tracker_yml.py NvDCF perf yml (patched container preset, locked keys)
src/parser/             NvDsInferParseYolo26E2e (Plan A) + vendored raw-head parser (Plan B)
src/templates/          pgie tmpl / tracker tmpl / report.html (__DATA__)
src/setup_env.sh        in-container idempotent bring-up (pyds, ONNX, parser, tracker)
src/remote.sh + runner.sh  Mac-side: brev lifecycle, staging, sweep, fetch, html, stop
src/selftest.py         Mac-local static gate (py_compile, bash -n, smokes)
runs/                   gitignored — all measurement output lands here
```

## Run

On the Mac (thin orchestrator; everything executes remote):

```bash
cd src && DRY_RUN=1 ./runner.sh                 # plan only, no instance work
DS_TAG=t4-baseline ./runner.sh                  # full run: T4 up -> sweep -> stop
DS_TAG=t4-fast DS_N_LIST=1,4,16,64 KEEP_UP=1 ./runner.sh   # short list, stay up
DS_INSTANCE=<other> DS_HOST_ALIAS=<other>-host DS_TAG=<tag> ./runner.sh  # other GPU
```

Sweep defaults: `N = 1,2,4,8,16,24,32,48,64,96,128,160`, warmup 20 s +
measure 30 s per N, per-N static engines via trtexec (shared timing cache).
Engine filenames are fingerprinted with GPU compute capability + TensorRT
version (`yolo26m_b16_sm75_trt10.9_fp16.engine`) so a cached engine is only
reused on matching hardware/TRT — each GPU type builds its set exactly once.
`failure.json` per failed N (VRAM exhaustion at high N is an expected,
captured datapoint — the sweep continues). `--resume` semantics: sweep skips
Ns that already have `result.json`.

## What each N run measures

- `fps_total`, `fps_per_stream`, `ms_per_batch` — fakesink batch counter over
  the 30 s steady-state window (each mux batch = N frames), epochs recorded so
  `collect.py` can intersect the 1 Hz `nvidia-smi` trace per N.
- `sm_pct_mean`, `power_w_mean`, `mem_mb_max` — from that trace.
- `probe` — class-filter sample: kept (remapped to class 0 "VOI") vs removed
  object metas, tallied only during the first 5 s.
- `engine_build_s` — trtexec build time, never inside the timed window.

## Method notes

- Pipeline built via the **Gst API** (DS8 parse-launch refuses
  decoder→nvstreammux links); mux sink pads requested eagerly (`sink_%d`).
  Legacy nvstreammux — never set `USE_NEW_NVSTREAMMUX=yes`.
- YOLO26 is NMS-free end-to-end, so nvinfer `cluster-mode=4` + the workstream's
  own `NvDsInferParseYolo26E2e` parser (Plan A). If the ONNX gate finds a
  static batch dim, setup re-exports a raw-head ONNX and falls back to the
  stock `NvDsInferParseYoloV11` + `cluster-mode=2` (Plan B, recorded).
- Sources are the in-container `sample_720p.mp4` with `file-loop=1`,
  `num-extra-surfaces=32` per source. `--uri` (app) cycles to fill N; RTSP
  URIs work unchanged (set `live-source=1` for real cameras).
- pyds on Python 3.12 uses the `--system-site-packages` venv + `numpy<2`
  ladder; the chosen interpreter is recorded to `/work/throughput/env/python.txt`.
