# Motion-detection algorithm comparison — `sample_cam6.mp4`

Five motion-detection algorithms scored frame-by-frame over one fixed-camera
parking-deck clip, with a single-page interactive comparison plus a T4
latency/occupancy benchmark.

## The deliverable

**`motion-compare.html`** — self-contained (video + data inlined), ~15 MB.
Open it in a browser. Scrub the one timeline; the video, five score plots and
five red/green swimlanes all move under one playhead. Toggle a per-algorithm
overlay on the video. Space / click = play-pause, arrows = step, drag anywhere
on the timeline = scrub. A performance section compares latency (e2e and
compute-only ms/frame) and SM occupancy across all five.

## The five algorithms

| id | what | device |
|----|------|--------|
| `cpu_absdiff` | greyscale `|Y_t − Y_{t−1}|`, mean over the fisheye disc / 255 | CPU, numpy |
| `gpu_absdiff` | the identical calculation in CuPy | GPU, Tesla T4 |
| `nvof` | NVIDIA Optical Flow Accelerator (`nvof` DeepStream plugin, SLOW preset, 4×4 grid); mean vector length / frame diagonal | GPU, Tesla T4 NvOFA |
| `mog2` | `cv2.createBackgroundSubtractorMOG2` foreground fraction (shadows excluded, 10-frame warm-up) | CPU, OpenCV |
| `farneback` | OpenCV `cv2.cuda.FarnebackOpticalFlow` dense flow (5-level pyramid, winSize 21); mean |flow| / frame diagonal | GPU, Tesla T4 CUDA cores |

## Pipeline (all ran in the DeepStream 8.0 container on a Brev T4)

```
sample_cam6.mp4  (1472×1384, 120 frames)
  │  ffmpeg  -x264-params bframes=0:keyint=1   →  all-intra, B-frame-free
  ▼
cam6_ai.mp4
  │  dedup.py        detect the broken 3-frame cadence (every 3rd frame is a
  │                  byte-duplicate) → keep 80 unique frames
  ▼
cam6d.yuv (80 frames, luma read by every algorithm)   +   cam6d_*.mp4
  │
  ├─ extract_cv.py     absdiff CPU, absdiff GPU (CuPy), MOG2  → cv_scores.npz
  ├─ nvof_extract.c    pad-probe on nvof src pad             → of_mv.raw + of_index.csv
  ├─ extract_farneback.py  OpenCV CUDA Farnebäck              → fb_scores.npz
  ▼
build_json.py   percentile normalisation (1–99, per clip; CPU+GPU share
                constants), Otsu thresholds, motion/static labels,
                pooled 46×43 overlay heat grids, per-algorithm
                Calculation/Normalization/Threshold descriptions,
                perf block  → motion_scores.json
  ▼
build_html.py   inline cam6d.mp4 (base64) + motion_scores.json into template.html
  ▼
motion-compare.html
```

## Key results

- **CPU and GPU absdiff are bit-identical** — the difference images match to the
  last bit, Δmax = 0.0 on the scalar scores. (GPU compute ~0.06 s vs CPU ~1.3 s,
  but host↔device transfer makes it a wash end-to-end at 80 frames.)
- All five algorithms detect the **same two vehicle-transit events** (deduped
  frames ~32–35 and ~68–71). They differ at the margins: MOG2 leads (background
  model reacts on arrival), NvOFA trails and reads the 2nd, smaller vehicle as
  weaker motion; absdiff has the noisiest static baseline.
- **NvOFA ≫ CUDA-core Farnebäck on throughput** — the dedicated optical-flow
  block finishes the dense-flow job an order of magnitude faster than the
  software equivalent on the SMs (numbers in the HTML perf section); absdiff on
  CuPy is the cheapest GPU work in the set. SM-busy% and Nsight Compute kernel
  detail are in the perf section.
- `sample_cam6.mp4` has a broken authoring cadence (documented above) and a
  ~2.6 grey-level global shimmer between its true frames — a data-quality issue
  in the clip, handled by de-duplication + per-clip percentile normalisation.

## Files

- `motion-compare.html` — the deliverable
- `template.html` — the page without the inlined assets (`__VIDEO__`, `__DATA__`)
- `cam6d.mp4`, `motion_scores.json` — the inlined assets, standalone
- `perf/` — benchmark provenance (`bench.json`, `sm.csv`, `perf.json`, logs) — local only, gitignored
- `src/` — the pipeline scripts; `src/diag/` — throwaway diagnostics
- `qa/` — verification frames (motion frames, difference images, ROI mask, NvOF fields) — local only, gitignored
- `container-workspace.inspect.json` — the Brev container's original `docker run` config

## Reproducing the benchmark

```bash
# in the DeepStream container (see src/build_opencv_cuda.sh for the 5th lane):
bash src/build_nvof.sh                                   # nvof score extractor (C)
python3 src/extract_cv.py cam6d.yuv 1472 1384 80 out     # cpu/gpu absdiff + MOG2 (+ parity)
python3 src/extract_farneback.py cam6d.yuv 1472 1384 80 out
bash src/run_bench.sh                                    # SM poller + bench.py + ncu + collect_perf
python3 src/build_json.py out                            # fuse → motion_scores.json (+ perf)
python3 src/build_html.py                                # local: inline assets → the deliverable
```
