# Motion-detection algorithm comparison — `sample_cam6.mp4`

Four motion-detection algorithms scored frame-by-frame over one fixed-camera
parking-deck clip, with a single-page interactive comparison.

## The deliverable

**`motion-compare.html`** — self-contained (video + data inlined), ~14 MB.
Open it in a browser. Scrub the one timeline; the video, four score plots and
four red/green swimlanes all move under one playhead. Toggle a per-algorithm
overlay on the video. Space / click = play-pause, arrows = step, drag anywhere
on the timeline = scrub.

## The four algorithms

| id | what | device |
|----|------|--------|
| `cpu_absdiff` | greyscale `|Y_t − Y_{t−1}|`, mean over the fisheye disc / 255 | CPU, numpy |
| `gpu_absdiff` | the identical calculation in CuPy | GPU, Tesla T4 |
| `nvof` | NVIDIA Optical Flow Accelerator (`nvof` DeepStream plugin, SLOW preset, 4×4 grid); mean vector length / frame diagonal | GPU, Tesla T4 NvOFA |
| `mog2` | `cv2.createBackgroundSubtractorMOG2` foreground fraction (shadows excluded, 10-frame warm-up) | CPU, OpenCV |

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
  ▼
build_json.py   percentile normalisation (1–99, per clip; CPU+GPU share
                constants), Otsu thresholds, motion/static labels,
                pooled 46×43 overlay heat grids  → motion_scores.json
  ▼
build_html.py   inline cam6d.mp4 (base64) + motion_scores.json into template.html
  ▼
motion-compare.html
```

## Key results

- **CPU and GPU absdiff are bit-identical** — the difference images match to the
  last bit, Δmax = 0.0 on the scalar scores. (GPU compute ~0.06 s vs CPU ~1.3 s,
  but host↔device transfer makes it a wash end-to-end at 80 frames.)
- All four algorithms detect the **same two vehicle-transit events** (deduped
  frames ~32–35 and ~68–71). They differ at the margins: MOG2 leads (background
  model reacts on arrival), NvOFA trails and reads the 2nd, smaller vehicle as
  weaker motion; absdiff has the noisiest static baseline.
- `sample_cam6.mp4` has a broken authoring cadence (documented above) and a
  ~2.6 grey-level global shimmer between its true frames — a data-quality issue
  in the clip, handled by de-duplication + per-clip percentile normalisation.

## Files

- `motion-compare.html` — the deliverable
- `template.html` — the page without the inlined assets (`__VIDEO__`, `__DATA__`)
- `cam6d.mp4`, `motion_scores.json` — the inlined assets, standalone
- `src/` — the pipeline scripts; `src/diag/` — throwaway diagnostics
- `qa/` — verification frames (motion frames, difference images, ROI mask, NvOF fields)
- `container-workspace.inspect.json` — the Brev container's original `docker run` config
