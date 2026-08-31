#!/usr/bin/env bash
# run_bench.sh — the whole Phase E in one detached command:
#   bench_sm (1 Hz poller) -> bench.py (5 algos + decode baseline)
#   -> best-effort ncu profiles -> collect_perf.py -> perf.json
set -euo pipefail
export PYTHONPATH=/work/opencv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=/work/opencv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd /work

# clip under test (v1 cam6 defaults; override for other clips, e.g. v2:
#   YUV=/work/clip2.yuv MP4=/work/clip2.mp4 W=1280 H=720 F=<n> OUT=out2 bash src/run_bench.sh)
YUV=${YUV:-/work/cam6d.yuv}
MP4=${MP4:-/work/cam6d.mp4}
W=${W:-1472}; H=${H:-1384}; F=${F:-80}
OUT=${OUT:-out}
export YUV W H F OUT

python3 src/bench_sm.py "$OUT/sm.csv" &
SM=$!
trap 'kill $SM 2>/dev/null || true' EXIT

python3 src/bench.py "$YUV" "$W" "$H" "$F" "$OUT" --min-window 30 --video "$MP4"

# ---- ncu, best effort ----------------------------------------------------
if command -v ncu >/dev/null 2>&1 || apt-get install -y -qq nsight-compute 2>/dev/null; then
  echo "--- ncu: gpu_absdiff (cupy) ---"
  ncu --target-processes all --csv --log-file out/ncu_gpu.csv \
      --launch-skip 8 --launch-count 24 \
      --section SpeedOfLight --section Occupancy \
      python3 -c "
import os, numpy as np, cupy as cp
YUV, W, H, F, OUT = os.environ['YUV'], int(os.environ['W']), int(os.environ['H']), int(os.environ['F']), os.environ['OUT']
buf = np.memmap(YUV, dtype=np.uint8, mode='r')
Y = buf.reshape(F, W*H*3//2)[:, :W*H].reshape(F, H, W)
roi = np.load(f'{OUT}/cv_scores.npz')['roi_mask']
rg = cp.asarray(roi)
dev = [cp.asarray(Y[t]).astype(cp.int16) for t in range(F)]
for t in range(1, F//2):
    dg = cp.abs(dev[t] - dev[t-1])
    _ = float(dg[rg].astype(cp.float64).mean())
" 2>&1 | tail -2 || true
  echo "--- ncu: farneback ---"
  mkdir -p out/ncu_out && cp out/cv_scores.npz out/ncu_out/
  ncu --target-processes all --csv --log-file "$OUT/ncu_far.csv" \
      --launch-skip 8 --launch-count 24 \
      --section SpeedOfLight --section Occupancy \
      python3 src/extract_farneback.py "$YUV" "$W" "$H" "$F" "$OUT/ncu_out" 2>&1 | tail -2 || true
  echo "--- ncu: nvof pipeline (Gst API — parse-launch can't link the mux) ---"
  ncu --target-processes all --csv --log-file "$OUT/ncu_nvof.csv" \
      --launch-skip 8 --launch-count 20 \
      --section SpeedOfLight --section Occupancy \
      python3 -c "import sys, os; sys.path.insert(0,'/work/src'); from bench import gst_run; gst_run('nvof', os.environ['MP4'], int(os.environ['W']), int(os.environ['H']))" 2>&1 | tail -2 || true
else
  echo "ncu unavailable — SM% only" | tee "$OUT/ncu_note.txt"
fi

python3 src/collect_perf.py "$OUT"
echo "run_bench done"
