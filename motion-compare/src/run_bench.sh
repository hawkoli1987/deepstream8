#!/usr/bin/env bash
# run_bench.sh — the whole Phase E in one detached command:
#   bench_sm (1 Hz poller) -> bench.py (5 algos + decode baseline)
#   -> best-effort ncu profiles -> collect_perf.py -> perf.json
set -euo pipefail
export PYTHONPATH=/work/opencv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=/work/opencv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd /work

python3 src/bench_sm.py out/sm.csv &
SM=$!
trap 'kill $SM 2>/dev/null || true' EXIT

python3 src/bench.py /work/cam6d.yuv 1472 1384 80 out --min-window 30

# ---- ncu, best effort ----------------------------------------------------
if command -v ncu >/dev/null 2>&1 || apt-get install -y -qq nsight-compute 2>/dev/null; then
  echo "--- ncu: gpu_absdiff (cupy) ---"
  ncu --target-processes all --csv --log-file out/ncu_gpu.csv \
      --launch-skip 8 --launch-count 24 \
      --section SpeedOfLight --section Occupancy \
      python3 -c "
import numpy as np, cupy as cp
buf = np.memmap('/work/cam6d.yuv', dtype=np.uint8, mode='r')
Y = buf.reshape(80, 1472*1384*3//2)[:, :1472*1384].reshape(80, 1384, 1472)
roi = np.load('out/cv_scores.npz')['roi_mask']
rg = cp.asarray(roi)
dev = [cp.asarray(Y[t]).astype(cp.int16) for t in range(80)]
for t in range(1, 40):
    dg = cp.abs(dev[t] - dev[t-1])
    _ = float(dg[rg].astype(cp.float64).mean())
" 2>&1 | tail -2 || true
  echo "--- ncu: farneback ---"
  mkdir -p out/ncu_out && cp out/cv_scores.npz out/ncu_out/
  ncu --target-processes all --csv --log-file out/ncu_far.csv \
      --launch-skip 8 --launch-count 24 \
      --section SpeedOfLight --section Occupancy \
      python3 src/extract_farneback.py /work/cam6d.yuv 1472 1384 80 out/ncu_out 2>&1 | tail -2 || true
  echo "--- ncu: nvof pipeline (Gst API — parse-launch can't link the mux) ---"
  ncu --target-processes all --csv --log-file out/ncu_nvof.csv \
      --launch-skip 8 --launch-count 20 \
      --section SpeedOfLight --section Occupancy \
      python3 -c "import sys; sys.path.insert(0,'/work/src'); from bench import gst_run; gst_run('nvof','/work/cam6d.mp4',1472,1384)" 2>&1 | tail -2 || true
else
  echo "ncu unavailable — SM% only" | tee out/ncu_note.txt
fi

python3 src/collect_perf.py out
echo "run_bench done"
