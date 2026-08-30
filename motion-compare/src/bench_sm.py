#!/usr/bin/env python3
"""
bench_sm.py — 1 Hz SM/occupancy poller for the T4 benchmark window.

Runs until killed (SIGTERM). One CSV line per second, flushed immediately so
collect_perf.py can window-filter by each algorithm's t_start/t_end.

usage: bench_sm.py <out.csv>
"""
import subprocess, sys, time

out_path = sys.argv[1]
f = open(out_path, "w", buffering=1)
f.write("epoch,util_gpu,util_mem,power_w,clocks_sm\n")

while True:
    t = time.time()
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True)
        f.write(",".join([f"{t:.3f}"] + r.stdout.strip().splitlines()[0].split(", ")) + "\n")
        f.flush()
    except Exception as e:  # never die mid-window
        sys.stderr.write(f"sm poll: {e}\n")
    time.sleep(max(0.0, 1.0 - (time.time() - t)))