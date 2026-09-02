#!/usr/bin/env python3
"""sm_poll.py — 1 Hz nvidia-smi poller -> CSV (epoch, sm_pct, power_w, mem_mb).

Exits on SIGTERM/SIGINT, flushing every row. One poller spans the whole sweep;
collect.py epoch-filters this file per-N to derive sm_pct_mean / power_w_mean /
mem_mb_max per N.
"""
import argparse
import csv
import signal
import subprocess
import time

FIELDS = ("utilization.gpu", "power.draw", "memory.used")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="sm.csv path")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    running = True

    def _stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    cmd = ["nvidia-smi", "--query-gpu=" + ",".join(FIELDS),
           "--format=csv,noheader,nounits"]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "sm_pct", "power_w", "mem_mb"])
        next_t = time.time()
        while running:
            try:
                out = subprocess.check_output(cmd, timeout=5).decode()
                row = [x.strip() for x in out.splitlines()[0].split(",")]
                row = [x for x in row if x and x != "N/A"]
                if len(row) == 3:
                    w.writerow([round(time.time(), 3)] + row)
                    f.flush()
            except Exception as exc:
                print("poll error: %s" % exc, flush=True)
            next_t += args.interval
            d = next_t - time.time()
            if d > 0:
                time.sleep(d)
            else:
                next_t = time.time()


if __name__ == "__main__":
    main()




