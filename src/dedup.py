#!/usr/bin/env python3
"""
dedup.py — sample_cam6.mp4 ships with a broken 3-frame cadence (every 3rd frame
is a byte-near duplicate of its predecessor). Detect the duplicates from the Y
plane and emit the list of unique frame indices, so every stage downstream works
on a clean sequence.

  in : cam6.yuv  (yuv420p, all 120 frames, from the all-intra re-encode)
  out: <outdir>/keep.json   {"keep":[0,2,3,5,...], "fps_src":2.0, "n":80}
"""
import sys, json, os
import numpy as np

W, H, F = 1472, 1384, 120
DUP_THRESH = 0.08     # mean |dY| below this  ==  duplicate frame


def main():
    yuv, outdir = sys.argv[1], sys.argv[2]
    fb = W * H * 3 // 2
    Y = np.memmap(yuv, dtype=np.uint8, mode="r").reshape(F, fb)[:, :W * H].reshape(F, H, W)
    keep = [0]
    deltas = []
    for t in range(1, F):
        d = float(np.abs(Y[t].astype(np.int16) - Y[t - 1].astype(np.int16)).mean())
        deltas.append(d)
        if d >= DUP_THRESH:
            keep.append(t)
    info = dict(keep=keep, n=len(keep), fps_src=2.0,
               dropped=[t for t in range(F) if t not in set(keep)],
               transition_deltas=deltas)
    json.dump(info, open(os.path.join(outdir, "keep.json"), "w"))
    print(f"kept {len(keep)}/{F} unique frames; dropped {F - len(keep)}")
    print("first kept:", keep[:12])


if __name__ == "__main__":
    main()
