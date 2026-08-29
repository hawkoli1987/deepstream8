import json, numpy as np
d = json.load(open("out/motion_scores.json"))
keep = d["video"]["kept_source_frames"]
print("F =", d["video"]["frame_count"], " fps =", d["video"]["fps"])
for a in d["algorithms"]:
    n = np.array([x if x is not None else np.nan for x in a["norm"]])
    mot = [i for i, x in enumerate(a["motion"]) if x]
    src = [keep[i] for i in mot]
    print(f"\n{a['id']:12s} thr={a['threshold']:.3f}")
    print(f"  motion idx (deduped): {mot}")
    print(f"  motion idx (orig120): {src}")
    fin = n[np.isfinite(n)]
    print(f"  norm  min/med/p90/max = {fin.min():.3f}/{np.median(fin):.3f}/"
          f"{np.percentile(fin,90):.3f}/{fin.max():.3f}")
    # static-region ripple: std of norm over non-motion valid frames
    valid = np.array(a["valid"])
    nm = np.array([bool(x) for x in a["motion"]])
    stat = n[valid & ~nm]
    print(f"  static-frame norm: mean={np.nanmean(stat):.3f} std={np.nanstd(stat):.3f} "
          f"max={np.nanmax(stat):.3f}")
