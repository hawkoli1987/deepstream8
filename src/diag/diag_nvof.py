import numpy as np, csv
idx = list(csv.DictReader(open("out/of_index.csv")))
mv = np.fromfile("out/of_mv.raw", dtype=np.int16)
for r in idx[:15]:
    o = int(r["byte_offset"]) // 2
    n = int(r["byte_len"]) // 2
    g = mv[o:o + n].reshape(int(r["rows"]), int(r["cols"]), 2).astype(float) / 32.0
    m = np.hypot(g[..., 0], g[..., 1])
    print(r["frame_num"], "mean=%.3f" % m.mean(), "p99=%.2f" % np.percentile(m, 99),
          "max=%.1f" % m.max(), "frac>0.1=%.3f" % (m > 0.1).mean())
