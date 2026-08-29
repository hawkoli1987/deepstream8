import numpy as np, csv, cv2
idx = list(csv.DictReader(open("out/of_index.csv")))
mv = np.fromfile("out/of_mv.raw", dtype=np.int16)


def field(k):
    r = idx[k]
    o = int(r["byte_offset"]) // 2
    n = int(r["byte_len"]) // 2
    g = mv[o:o + n].reshape(int(r["rows"]), int(r["cols"]), 2).astype(float) / 32.0
    return r["frame_num"], g


for k in (0, 1, 2, 3):
    fn, g = field(k)
    fx, fy = g[..., 0], g[..., 1]
    m = np.hypot(fx, fy)
    print(f"frame {fn}: mean_mag={m.mean():.3f}  mean_fx={fx.mean():+.3f}  "
          f"mean_fy={fy.mean():+.3f}  median_mag={np.median(m):.3f}")
    hsv = np.zeros((*m.shape, 3), np.uint8)
    ang = (np.arctan2(fy, fx) + np.pi) * 90 / np.pi
    hsv[..., 0] = ang.astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(m * 12, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f"qa/of_field_{fn}.png", cv2.resize(bgr, (460, 430), interpolation=cv2.INTER_NEAREST))
