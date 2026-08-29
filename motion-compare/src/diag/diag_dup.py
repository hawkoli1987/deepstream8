import numpy as np
W, H, F = 1472, 1384, 120
fb = W * H * 3 // 2
Y = np.memmap("cam6.yuv", dtype=np.uint8, mode="r").reshape(F, fb)[:, :W * H].reshape(F, H, W)
print("frame-to-frame mean|dY| (first 20):")
for t in range(1, 20):
    d = np.abs(Y[t].astype(np.int16) - Y[t - 1].astype(np.int16)).mean()
    print(f"  {t:3d} vs {t-1:3d}:  {d:.4f}   {'== DUP' if d < 0.01 else ''}")
# also from the ORIGINAL (not the re-encode), via a fresh raw
