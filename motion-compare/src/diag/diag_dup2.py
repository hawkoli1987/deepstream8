import numpy as np, subprocess, os
W, H, F = 1472, 1384, 120
fb = W * H * 3 // 2
subprocess.run(["ffmpeg", "-v", "error", "-i", "sample_cam6.mp4",
                "-f", "rawvideo", "-pix_fmt", "yuv420p", "/tmp/orig.yuv"], check=True)
Y = np.memmap("/tmp/orig.yuv", dtype=np.uint8, mode="r").reshape(F, fb)[:, :W * H].reshape(F, H, W)
print("ORIGINAL sample_cam6.mp4 — frame-to-frame mean|dY|:")
dups = 0
for t in range(1, F):
    d = np.abs(Y[t].astype(np.int16) - Y[t - 1].astype(np.int16)).mean()
    tag = "  <-- DUP" if d < 0.05 else ""
    if d < 0.05:
        dups += 1
    if t < 22 or tag:
        print(f"  {t:3d}: {d:.4f}{tag}")
print(f"total near-duplicate transitions: {dups}/{F-1}")
os.remove("/tmp/orig.yuv")
