#!/usr/bin/env bash
# build_opencv_cuda.sh — OpenCV 4.10.0, headless + CUDA (sm_75 only) + python3
# bindings, for the Farnebäck lane. Pinned version + arch + trimmed BUILD_LIST
# → reproducible; ~20-40 min on 8 vCPU.
#
# conda-forge ships no CUDA OpenCV (checked 2026-08), so a source build is the
# only route to cv2.cuda.FarnebackOpticalFlow.
set -euo pipefail
OCV_VER=4.10.0
PREFIX=/work/opencv
BUILD=/work/build/checkout
PYV=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')

# ---- deps (idempotent) --------------------------------------------------
if ! command -v cmake >/dev/null || ! command -v ninja >/dev/null || ! command -v unzip >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends cmake ninja-build build-essential python3-dev unzip
fi

# ---- sources ------------------------------------------------------------
mkdir -p "$BUILD"
cd "$BUILD"
[ -d opencv-$OCV_VER ] || { curl -fsSL -o o.zip https://github.com/opencv/opencv/archive/refs/tags/$OCV_VER.zip; unzip -q o.zip; }
[ -d opencv_contrib-$OCV_VER ] || { curl -fsSL -o c.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/$OCV_VER.zip; unzip -q c.zip; }
mkdir -p "$PREFIX"

# ---- configure ----------------------------------------------------------
cmake -S opencv-$OCV_VER -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PREFIX \
  -DPYTHON3_EXECUTABLE=$(which python3) \
  -DPYTHON3_PACKAGES_PATH=$PREFIX/lib/python$PYV/site-packages \
  -DOPENCV_EXTRA_MODULES_PATH=$PWD/opencv_contrib-$OCV_VER/modules \
  -DBUILD_LIST=core,imgproc,cudev,cudaarithm,cudaimgproc,cudafilters,cudawarping,cudaoptflow,python3,ts \
  -DBUILD_opencv_python3=ON \
  -DWITH_CUDA=ON -DCUDA_ARCH_BIN=7.5 \
  -DWITH_CUBLAS=ON \
  -DWITH_NVCUVID=OFF -DWITH_NVCUVENC=OFF \
  -DWITH_IPP=OFF -DWITH_OPENCL=OFF -DWITH_ITT=OFF -DWITH_QUIRC=OFF \
  -DWITH_QT=OFF -DWITH_GTK=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF \
  -DWITH_V4L=OFF -DWITH_1394=OFF -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENEXR=OFF \
  -DWITH_PNG=OFF -DWITH_JPEG=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF

# ---- fail-fast gate: bindings drop silently if python3-dev/numpy missing --
echo "--- configure gate ---"
grep -E '^BUILD_opencv_python3:BOOL=ON'        build/CMakeCache.txt
grep -E '^PYTHON3_INCLUDE_DIR:PATH=.'          build/CMakeCache.txt
grep -E '^PYTHON3_NUMPY_INCLUDE_DIRS:PATH=.'   build/CMakeCache.txt
grep -E '^CUDA_ARCH_BIN:STRING=7.5'            build/CMakeCache.txt
grep -E '^WITH_CUDA:BOOL=ON'                   build/CMakeCache.txt

# ---- build + install ----------------------------------------------------
cmake --build build -j"$(nproc)"
cmake --install build

# ---- smoke test ---------------------------------------------------------
export PYTHONPATH=$PREFIX/lib/python$PYV/site-packages
export LD_LIBRARY_PATH=$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python3 - <<'PY'
import cv2, numpy as np
n = cv2.cuda.getCudaEnabledDeviceCount()
assert n == 1, f"expected 1 CUDA device, got {n}\n{cv2.getBuildInformation()}"
a = cv2.cuda.GpuMat(); a.upload(np.zeros((1384, 1472), np.uint8))
far = cv2.cuda.FarnebackOpticalFlow.create(numLevels=5, pyrScale=0.5, fastPyramids=False,
                                           winSize=21, numIters=3, polyN=5, polySigma=1.1, flags=0)
f = far.calc(a, a, None)
assert f.size() == (1472, 1384) and f.type() == cv2.CV_32FC2, f.size()
print("cv2", cv2.__version__, "| cuda devices:", n, "| farneback OK")
PY

echo ""
echo "smoke test OK — every later step exports:"
echo "  export PYTHONPATH=$PREFIX/lib/python$PYV/site-packages LD_LIBRARY_PATH=$PREFIX/lib"