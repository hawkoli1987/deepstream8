#!/usr/bin/env bash
# build_nvof.sh — compile the NvOF score extractor in the DeepStream container.
# Reuses the DS8 SDK headers shipped in the image; links the DS metadata libs.
set -euo pipefail
DS=/opt/nvidia/deepstream/deepstream-8.0
BIN=/work/nvof_extract
SRC=/work/src/nvof_extract.c

gcc -O2 -o "$BIN" "$SRC" \
  -I"$DS/sources/includes" \
  -I"$DS/sources/apps/apps-common/src" \
  $(pkg-config --cflags --libs gstreamer-1.0 gobject-2.0 glib-2.0) \
  -L"$DS/lib" \
  -Wl,-rpath,"$DS/lib" \
  -lnvdsgst_meta -lnvds_meta

echo "built $BIN"