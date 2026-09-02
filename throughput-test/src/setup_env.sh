#!/usr/bin/env bash
# setup_env.sh — in-container idempotent bring-up for throughput-test.
# Runs INSIDE the deepstream container (root, no sudo needed). Idempotent:
# each step is skipped when its artifact already exists. Cached artifacts live
# under /work/throughput (bind-mounted, survives container recreation).
#
# Usage: bash setup_env.sh
set -uo pipefail

SRC="${SRC:-/work/throughput-test/src}"
WORK="${WORK:-/work/throughput}"
DS_HOME="${DS_HOME:-/opt/nvidia/deepstream/deepstream}"
APP_PY="${APP_PY:-}"          # resolved python recorded to $WORK/env/python.txt

log() { printf '[setup] %s\n' "$*" >&2; }
die() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }

log "SRC=$SRC WORK=$WORK"
[ -d "$SRC" ] || die "staged src not found at $SRC (stage the workstream first)"
mkdir -p "$WORK"/{onnx,env,cfg,lib,models}

# ---------------------------------------------------------------- 1. env record
ENV_DIR="$WORK/env"
mkdir -p "$ENV_DIR"
[ -f "$ENV_DIR/driver.txt" ] || nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$ENV_DIR/driver.txt" 2>&1
[ -f "$ENV_DIR/nvidia_smi_L.txt" ] || nvidia-smi -L > "$ENV_DIR/nvidia_smi_L.txt" 2>&1
python3 --version > "$ENV_DIR/python.txt" 2>&1
for p in nvstreammux nvinfer nvtracker nvurisrcbin; do
  gst-inspect-1.0 "$p" > "$ENV_DIR/gst_$p.txt" 2>&1 || true
done
nvidia-smi --query-gpu=driver_version --format=csv,noheader > /dev/null 2>&1 \
  || die "nvidia-smi not functional in this container"
for p in nvstreammux nvinfer nvtracker nvurisrcbin; do
  gst-inspect-1.0 "$p" > /dev/null 2>&1 || die "gst-inspect-1.0: plugin $p missing"
done
grep -qi VPI "$DS_HOME/lib"/*.so 2>/dev/null; VPI_HINT=$?
ls "$DS_HOME"/lib/libvpi* > "$ENV_DIR/vpi_libs.txt" 2>&1 || true
log "env recorded in $ENV_DIR"

# ---------------------------------------------------------------- 2. pyds ladder
PYDS_PY=""
if python3 -c "import pyds" > /dev/null 2>&1; then
  PYDS_PY="$(command -v python3)"
  log "pyds: importable from system python3 ($(command -v python3))"
fi
if [ -z "$PYDS_PY" ]; then
  VENV="$WORK/venv"
  if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import pyds, numpy" > /dev/null 2>&1; then
    PYDS_PY="$VENV/bin/python"
    log "pyds: via existing venv $VENV"
  fi
fi
# rung 2: create the venv + NVIDIA wheel
if [ -z "$PYDS_PY" ]; then
  VENV="$WORK/venv"
  log "pyds: creating venv (rung 2: --system-site-packages + numpy<2)"
  python3 -m venv --system-site-packages "$VENV" \
    || die "venv creation failed"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet "numpy<2"
  WHEEL_DIR="$DS_HOME"/python
  WHEEL=$(ls "$WHEEL_DIR"/pyds-*.whl 2>/dev/null | head -n1 || true)
  if [ -n "$WHEEL" ]; then
    "$VENV/bin/pip" install --quiet "$WHEEL" || die "pyds wheel install failed"
  else
    die "no pyds wheel under $WHEEL_DIR (rung 3: source build not automated; install deepstream_python_apps bindings manually)"
  fi
  "$VENV/bin/python" -c "import pyds" || die "pyds still not importable"
  PYDS_PY="$VENV/bin/python"
fi
echo "$PYDS_PY" > "$WORK/env/python.txt"
printf 'rung: %s\n' "$(
  case "$PYDS_PY" in */venv/*) echo "venv+wheel";; *) echo "system";; esac)" \
  > "$WORK/env/python_rung.txt"
log "pyds ok via: $PYDS_PY"
log "pyds NvDsFrameMeta object attrs: $(python3 -c 'import pyds; print([a for a in dir(pyds.NvDsFrameMeta) if "obj" in a.lower()])' 2>&1 | tail -1)"

# ------------------------------------------------- 3. sample source + labels
SAMPLE="$DS_HOME/samples/streams/sample_720p.mp4"
[ -f "$SAMPLE" ] || die "sample source missing: $SAMPLE"
log "sample source ok: $SAMPLE"

LABELS_SRC="${LABELS_SRC:-$SRC/templates/coco80.txt}"
if [ ! -f "$LABELS_SRC" ]; then
  LABELS_SRC="/work/deepstream9/tools/yolo_deepstream/deepstream_yolo/labels.txt"
fi
if [ ! -f "$LABELS_SRC" ]; then
  die "COCO labels not found (looked in samples + /work/deepstream9)"
fi
cp -f "$LABELS_SRC" "$WORK/cfg/labels.txt"
log "labels: $WORK/cfg/labels.txt ($(wc -l < "$WORK/cfg/labels.txt") lines)"

# ------------------------------------------------- 4. model export + A/B gate
ONNX="$WORK/onnx/yolo26m.onnx"
ONNX_RAW="$WORK/onnx/yolo26m-raw.onnx"
IO_JSON="$WORK/env/onnx_io.json"
UV="$WORK/uv-venv"
if [ ! -f "$ONNX" ] && [ ! -f "$ONNX_RAW" ]; then
  if [ ! -x "$UV/bin/python" ]; then
    log "creating ultralytics export venv (torch download is the slow part)"
    python3 -m venv "$UV" || die "export venv creation failed"
    "$UV/bin/pip" install --quiet --upgrade pip
    "$UV/bin/pip" install --quiet ultralytics onnx
  fi
  log "exporting yolo26m -> ONNX (e2e)"
  ( cd "$WORK/onnx" \
    && "$UV/bin/yolo" export model=yolo26m.pt format=onnx imgsz=640 \
         dynamic=True half=False opset=17 simplify=True \
    && mv yolo26m.onnx yolo26m.e2e.onnx ) \
    || die "yolo e2e export failed"
  mv "$WORK/onnx/yolo26m.e2e.onnx" "$ONNX"
fi
if [ ! -f "" ] || ! grep -q "\"outputs\"" "" 2>/dev/null; then
  if [ ! -x "$UV/bin/python" ]; then
    log "creating export venv (onnx inspect only)"
    python3 -m venv "$UV" || die "export venv creation failed"
    "$UV/bin/pip" install --quiet --upgrade pip
    "$UV/bin/pip" install --quiet onnx
  fi
  log "inspecting ONNX I/O (Plan A/B gate)"
  PLAN=$("$UV/bin/python" - "$ONNX" "$IO_JSON" <<'PY'
import json, sys, time
try:
    import onnx
except ImportError:
    print("A-fallback", file=sys.stderr)
    print("A")
    raise SystemExit(0)
m = onnx.load(sys.argv[1])
inp = m.graph.input[0]
name = inp.name
dims = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
symbolic = isinstance(dims[0], str) and dims[0] != ""
plan = "A" if symbolic else "B"
with open(sys.argv[2], "w") as f:
    outs = []
    for o in m.graph.output:
        odims = [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]
        outs.append({"name": o.name, "dims": [str(d) for d in odims]})
    json.dump({"input_name": name, "dims": [str(d) for d in dims],
               "outputs": outs, "plan": plan, "onnx": sys.argv[1],
               "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=2)
    f.write("\n")
print(plan)
PY
  )
  log "gate said plan: $PLAN"
fi
if [ -f "$IO_JSON" ] && grep -q '"plan": "B"' "$IO_JSON" && [ ! -f "$ONNX_RAW" ]; then
  log "Plan B: re-exporting raw-head ONNX (end2end=False)"
  ( cd "$WORK/onnx" \
    && "$UV/bin/yolo" export model=yolo26m.pt format=onnx imgsz=640 \
         dynamic=True half=False opset=17 simplify=True end2end=False \
    && mv yolo26m.onnx yolo26m-raw.onnx ) \
    || die "Plan B raw export failed"
fi
if [ -f "$IO_JSON" ] && grep -q '"plan": "B"' "$IO_JSON"; then
  ONNX="$ONNX_RAW"
fi
log "chosen ONNX: $ONNX (plan $(grep -o '"plan": "[AB]"' "$IO_JSON" | cut -d'"' -f4))"

# ------------------------------------------------------- 5. parser + tracker
if [ ! -f "$WORK/lib/libnvdsinfer_custom_impl_Yolo26E2e.so" ]; then
  log "building parsers (SMS=75)"
  make -C "$SRC/parser" all SMS="${SMS:-75}" DS_HOME="$DS_HOME" \
    || die "parser build failed"
  cp -f "$SRC/parser/libnvdsinfer_custom_impl_Yolo26E2e.so" "$WORK/lib/"
  cp -f "$SRC/parser/yolo_raw/libnvdsinfer_custom_impl_Yolo.so" "$WORK/lib/" \
    || log "WARN: raw parser not built (fine for Plan A)"
fi
ls -la "$WORK/lib" > "$ENV_DIR/parser_libs.txt" 2>&1

TRACKER_YML="$WORK/cfg/tracker.yml"
python3 "$SRC/make_tracker_yml.py" --out "$TRACKER_YML" \
  || die "tracker yml generation failed"
log "tracker yml: $TRACKER_YML ($(sed -n 2p "$TRACKER_YML"))"

# ------------------------------------------------------------ 6. final checks
for f in "$ONNX" "$IO_JSON" "$TRACKER_YML" "$WORK/cfg/labels.txt"; do
  [ -f "$f" ] || die "missing artifact after setup: $f"
done
PARSER_LIB="$WORK/lib/libnvdsinfer_custom_impl_Yolo26E2e.so"
[ -f "$PARSER_LIB" ] || die "missing parser lib: $PARSER_LIB"
log "setup complete: pyds=$(cat "$WORK/env/python.txt") plan=$(grep -o '"plan": "[AB]"' "$IO_JSON" | cut -d'"' -f4)"
exit 0