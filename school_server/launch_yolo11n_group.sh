#!/usr/bin/env bash
set -euo pipefail

GROUP="${1:-}"
GPU="${2:-}"
ROOT="${WORKPIECE_ROOT:-$HOME/workpiece_data_studio}"
INPUTS="$ROOT/training_inputs"
PYTHON="$ROOT/.school-env/bin/python"
TRAIN="$ROOT/school_server/train_yolo.py"
NEW_ZIP="$INPUTS/new-workpiece-clean-yolov11.zip"
TOOL2_ZIP="$INPUTS/tool2-clean-yolov11.zip"
SYNTHETIC_ZIP="${SYNTHETIC_ZIP:-$INPUTS/factory-synthetic-reviewed.yolov8.zip}"

if [[ ! "$GROUP" =~ ^[ABCD]$ ]]; then
  echo "Usage: $0 A|B|C|D 2|3|6" >&2
  exit 2
fi
if [[ ! "$GPU" =~ ^(6|8)$ ]]; then
  echo "GPU must be physical id 6 or 8." >&2
  exit 2
fi
if [[ ! -f "$NEW_ZIP" ]]; then
  echo "Missing: $NEW_ZIP" >&2
  exit 2
fi
if [[ "$GROUP" =~ ^[BD]$ && ! -f "$TOOL2_ZIP" ]]; then
  echo "Missing: $TOOL2_ZIP" >&2
  exit 2
fi
if [[ "$GROUP" =~ ^[CD]$ && ! -f "$SYNTHETIC_ZIP" ]]; then
  echo "Missing reviewed synthetic ZIP: $SYNTHETIC_ZIP" >&2
  exit 2
fi

ARGS=(
  --source "$NEW_ZIP"
  --model yolo11n.pt
  --epochs 150
  --imgsz 640
  --export-imgsz 640
  --batch -1
  --device "$GPU"
  --workers 8
  --seed 42
  --name "YOLO11n_${GROUP}"
  --output "$ROOT/school_training/YOLO11n_${GROUP}"
)

if [[ "$GROUP" =~ ^[BD]$ ]]; then
  ARGS+=(--aux-yolo "$TOOL2_ZIP")
fi
if [[ "$GROUP" =~ ^[CD]$ ]]; then
  # The same reviewed pool is used by C and D. A cap of 1.0 lets all of a
  # roughly 300-image pool enter C while avoiding accidental runaway imports.
  ARGS+=(--synthetic "$SYNTHETIC_ZIP" --synthetic-max-fraction 1.0)
fi

mkdir -p "$ROOT/school_training/logs"
LOG="$ROOT/school_training/logs/YOLO11n_${GROUP}.log"
PID_FILE="$ROOT/school_training/logs/YOLO11n_${GROUP}.pid"

nohup "$PYTHON" "$TRAIN" "${ARGS[@]}" >"$LOG" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"
echo "Started YOLO11n group $GROUP on physical GPU $GPU (pid=$PID)"
echo "Log: $LOG"
