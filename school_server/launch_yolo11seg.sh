#!/usr/bin/env bash
set -euo pipefail

# Run the segmentation branch independently from the detector experiments on
# the ping account.  The two physical cards currently authorized are 6 and 8.
GPU="${1:-6}"
ROOT="${WORKPIECE_ROOT:-$HOME/workpiece_data_studio}"
DATA="${SEG_DATA:-$ROOT/datasets/workpiece_wrench_seg/data.yaml}"
PYTHON="$ROOT/.school-env/bin/python"
TRAIN="$ROOT/school_server/train_yolo11_seg.py"
NAME="${SEG_NAME:-YOLO11n_workpiece_wrench_seg}"
OUT="$ROOT/school_training/segmentation"
SESSION="workpiece_seg_${GPU//,/_}"
LOG_DIR="$OUT/logs"
LOG="$LOG_DIR/${NAME}_gpu${GPU//,/_}.log"

if [[ "$GPU" != "6" && "$GPU" != "8" && "$GPU" != "6,8" && "$GPU" != "8,6" ]]; then
  echo "GPU must be 6, 8, or the explicit pair 6,8." >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing school environment: $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$DATA" ]]; then
  echo "Missing segmentation data.yaml: $DATA" >&2
  exit 2
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "A tmux session is already running: $SESSION" >&2
  exit 3
fi

mkdir -p "$LOG_DIR"
RUN_CMD=$(printf '%q ' "$PYTHON" -u "$TRAIN" \
  --data "$DATA" --model yolo11n-seg.pt --epochs 120 --imgsz 640 \
  --batch 16 --device "$GPU" --name "$NAME" --output "$OUT")
tmux new-session -d -s "$SESSION" "cd '$ROOT' && exec $RUN_CMD > '$LOG' 2>&1"
echo "Started segmentation training on physical GPU $GPU (tmux=$SESSION)"
echo "Log: $LOG"
