#!/usr/bin/env bash
set -euo pipefail

# Main comparison for the question: does the reviewed synthetic pool improve
# generalization over the same real training data?  The existing best.pt is
# evaluated separately as the historical baseline; these two runs are a
# reproducible retraining control/treatment pair.
GROUP="${1:-}"
GPU="${2:-}"
ROOT="${WORKPIECE_ROOT:-$HOME/workpiece_data_studio}"
INPUTS="$ROOT/training_inputs"
PYTHON="$ROOT/.school-env/bin/python"
TRAIN="$ROOT/school_server/train_yolo.py"
NEW_ZIP="$INPUTS/new-workpiece-clean-yolov11.zip"
TOOL2_ZIP="$INPUTS/tool2-clean-yolov11.zip"
SYNTHETIC_ZIP="${SYNTHETIC_ZIP:-$INPUTS/factory-synthetic-reviewed.yolov8.zip}"

if [[ "$GROUP" != "CONTROL" && "$GROUP" != "SYNTHETIC" ]]; then
  echo "Usage: $0 CONTROL|SYNTHETIC 6|8" >&2
  exit 2
fi
if [[ "$GPU" != "6" && "$GPU" != "8" && "$GPU" != "6,8" && "$GPU" != "8,6" ]]; then
  echo "GPU must be physical id 6, 8, or the explicit pair 6,8." >&2
  exit 2
fi
for required in "$NEW_ZIP" "$TOOL2_ZIP"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing: $required" >&2
    exit 2
  fi
done
if [[ "$GROUP" == "SYNTHETIC" && ! -f "$SYNTHETIC_ZIP" ]]; then
  echo "Missing reviewed synthetic ZIP: $SYNTHETIC_ZIP" >&2
  exit 2
fi

ARGS=(
  --source "$NEW_ZIP"
  --aux-yolo "$TOOL2_ZIP"
  --aux-yolo-train-only
  --model yolo11n.pt
  --epochs 150
  --imgsz 640
  --export-imgsz 640
  # AutoBatch is not supported by Ultralytics DDP; 32 is divisible by two
  # A5000 cards and leaves ample memory for YOLO11n at 640px.
  --batch 32
  --device "$GPU"
  --workers 8
  --seed 42
  --name "YOLO11n_${GROUP}"
  --output "$ROOT/school_training/YOLO11n_${GROUP}"
)
if [[ "$GROUP" == "SYNTHETIC" ]]; then
  # Include every one of the 321 reviewed images.  The script still keeps
  # synthetic data train-only and applies the same online augmentation to all
  # groups.
  ARGS+=(--synthetic "$SYNTHETIC_ZIP" --synthetic-max-fraction 1.0)
fi

mkdir -p "$ROOT/school_training/logs"
LOG="$ROOT/school_training/logs/YOLO11n_${GROUP}.log"
PID_FILE="$ROOT/school_training/logs/YOLO11n_${GROUP}.pid"
SESSION="workpiece_yolo11n_${GROUP,,}"

# The school's SSH service cleans up ordinary background children on logout.
# tmux keeps the one training process alive independently of this SSH session.
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "A tmux session is already running: $SESSION" >&2
  exit 3
fi
: >"$LOG"
RUN_CMD=$(printf '%q ' "$PYTHON" -u "$TRAIN" "${ARGS[@]}")
tmux new-session -d -s "$SESSION" "cd '$ROOT' && exec $RUN_CMD > '$LOG' 2>&1"
PID=$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -n 1)
echo "$PID" >"$PID_FILE"
echo "Started YOLO11n generalization group $GROUP on physical GPU $GPU (tmux=$SESSION pid=$PID)"
echo "Log: $LOG"
