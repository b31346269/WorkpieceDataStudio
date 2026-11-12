#!/usr/bin/env bash
set -euo pipefail

# Leakage-safe C/D/E comparison requested by the user:
# C = Tool 2 train + approved factory synthetic train
# D = Tool 2 train + NEW workpiece train
# E = Tool 2 train + NEW workpiece train + approved factory synthetic train
# The primary NEW archive supplies the common fixed valid/test holdout.  Its
# train split is included only for D/E.  Raw Roboflow train composition is
# retained except for source-id/hash overlaps with the locked holdout, so the
# experiment does not leak near-duplicate variants into evaluation.
GROUP="${1:-}"
GPU="${2:-}"
ROOT="${WORKPIECE_ROOT:-$HOME/workpiece_data_studio}"
INPUTS="$ROOT/training_inputs"
PYTHON="$ROOT/.school-env/bin/python"
TRAIN="$ROOT/school_server/train_yolo.py"
NEW_ZIP="$INPUTS/new-workpiece-clean-yolov11.zip"
TOOL2_ZIP="$INPUTS/tool2-clean-yolov11.zip"
SYNTHETIC_ZIP="${SYNTHETIC_ZIP:-$INPUTS/factory-synthetic-reviewed.yolov8.zip}"

if [[ ! "$GROUP" =~ ^[CDE]$ ]]; then
  echo "Usage: $0 C|D|E 6|8|6,8" >&2
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
if [[ "$GROUP" =~ ^[CE]$ && ! -f "$SYNTHETIC_ZIP" ]]; then
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
  --batch 32
  --device "$GPU"
  --workers 8
  --seed 42
  --name "YOLO11n_${GROUP}"
  --output "$ROOT/school_training/YOLO11n_${GROUP}"
)

# C is intentionally Tool2-only; the primary NEW train split remains absent
# while NEW valid/test stay available as the common fixed holdout.
if [[ "$GROUP" == "C" ]]; then
  ARGS+=(--exclude-primary-train)
fi
if [[ "$GROUP" =~ ^[CE]$ ]]; then
  ARGS+=(--synthetic "$SYNTHETIC_ZIP" --synthetic-max-fraction 1.0)
fi

mkdir -p "$ROOT/school_training/logs"
LOG="$ROOT/school_training/logs/YOLO11n_${GROUP}.log"
PID_FILE="$ROOT/school_training/logs/YOLO11n_${GROUP}.pid"
SESSION="workpiece_yolo11n_${GROUP,,}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "A tmux session is already running: $SESSION" >&2
  exit 3
fi
: >"$LOG"
RUN_CMD=$(printf '%q ' "$PYTHON" -u "$TRAIN" "${ARGS[@]}")
tmux new-session -d -s "$SESSION" "cd '$ROOT' && exec $RUN_CMD > '$LOG' 2>&1"
PID=$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -n 1)
echo "$PID" >"$PID_FILE"
echo "Started YOLO11n group $GROUP on physical GPU $GPU (tmux=$SESSION pid=$PID)"
echo "Log: $LOG"
