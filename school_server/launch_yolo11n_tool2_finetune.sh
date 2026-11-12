#!/usr/bin/env bash
set -euo pipefail

# Leakage-safe YOLO11n experiment:
#   M0 = Tool2 train-only base model (fixed NEW valid/test holdout)
#   C  = M0 fine-tuned on Tool2 + approved factory synthetic images
#   D  = M0 fine-tuned on Tool2 + NEW clean train images
#   E  = M0 fine-tuned on Tool2 + NEW clean train + approved synthetic images
#
# C, D and E all load the exact same M0 checkpoint.  This keeps the branch
# comparison fair and avoids using the result of one branch as the starting
# point for another.  The script is intended to run inside tmux so it survives
# an SSH/VS Code disconnect.

ROOT="${WORKPIECE_ROOT:-$HOME/workpiece_data_studio}"
PYTHON="$ROOT/.school-env/bin/python"
TRAIN="$ROOT/school_server/train_yolo.py"
INPUTS="$ROOT/training_inputs"
NEW_ZIP="$INPUTS/new-workpiece-clean-yolov11.zip"
TOOL2_ZIP="$INPUTS/tool2-clean-yolov11.zip"
SYNTHETIC_ZIP="${SYNTHETIC_ZIP:-$INPUTS/factory-synthetic-reviewed.yolov8.zip}"
OUT="$ROOT/school_training/tool2_finetune"
LOG_DIR="$OUT/logs"
SESSION="${TMUX_SESSION:-workpiece_yolo11n_tool2_finetune}"

for required in "$PYTHON" "$TRAIN" "$NEW_ZIP" "$TOOL2_ZIP" "$SYNTHETIC_ZIP"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$LOG_DIR"

run_train() {
  local label="$1"
  shift
  local log="$LOG_DIR/${label}.log"
  echo "[$(date '+%F %T')] START $label" | tee -a "$log"
  "$PYTHON" -u "$TRAIN" "$@" 2>&1 | tee -a "$log"
  echo "[$(date '+%F %T')] DONE $label" | tee -a "$log"
}

run_video_eval() {
  local log="$LOG_DIR/video_evaluation.log"
  echo "[$(date '+%F %T')] START video_evaluation" | tee -a "$log"
  "$PYTHON" "$VIDEO_EVAL" "$@" 2>&1 | tee -a "$log"
  echo "[$(date '+%F %T')] DONE video_evaluation" | tee -a "$log"
}

COMMON=(
  --source "$NEW_ZIP"
  --aux-yolo "$TOOL2_ZIP"
  --aux-yolo-train-only
  --imgsz 640
  --export-imgsz 640
  --batch 32
  --device 6,8
  --workers 8
  --seed 42
)

# M0: Tool2 is the only real training source.  NEW train is intentionally
# excluded, while NEW valid/test remain the fixed unseen-workpiece holdout.
run_train M0_tool2 \
  "${COMMON[@]}" \
  --exclude-primary-train \
  --model yolo11n.pt \
  --epochs 150 \
  --optimizer auto \
  --name YOLO11n_M0_tool2 \
  --output "$OUT/M0"

M0_BEST="$OUT/M0/runs/YOLO11n_M0_tool2/weights/best.pt"
if [[ ! -f "$M0_BEST" ]]; then
  echo "M0 checkpoint was not produced: $M0_BEST" >&2
  exit 3
fi

# Fine-tuning branches use the same replay source (Tool2) and a conservative
# AdamW learning rate to reduce catastrophic forgetting.  C/D/E differ only
# in which additional dataset is supplied.
run_train C_ft_tool2_synthetic \
  "${COMMON[@]}" \
  --exclude-primary-train \
  --synthetic "$SYNTHETIC_ZIP" \
  --synthetic-max-fraction 1.0 \
  --model "$M0_BEST" \
  --epochs 60 \
  --optimizer AdamW \
  --lr0 0.001 \
  --lrf 0.01 \
  --name YOLO11n_C_ft_tool2_synthetic \
  --output "$OUT/C"

run_train D_ft_tool2_new \
  "${COMMON[@]}" \
  --model "$M0_BEST" \
  --epochs 60 \
  --optimizer AdamW \
  --lr0 0.001 \
  --lrf 0.01 \
  --name YOLO11n_D_ft_tool2_new \
  --output "$OUT/D"

run_train E_ft_tool2_new_synthetic \
  "${COMMON[@]}" \
  --synthetic "$SYNTHETIC_ZIP" \
  --synthetic-max-fraction 1.0 \
  --model "$M0_BEST" \
  --epochs 60 \
  --optimizer AdamW \
  --lr0 0.001 \
  --lrf 0.01 \
  --name YOLO11n_E_ft_tool2_new_synthetic \
  --output "$OUT/E"

# Video evaluation is test-only and runs only after all three branch models
# exist.  The two videos are copied to video_tests by the deployment step.
VIDEO_EVAL="$ROOT/video_tests/evaluate_videos.py"
VIDEO_1="$ROOT/video_tests/808041065.167705.mp4"
VIDEO_2="$ROOT/video_tests/808041065.207860.mp4"
if [[ -f "$VIDEO_EVAL" && -f "$VIDEO_1" && -f "$VIDEO_2" ]]; then
  run_video_eval \
      --model "$OUT/C/runs/YOLO11n_C_ft_tool2_synthetic/weights/best.pt" \
      --model "$OUT/D/runs/YOLO11n_D_ft_tool2_new/weights/best.pt" \
      --model "$OUT/E/runs/YOLO11n_E_ft_tool2_new_synthetic/weights/best.pt" \
      --video "$VIDEO_1" \
      --video "$VIDEO_2" \
      --output "$OUT/video_evaluation" \
      --device 6 \
      --conf 0.25
else
  echo "[$(date '+%F %T')] Video files/evaluator not present; skipping video evaluation." | tee -a "$LOG_DIR/video_evaluation.log"
fi

echo "[$(date '+%F %T')] ALL DONE. Outputs: $OUT" | tee -a "$LOG_DIR/sequence.log"
