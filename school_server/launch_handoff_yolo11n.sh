#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  echo "Usage: $0 SMOKE|M0|C|D|E 6|8 RUN_ID [M0_BEST_PT]" >&2
  exit 2
fi
group="$1"
gpu="$2"
run_id="$3"
if [[ ! "$group" =~ ^(SMOKE|M0|C|D|E)$ || ! "$gpu" =~ ^(6|8)$ ||
      ! "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "Choose a listed group, one allocated physical GPU (6 or 8), and a simple run ID." >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
python="$root/.school-env/bin/python"
inputs="$root/training_inputs"
output="$root/school_training/handoff/$run_id"
[[ -x "$python" ]] || { echo "Run school_server/bootstrap.sh first." >&2; exit 2; }

args=(
  --source "$inputs/new-workpiece-clean-yolov11.zip"
  --imgsz 640 --export-imgsz 640 --device "$gpu" --seed 42
  --name "$group" --output "$output"
)
case "$group" in
  SMOKE)
    args+=(--model yolo11n.pt --epochs 1 --batch 8 --workers 2)
    ;;
  M0)
    args+=(--model yolo11n.pt --epochs 150 --optimizer auto --exclude-primary-train)
    ;;
  C|D|E)
    checkpoint="${4:-}"
    [[ -f "$checkpoint" ]] || { echo "Supply the shared M0 best.pt as argument 4." >&2; exit 2; }
    args+=(--model "$checkpoint" --epochs 60 --optimizer AdamW --lr0 0.001 --lrf 0.01)
    ;;
esac
if [[ "$group" != SMOKE ]]; then
  args+=(--aux-yolo "$inputs/tool2-clean-yolov11.zip" --aux-yolo-train-only --batch 32 --workers 8)
fi
if [[ "$group" == C ]]; then
  args+=(--exclude-primary-train)
fi
if [[ "$group" =~ ^[CE]$ ]]; then
  args+=(--synthetic "${SYNTHETIC_ZIP:-$inputs/factory-synthetic-reviewed.yolov8.zip}" --synthetic-max-fraction 1.0)
fi

mkdir -p "$(dirname "$output")"
mkdir "$output"
printf '%q ' "$python" -u school_server/train_yolo.py "${args[@]}" > "$output/command.txt"
printf '\n' >> "$output/command.txt"
"$python" -m pip freeze > "$output/environment.txt"
"$python" -u school_server/train_yolo.py "${args[@]}" 2>&1 | tee "$output/train.log"
