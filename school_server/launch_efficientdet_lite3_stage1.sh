#!/usr/bin/env bash
set -euo pipefail

cd /home/ping/efficientdet_project
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate efficientdet_lite_env

export CUDA_VISIBLE_DEVICES=2,3,6
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TFHUB_CACHE_DIR=/home/ping/efficientdet_project/tfhub_cache

dataset=datasets/tool2_roboflow_20260726_512_clean_voc
output=experiments/tool2_v19_512_lite3_speed_accuracy_3gpu

[[ -d "$dataset/train" && -d "$dataset/valid" && -d "$dataset/test" ]] || {
  echo "Missing 512x512 Tool2 Pascal VOC dataset: $dataset" >&2
  exit 1
}
mkdir -p "$output"

if [[ -f "$output/train.pid" ]] && kill -0 "$(cat "$output/train.pid")" 2>/dev/null; then
  echo "Training is already running with PID $(cat "$output/train.pid")." >&2
  exit 1
fi

nohup python -u train_efficientdet_lite2_stage1.py \
  --model-name efficientdet_lite3 \
  --dataset "$dataset" \
  --output-dir "$output" \
  --gpus 2,3,6 \
  --epochs 100 \
  --global-batch-size 36 \
  --eval-batch-size 12 \
  --learning-rate 0.06 \
  --anchor-scale 1.5 \
  --aspect-ratios 1.0 \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --max-instances 300 \
  --max-detections 30 \
  --score-threshold 0.20 \
  --iou-threshold 0.45 \
  --checkpoint-every-epochs 5 \
  --keep-checkpoints 3 \
  --tflite-filename efficientdet_lite3_tool2_512_fp16.tflite \
  > "$output/train.log" 2>&1 &

echo "$!" > "$output/train.pid"
echo "Started EfficientDet-Lite3 stage 1 with PID $!"
echo "Log: /home/ping/efficientdet_project/$output/train.log"
