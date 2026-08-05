#!/usr/bin/env bash
set -euo pipefail

cd /home/ping/efficientdet_project
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate efficientdet_lite_env

export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TFHUB_CACHE_DIR=/home/ping/efficientdet_project/tfhub_cache

out=experiments/tool2_v18_lite2_speed_3gpu
nohup python -u train_efficientdet_lite2_stage1.py \
  --dataset datasets/tool2_roboflow_20260720_voc \
  --output-dir "$out" \
  --gpus 2 \
  --epochs 100 \
  --global-batch-size 16 \
  --eval-batch-size 12 \
  --learning-rate 0.08 \
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
  --resume-weights "$out/checkpoints/ckpt-100" \
  --evaluate-only \
  > "$out/rescue_eval_export.log" 2>&1 &

echo "$!" > "$out/rescue_eval_export.pid"
echo "Started rescue evaluation/export with PID $!"
