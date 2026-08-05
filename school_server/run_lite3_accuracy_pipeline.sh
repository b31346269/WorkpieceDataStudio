#!/usr/bin/env bash
set -euo pipefail

cd /home/ping/efficientdet_project
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate efficientdet_lite_env

export CUDA_VISIBLE_DEVICES=2,3,6
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TFHUB_CACHE_DIR=/home/ping/efficientdet_project/tfhub_cache

stage1_dataset=datasets/tool2_roboflow_20260726_512_clean_voc
stage2_dataset=datasets/stage2_new_workpiece_accuracy_512_voc
stage1_output=experiments/tool2_v20_512_lite3_small_anchor_disjoint_3gpu
stage2_output=experiments/new_workpiece_v9_512_lite3_small_anchor_rehearsal_3gpu

for dataset in "$stage1_dataset" "$stage2_dataset"; do
  [[ -d "$dataset/train" && -d "$dataset/valid" && -d "$dataset/test" ]] || {
    echo "Missing prepared dataset: $dataset" >&2
    exit 1
  }
done

mkdir -p "$stage1_output" "$stage2_output"

python -u train_efficientdet_lite2_stage1.py \
  --model-name efficientdet_lite3 \
  --dataset "$stage1_dataset" \
  --output-dir "$stage1_output" \
  --gpus 2,3,6 \
  --epochs 120 \
  --global-batch-size 24 \
  --eval-batch-size 6 \
  --learning-rate 0.04 \
  --anchor-scale 0.75 \
  --aspect-ratios 0.5 1.0 2.0 \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --max-instances 300 \
  --max-detections 30 \
  --score-threshold 0.10 \
  --iou-threshold 0.45 \
  --checkpoint-every-epochs 5 \
  --keep-checkpoints 3 \
  --skip-keras-eval \
  --tflite-filename efficientdet_lite3_tool2_512_small_anchor_fp16.tflite \
  > "$stage1_output/train.log" 2>&1

python -u train_efficientdet_lite2_stage1.py \
  --model-name efficientdet_lite3 \
  --dataset "$stage2_dataset" \
  --output-dir "$stage2_output" \
  --gpus 2,3,6 \
  --epochs 30 \
  --global-batch-size 12 \
  --eval-batch-size 6 \
  --learning-rate 0.0005 \
  --anchor-scale 0.75 \
  --aspect-ratios 0.5 1.0 2.0 \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --max-instances 300 \
  --max-detections 30 \
  --score-threshold 0.10 \
  --iou-threshold 0.45 \
  --checkpoint-every-epochs 5 \
  --keep-checkpoints 3 \
  --resume-weights "$stage1_output/checkpoints/ckpt-120" \
  --skip-keras-eval \
  --tflite-filename lite3_accuracy_newpieces_fp16.tflite \
  > "$stage2_output/train.log" 2>&1
