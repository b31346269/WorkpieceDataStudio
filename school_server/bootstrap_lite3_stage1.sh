#!/usr/bin/env bash
set -euo pipefail

cd /home/ping/efficientdet_project

python3 prepare_disjoint_voc.py \
  --input datasets/tool2_roboflow_20260726_512_voc \
  --output datasets/tool2_roboflow_20260726_512_clean_voc

python3 prepare_stage2_rehearsal.py \
  --new-dataset datasets/new_workpiece_roboflow_20260726_512_voc \
  --tool2-dataset datasets/tool2_roboflow_20260726_512_clean_voc \
  --output datasets/stage2_new_workpiece_rehearsal_512_voc \
  --rehearsal-images 300 \
  --seed 20260726

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate efficientdet_lite_env

python train_efficientdet_lite2_stage1.py \
  --model-name efficientdet_lite3 \
  --dataset datasets/tool2_roboflow_20260726_512_clean_voc \
  --output-dir experiments/tool2_v19_512_lite3_speed_accuracy_3gpu \
  --gpus 2,3,6 \
  --epochs 100 \
  --global-batch-size 36 \
  --eval-batch-size 12 \
  --audit-only

bash launch_efficientdet_lite3_stage1.sh
