#!/usr/bin/env bash
set -euo pipefail

CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/pretrain_config_overlap.json"
PORT=23502
NPROC=2


echo "Running subject_specific..."
torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${CONFIG}" \
  --model_type subject_specific \
  --resume_finetune 0

echo "All runs completed."