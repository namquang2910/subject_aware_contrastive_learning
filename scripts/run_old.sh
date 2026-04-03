#!/usr/bin/env bash
set -euo pipefail

CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/decrapted/pretrain_config_different.json"
PORT=23505
NPROC=2

echo "Running Moe..."
torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  train_dp.py \
  --config_path "${CONFIG}" \

echo "All runs completed."