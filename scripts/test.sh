#!/usr/bin/env bash
set -euo pipefail

CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/pretrain_moe_wesad_loso.json"
PORT=23503
NPROC=2

echo "Running Moe..."
torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${CONFIG}" \
  --model_type moe_dual_branch

echo "All runs completed."