#!/usr/bin/env bash
set -euo pipefail

CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/pretrain_loso_wesad.json"
PORT=23505
NPROC=2

echo "Running contrastive..."
torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  loso.py \
  --config_path "${CONFIG}" \
  --model_type contrastive


echo "Running subject_specific..."
torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  loso.py \
  --config_path "${CONFIG}" \
  --model_type subject_specific
  
echo "Running subject_invariant..."
torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  loso.py \
  --config_path "${CONFIG}" \
  --model_type subject_invariant

echo "All runs completed."