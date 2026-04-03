#!/usr/bin/env bash
set -euo pipefail

CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/pretrain_moe_n_branch.json"
PORT=23501
NPROC=2

echo "Running Moe..."

torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${CONFIG}" \
  --model_type moe_dual_n_pretrain \
  --dataset "SWELLDataset" \
  --resume_finetune 0

echo "All runs completed."