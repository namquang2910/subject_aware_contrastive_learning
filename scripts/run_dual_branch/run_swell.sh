#!/usr/bin/env bash
set -euo pipefail

WESAD_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/pretrain_dual_branch/moe_dual/pretrain_wesad.json"
SWELL_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/pretrain_dual_branch/moe_dual/pretrain_swell.json"
PORT=23515
NPROC=4

echo "Running Dual Branch Subject-aware contrastive learning for dataset SWELL..."


torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${SWELL_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "SWELLDataset" \
  --resume_finetune 0\
  --finetune_fraction 0.01 \
  
echo "All runs completed."