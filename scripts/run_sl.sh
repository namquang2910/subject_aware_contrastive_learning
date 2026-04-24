#!/usr/bin/env bash
set -euo pipefail

WESAD_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/pretrain_dual_branch/moe_dual/SL_wesad.json"
SWELL_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/pretrain_dual_branch/moe_dual/SL_swell.json"
PORT=23503
NPROC=4

echo "Run SL on WESAD"


torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${WESAD_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "WESADDataset" \
  --resume_finetune 0 \
  --finetune_fraction 0.01

torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${WESAD_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "SWELLDataset" \
  --resume_finetune 0 \
  --finetune_fraction 0.01
