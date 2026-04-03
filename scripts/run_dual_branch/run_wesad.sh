#!/usr/bin/env bash
set -euo pipefail

WESAD_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/pretrain_dual_branch/no_scale/pretrain_wesad.json"
SWELL_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/pretrain_dual_branch/no_scale/pretrain_swell.json"
STRESSID_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/pretrain_dual_branch/no_scale/pretrain_stressid.json"
PORT=23503
NPROC=2

echo "Running Dual Branch Subject-aware contrastive learning for dataset WESAD..."

torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${WESAD_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "WESADDataset" 

torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${SWELL_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "WESADDataset" \
  --resume_finetune 0
  
torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${STRESSID_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "WESADDataset" \
  --resume_finetune 0

echo "All runs completed."