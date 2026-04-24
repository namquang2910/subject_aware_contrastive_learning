#!/usr/bin/env bash
set -euo pipefail

WESAD_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/byol_simsiam/moe_dual/pretrain_wesad_simsiam.json"
SWELL_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/byol_simsiam/moe_dual/pretrain_swell_simsiam.json"
PORT=23502
NPROC=8

echo "Running Dual Branch Subject-aware contrastive learning for dataset PsychioNet..."

torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${WESAD_CONFIG}" \
  --model_type simsiam \
  --dataset "WESADDataset" 

torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${SWELL_CONFIG}" \
  --model_type simsiam \
  --dataset "SWELLDataset" 

echo "All runs completed."