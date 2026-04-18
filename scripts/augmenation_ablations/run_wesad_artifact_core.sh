#!/usr/bin/env bash
set -euo pipefail

WESAD_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/augmentation_ablation/pretrain_wesad_artifact_core.json"
PORT=23502
NPROC=8

echo "Running Dual Branch Subject-aware contrastive learning for dataset WESAD..."


torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${WESAD_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "WESADDataset" 



echo "All runs completed."