#!/usr/bin/env bash
set -euo pipefail

SWELL_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/pretrain_swell_subject_specific.json"
PORT=23501
NPROC=4

echo "Running Dual Branch Subject-aware contrastive learning for dataset PsychioNet..."


torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${SWELL_CONFIG}" \
  --model_type contrastive \
  --dataset "SWELLDataset"
