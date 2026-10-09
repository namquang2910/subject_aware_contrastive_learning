#!/usr/bin/env bash
set -euo pipefail

CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/pretrain_constrative_mdd.json"
PORT=23505
NPROC=2


# /opt/miniconda3/bin is ahead of ~/miniconda3 in PATH, so plain `conda
# activate` still resolves torchrun/python to the wrong install (its base
# env has a torch/nccl version mismatch: undefined symbol ncclCommResume).
# Call the ssl_torch env's torchrun by full path instead.
TORCHRUN="/home/s223149341/miniconda3/envs/ssl_torch/bin/torchrun"


echo "Running contrastive..."
"${TORCHRUN}" \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${CONFIG}" \
  --model_type contrastive \
  --dataset "MDDDataset" \
  --resume_finetune 0 \
  --finetune_fraction 0.01

echo "Running contrastive..."
"${TORCHRUN}" \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${CONFIG}" \
  --model_type contrastive \
  --dataset "PsychioNet" \
  --resume_finetune 0 \
  --finetune_fraction 0.01

"${TORCHRUN}" \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${CONFIG}" \
  --model_type contrastive \
  --dataset "MDDDataset" \
  --resume_finetune 0 \
  --finetune_fraction 0.05

"${TORCHRUN}" \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${CONFIG}" \
  --model_type contrastive \
  --dataset "PsychioNet" \
  --resume_finetune 0 \
  --finetune_fraction 0.05

"${TORCHRUN}" \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${CONFIG}" \
  --model_type contrastive \
  --dataset "MDDDataset" \
  --resume_finetune 0 \
  --finetune_fraction 0.1

"${TORCHRUN}" \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${CONFIG}" \
  --model_type contrastive \
  --dataset "PsychioNet" \
  --resume_finetune 0 \
  --finetune_fraction 0.1


echo "All runs completed."