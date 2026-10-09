#!/usr/bin/env bash
set -euo pipefail

MDD_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/pretrain_dual_branch/moe_dual/pretrain_mdd.json"
PORT=23515
NPROC=2

echo "Running Dual Branch Subject-aware contrastive learning for dataset SWELL..."


torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${MDD_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "MDDDataset" \
  --resume_finetune 0\
  --finetune_fraction 0.01 

torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${MDD_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "MDDDataset" \
  --resume_finetune 0\
  --finetune_fraction 0.05

  torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${MDD_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "MDDDataset" \
  --resume_finetune 0\
  --finetune_fraction 0.1

  torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${MDD_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "MDDDataset" \
  --resume_finetune 0\
  --finetune_fraction 1.0

torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${MDD_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "PsychioNet" \
  --resume_finetune 0\
  --finetune_fraction 0.01 
  
torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${MDD_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "PsychioNet" \
  --resume_finetune 0\
  --finetune_fraction 0.05

torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${MDD_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "PsychioNet" \
  --resume_finetune 0\
  --finetune_fraction 0.1


torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${MDD_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "PsychioNet" \
  --resume_finetune 0\
  --finetune_fraction 1.0

echo "All runs completed."