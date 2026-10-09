#!/usr/bin/env bash
set -euo pipefail

WESAD_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/byol_simsiam/moe_dual/pretrain_wesad_byol.json"
SWELL_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/byol_simsiam/moe_dual/pretrain_swell_byol.json"
MDD_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/byol_simsiam/moe_dual/pretrain_mdd_byol.json"
PORT=23508
NPROC=2

echo "Running Dual Branch Subject-aware contrastive learning for dataset PsychioNet..."


torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${MDD_CONFIG}" \
  --model_type byol \
  --dataset "MDDDataset" 

torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${MDD_CONFIG}" \
  --model_type byol \
  --dataset "MDDDataset" \
  --resume_finetune 0\
  --finetune_fraction 0.05

  torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${MDD_CONFIG}" \
  --model_type byol \
  --dataset "MDDDataset" \
  --resume_finetune 0\
  --finetune_fraction 0.1
  
torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${MDD_CONFIG}" \
  --model_type byol \
  --dataset "PsychioNet" 

torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${MDD_CONFIG}" \
  --model_type byol \
  --dataset "PsychioNet" \
  --resume_finetune 0\
  --finetune_fraction 0.05

  torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${MDD_CONFIG}" \
  --model_type byol \
  --dataset "PsychioNet" \
  --resume_finetune 0\
  --finetune_fraction 0.1
echo "All runs completed."