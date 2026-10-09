#!/bin/bash
#SBATCH --job-name=simsiam_pretrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=60:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "========================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURM_NODELIST}"
echo "GPUs:         ${CUDA_VISIBLE_DEVICES:-not_set}"
echo "Start time:   $(date)"
echo "========================================"

# Go to project directory
cd /home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning

# Activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch310

# Create log directory
mkdir -p logs

# Check GPUs
nvidia-smi

echo "========================================"
echo "Starting training..."
echo "========================================"



WESAD_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/byol_simsiam/moe_dual/pretrain_wesad_simsiam.json"
SWELL_CONFIG="/home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning/configs/byol_simsiam/moe_dual/pretrain_swell_simsiam.json"
PORT=23510
NPROC=1

echo "Running Dual Branch Subject-aware contrastive learning for dataset PsychioNet..."


  torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${SWELL_CONFIG}" \
  --model_type simsiam \
  --dataset "PsychioNet"  \
  --resume_finetune 0\
  --finetune_fraction 0.01


echo "========================================"
echo "All runs completed."
echo "End time: $(date)"
echo "========================================"