#!/bin/bash
#SBATCH --job-name=zinv_ablation_wesad
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "========================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURM_NODELIST}"
echo "GPUs:         ${CUDA_VISIBLE_DEVICES:-not_set}"
echo "Start time:   $(date)"
echo "========================================"

cd /home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch310

mkdir -p logs

nvidia-smi

echo "========================================"
echo "Starting z_inv-only ablation (wesad_wesad, resuming from any completed folds)..."
echo "========================================"

/home/s223149341/miniconda3/envs/pytorch310/bin/python3 analysis/finetune_zinv_ablation.py --run wesad_wesad --device cuda --resume

echo "Done: $(date)"
