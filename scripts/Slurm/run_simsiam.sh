#!/bin/bash
#SBATCH --job-name=dual_branch_ssl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00               # 30hr estimate + 20% buffer
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL    # email you when job starts, ends, or crashes
#SBATCH --mail-user=s223149341@deakin.edu.au

cd /home/s223149341/SSL-invariance-Subject_Project_model/subject_aware_contrastive_learning

source $(conda info --base)/etc/profile.d/conda.sh
conda activate pytorch310

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPUs visible: $(python -c 'import torch; print(torch.cuda.device_count())')"
nvidia-smi

WESAD_CONFIG="configs/byol_simsiam/moe_dual/pretrain_wesad_simsiam.json"
SWELL_CONFIG="configs/byol_simsiam/moe_dual/pretrain_swell_simsiam.json"
NPROC=2

# ── WESAD pretraining ────────────────────────────────────────────
echo "========================================"
echo "START: WESAD pretraining $(date)"
echo "========================================"

torchrun \
  --nproc_per_node ${NPROC} \
  --master_port 23502 \
  single_train.py \
  --config_path "${WESAD_CONFIG}" \
  --model_type simsiam \
  --dataset "WESADDataset"

echo "DONE: WESAD pretraining $(date)"

# ── SWELL pretraining ────────────────────────────────────────────
echo "========================================"
echo "START: SWELL pretraining $(date)"
echo "========================================"

torchrun \
  --nproc_per_node ${NPROC} \
  --master_port 23503 \
  single_train.py \
  --config_path "${SWELL_CONFIG}" \
  --model_type simsiam \
  --dataset "SWELLDataset"

echo "DONE: SWELL pretraining $(date)"

echo "========================================"
echo "All runs completed $(date)"
echo "========================================"