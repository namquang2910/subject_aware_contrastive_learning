#!/bin/bash
#SBATCH --job-name=dual_branch_ssl_core
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=5:00:00               # 30hr estimate + 20% buffer
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

WESAD_CONFIG="configs/augmenation_ablations/pretrain_wesad_core.json"
PORT=23507
NPROC=1

echo "Running Dual Branch Subject-aware contrastive learning for dataset WESAD..."


torchrun \
  --nproc_per_node ${NPROC} \
  --master_port ${PORT} \
  single_train.py \
  --config_path "${WESAD_CONFIG}" \
  --model_type moe_dual_branch \
  --dataset "WESADDataset" \
  --resume_finetune 0



echo "All runs completed."