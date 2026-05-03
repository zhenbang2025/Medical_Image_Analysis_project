#!/bin/bash
#SBATCH --job-name=mi_robust
#SBATCH --output=logs/mi_robust_%j.log
#SBATCH --error=logs/mi_robust_err_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

set -e

# =============================================================================
# Robustness evaluation for CNN (ResNet18)
# =============================================================================

PROJECT_ROOT="/home/hxiaoap/Medical_Image_Analysis_project"
ENV_PATH="/home/hxiaoap/.conda/envs/mia"

cd "$PROJECT_ROOT"
mkdir -p logs

module purge
module load Anaconda3/2023.09-0
module load cuda12.2/toolkit/12.2.2

export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"

echo "=== Environment Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python:    $(which python)"
echo "Version:   $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

echo ""
echo "=== Robustness Evaluation (ResNet18) ==="
echo "Start time: $(date +'%Y-%m-%d %H:%M:%S')"

python scripts/robustness_eval.py \
  --split_file ./splits/full_split.json \
  --model resnet18 \
  --model_path ./output/cnn_resnet18.pt \
  --batch_size 32 \
  --output_dir ./output

echo ""
echo "End time: $(date +'%Y-%m-%d %H:%M:%S')"
echo "=== Done ==="