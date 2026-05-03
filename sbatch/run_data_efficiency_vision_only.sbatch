#!/bin/bash
#SBATCH --job-name=mi_eff_vis
#SBATCH --output=logs/mi_eff_vis_%j.log
#SBATCH --error=logs/mi_eff_vis_err_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

set -e

# =============================================================================
# Data efficiency sweep with vision_only embeddings (MLP probe)
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
echo "=== Data Efficiency Sweep (vision_only MLP) ==="
echo "Start time: $(date +'%Y-%m-%d %H:%M:%S')"

for frac in 0.01 0.05 0.1 0.2 1.0; do
  for seed in 1 2 3; do
    echo "--- frac=$frac seed=$seed ---"
    python scripts/train_probe.py \
      --embedding_dir ./embeddings/vision_only \
      --model_type mlp \
      --epochs 30 \
      --lr 1e-4 \
      --batch_size 64 \
      --train_fraction "$frac" \
      --seed "$seed" \
      --run_name pooling_vision_only \
      --output_dir ./output/probe_vision_only
  done
done

echo ""
echo "End time: $(date +'%Y-%m-%d %H:%M:%S')"
echo "=== Done ==="