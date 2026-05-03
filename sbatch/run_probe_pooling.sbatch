#!/bin/bash
#SBATCH --job-name=mi_probe_pool
#SBATCH --output=logs/mi_probe_pool_%j.log
#SBATCH --error=logs/mi_probe_pool_err_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

set -e

# =============================================================================
# Probe training (Linear + MLP) for pooling ablation
# =============================================================================

# ============================
# 1) Environment
# ============================

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

# ============================
# 2) Probe training (pooling ablation)
# ============================

echo ""
echo "=== Probe Training (Pooling Ablation) ==="
echo "Start time: $(date +'%Y-%m-%d %H:%M:%S')"

for pooling in all_tokens vision_only; do
  echo ""
  echo ">>> Processing pooling=$pooling"
  EMBEDDING_DIR="./embeddings/$pooling"
  OUTPUT_DIR="./output/probe_$pooling"
  mkdir -p "$OUTPUT_DIR"

  echo "--- Linear probe ---"
  python scripts/train_probe.py \
    --embedding_dir "$EMBEDDING_DIR" \
    --model_type linear \
    --epochs 30 \
    --lr 1e-3 \
    --batch_size 64 \
    --run_name pooling_$pooling \
    --output_dir "$OUTPUT_DIR"

  echo "--- MLP probe ---"
  python scripts/train_probe.py \
    --embedding_dir "$EMBEDDING_DIR" \
    --model_type mlp \
    --epochs 50 \
    --lr 1e-4 \
    --batch_size 64 \
    --run_name pooling_$pooling \
    --output_dir "$OUTPUT_DIR"

  echo "<<< Done pooling=$pooling"
done

echo ""
echo "End time: $(date +'%Y-%m-%d %H:%M:%S')"
echo "=== Done ==="