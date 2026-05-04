#!/bin/bash
#SBATCH --job-name=train_mlp_all
#SBATCH --output=logs/train_mlp_all_%j.log
#SBATCH --error=logs/train_mlp_all_err_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

set -e

# =============================================================================
# Train MLP bbox regressor on pre-extracted embeddings
# Runs ALL configurations: arch (residual/plain) x loss (ciou/smoothl1/iou)
# for both all_token and image_token embeddings in a single sbatch
# =============================================================================

PROJECT_ROOT="/home/hxiaoap/Medical_Image_Analysis_project_2"
ENV_PATH="$HOME/.conda/envs/mia"

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
# Embedding directories (already extracted)
# ============================
EMB_ALLTOKEN="/home/hxiaoap/Medical_Image_Analysis_project_2/embeddings/emb_alltoken"
EMB_IMAGETOKEN="/home/hxiaoap/Medical_Image_Analysis_project_2/embeddings/emb_imagetoken"

# ============================
# Configurations to run
# ============================
ARCHS=(residual plain)
LOSSES=(ciou smoothl1 iou)
EPOCHS=100
BATCH_SIZE=128

TRAIN_START=$(date +%s)

# ============================
# all_token embeddings: 6 configs
# ============================
echo ""
echo "=== Training on emb_alltoken_405274 ==="
echo "Start time: $(date +'%Y-%m-%d %H:%M:%S')"

for arch in "${ARCHS[@]}"; do
  for loss in "${LOSSES[@]}"; do
    echo ""
    echo ">>> arch=$arch loss=$loss"
    python scripts/train_bbox_mlp.py \
      --embedding_dir "$EMB_ALLTOKEN" \
      --arch "$arch" \
      --loss "$loss" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --device cuda \
      --run_name "train_${arch}_${loss}_alltoken"
    echo "<<< done"
  done
done

# ============================
# image_token embeddings: 6 configs
# ============================
echo ""
echo "=== Training on emb_imagetoken_405274 ==="
echo "Start time: $(date +'%Y-%m-%d %H:%M:%S')"

for arch in "${ARCHS[@]}"; do
  for loss in "${LOSSES[@]}"; do
    echo ""
    echo ">>> arch=$arch loss=$loss"
    python scripts/train_bbox_mlp.py \
      --embedding_dir "$EMB_IMAGETOKEN" \
      --arch "$arch" \
      --loss "$loss" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --device cuda \
      --run_name "train_${arch}_${loss}_imagetoken"
    echo "<<< done"
  done
done

TRAIN_END=$(date +%s)
DURATION=$((TRAIN_END - TRAIN_START))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "============================================"
echo "All 12 MLP training configs finished!"
echo "Total duration: ${HOURS}h ${MINUTES}m"
echo "End time: $(date +'%Y-%m-%d %H:%M:%S')"
echo "============================================"