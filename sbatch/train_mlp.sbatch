#!/bin/bash
#SBATCH --job-name=train_mlp
#SBATCH --output=logs/train_mlp_%j.log
#SBATCH --error=logs/train_mlp_err_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

set -e

# =============================================================================
# Train MLP bbox regressor on pre-extracted embeddings
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
# Parameters (with defaults)
# ============================
EMBEDDING_DIR="${EMBEDDING_DIR:-/home/hxiaoap/Medical_Image_Analysis_project_2/embeddings/emb_alltoken}"
ARCH="${ARCH:-residual}"
LOSS_NAME="${LOSS_NAME:-ciou}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"

echo ""
echo "=== MLP Training ==="
echo "embedding_dir: $EMBEDDING_DIR"
echo "arch:          $ARCH"
echo "loss:          $LOSS_NAME"
echo "epochs:        $EPOCHS"
echo "batch_size:    $BATCH_SIZE"
echo "Start time:    $(date +'%Y-%m-%d %H:%M:%S')"

python scripts/train_bbox_mlp.py \
  --embedding_dir "$EMBEDDING_DIR" \
  --arch "$ARCH" \
  --loss "$LOSS_NAME" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --device cuda \
  --run_name "train_${ARCH}_${LOSS_NAME}"

echo ""
echo "End time: $(date +'%Y-%m-%d %H:%M:%S')"
echo "=== Done ==="