#!/bin/bash
#SBATCH --job-name=eval_mlp
#SBATCH --output=logs/eval_mlp_%j.log
#SBATCH --error=logs/eval_mlp_err_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

set -e

# =============================================================================
# Evaluate MLP bbox regressor
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
MODEL_CKPT="${MODEL_CKPT:-/home/hxiaoap/Medical_Image_Analysis_project_2/output/train/train_residual_ciou/bbox_mlp.pt}"
EMBEDDING_DIR="${EMBEDDING_DIR:-/home/hxiaoap/Medical_Image_Analysis_project_2/embeddings/emb_alltoken}"

echo ""
echo "=== MLP Evaluation ==="
echo "model_ckpt:     $MODEL_CKPT"
echo "embedding_dir:  $EMBEDDING_DIR"
echo "Start time:     $(date +'%Y-%m-%d %H:%M:%S')"

python scripts/evaluate_bbox.py \
  --model_path "$MODEL_CKPT" \
  --embedding_dir "$EMBEDDING_DIR" \
  --device cuda \
  --run_name "eval_residual_ciou"

echo ""
echo "End time: $(date +'%Y-%m-%d %H:%M:%S')"
echo "=== Done ==="