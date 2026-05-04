#!/bin/bash
#SBATCH --job-name=vlm_bases
#SBATCH --output=logs/vlm_bases_%j.log
#SBATCH --error=logs/vlm_bases_err_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

set -e

# =============================================================================
# Zero-shot + Few-shot (n=3) evaluation with Qwen2-VL
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
# Parameters
# ============================
EMBEDDING_DIR="${EMBEDDING_DIR:-/home/hxiaoap/Medical_Image_Analysis_project_2/embeddings/emb_alltoken}"
MODEL="./models/Qwen2-VL-2B-Instruct"

echo ""
echo "=== Zero-shot Evaluation ==="
echo "embedding_dir: $EMBEDDING_DIR"
echo "Start time: $(date +'%Y-%m-%d %H:%M:%S')"

python scripts/evaluate_few_shot.py \
  --embedding_dir "$EMBEDDING_DIR" \
  --model "$MODEL" \
  --n_shot 0 \
  --device cuda \
  --run_name "vlm_zeroshot"

echo ""
echo "=== Few-shot Evaluation (n=3) ==="
echo "Start time: $(date +'%Y-%m-%d %H:%M:%S')"

python scripts/evaluate_few_shot.py \
  --embedding_dir "$EMBEDDING_DIR" \
  --model "$MODEL" \
  --n_shot 3 \
  --device cuda \
  --run_name "vlm_fewshot3"

echo ""
echo "End time: $(date +'%Y-%m-%d %H:%M:%S')"
echo "=== Done ==="