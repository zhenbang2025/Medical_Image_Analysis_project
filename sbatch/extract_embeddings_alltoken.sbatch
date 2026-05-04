#!/bin/bash
#SBATCH --job-name=emb_alltok
#SBATCH --output=logs/emb_alltok_%j.log
#SBATCH --error=logs/emb_alltok_err_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

set -e

# =============================================================================
# Extract embeddings with Qwen2-VL (all_token_mean pooling only)
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

echo ""
echo "=== Extract Embeddings (all_token_mean) ==="
echo "Start time: $(date +'%Y-%m-%d %H:%M:%S')"

python scripts/extract_bbox_embeddings.py \
  --data_root ./data/3 \
  --bbox_csv ./data/3/BBox_List_2017.csv \
  --model ./models/Qwen2-VL-2B-Instruct \
  --embedding_mode all_token_mean \
  --max_per_class 2200 \
  --device cuda \
  --run_name "emb_alltoken_${SLURM_JOB_ID}"

echo ""
echo "End time: $(date +'%Y-%m-%d %H:%M:%S')"
echo "=== Done ==="