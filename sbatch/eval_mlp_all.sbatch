#!/bin/bash
#SBATCH --job-name=eval_mlp_all
#SBATCH --output=logs/eval_mlp_all_%j.log
#SBATCH --error=logs/eval_mlp_all_err_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

set -e

# =============================================================================
# Evaluate ALL trained MLP models
# Expects output/train/train_<arch>_<loss>_<embedding>/bbox_mlp.pt
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
# Embedding directories
# ============================
EMB_ALLTOKEN="/home/hxiaoap/Medical_Image_Analysis_project_2/embeddings/emb_alltoken"
EMB_IMAGETOKEN="/home/hxiaoap/Medical_Image_Analysis_project_2/embeddings/emb_imagetoken"

# ============================
# Configurations to evaluate
# ============================
ARCHS=(residual plain)
LOSSES=(ciou smoothl1 iou)

# Embedding types and their variable names for run name construction
EMB_TYPES=("alltoken" "imagetoken")
EMB_DIRS=("$EMB_ALLTOKEN" "$EMB_IMAGETOKEN")

EVAL_START=$(date +%s)

echo ""
echo "=== Evaluating ALL MLP models ==="
echo "Start time: $(date +'%Y-%m-%d %H:%M:%S')"

for emb_idx in 0 1; do
  emb_type="${EMB_TYPES[$emb_idx]}"
  emb_dir="${EMB_DIRS[$emb_idx]}"

  for arch in "${ARCHS[@]}"; do
    for loss in "${LOSSES[@]}"; do
      run_name="train_${arch}_${loss}_${emb_type}"
      ckpt_dir="$PROJECT_ROOT/output/train/$run_name"
      ckpt_path="$ckpt_dir/bbox_mlp.pt"

      if [ -f "$ckpt_path" ]; then
        echo ""
        echo ">>> Evaluating $run_name"
        echo "    ckpt: $ckpt_path"
        echo "    emb:  $emb_dir"
        python scripts/evaluate_bbox.py \
          --model_path "$ckpt_path" \
          --embedding_dir "$emb_dir" \
          --device cuda \
          --run_name "eval_${arch}_${loss}_${emb_type}"
        echo "<<< done"
      else
        echo ""
        echo "!!! Checkpoint not found, skipping: $ckpt_path"
      fi
    done
  done
done

EVAL_END=$(date +%s)
DURATION=$((EVAL_END - EVAL_START))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "============================================"
echo "All MLP evaluations finished!"
echo "Total duration: ${HOURS}h ${MINUTES}m"
echo "End time: $(date +'%Y-%m-%d %H:%M:%S')"
echo "============================================"