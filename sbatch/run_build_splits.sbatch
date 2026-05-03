#!/bin/bash
#SBATCH --job-name=mi_splits
#SBATCH --output=logs/mi_splits_%j.log
#SBATCH --error=logs/mi_splits_err_%j.log
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

set -e

# ============================
# 1) Environment
# ============================

PROJECT_ROOT="/home/hxiaoap/Medical_Image_Analysis_project"
ENV_PATH="/home/hxiaoap/.conda/envs/mia"

cd "$PROJECT_ROOT"
mkdir -p logs

module purge
module load Anaconda3/2023.09-0

export PATH="$ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PATH/lib:$LD_LIBRARY_PATH"

echo "=== Environment Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Python:    $(which python)"
echo "Version:   $(python --version)"
python --version

# ============================
# 2) Build splits
# ============================

echo ""
echo "=== Building Splits ==="
echo "Start time: $(date +'%Y-%m-%d %H:%M:%S')"

python scripts/build_splits.py \
  --dataset_dir ./data/chest_xray \
  --val_fraction 0.1 \
  --seed 42 \
  --output_file ./splits/full_split.json

echo "End time: $(date +'%Y-%m-%d %H:%M:%S')"
echo "=== Done ==="