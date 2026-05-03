#!/bin/bash
#SBATCH --job-name=mi_all
#SBATCH --output=logs/mi_all_%j.log
#SBATCH --error=logs/mi_all_err_%j.log
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

set -e

# =============================================================================
# Run all experiments end-to-end
#
# 提交:  sbatch sbatch/run_all_experiments.sbatch
# 查看:  squeue -u $USER
# 日志:  tail -f logs/mi_all_*.log
# 取消:  scancel <job_id>
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

if [[ "$(python --version)" != *"3.10"* ]]; then
  echo "Python version mismatch. Expected 3.10"
  exit 1
fi
echo "Environment check passed: Python 3.10"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# ============================
# 2) Run all experiments
# ============================

echo ""
echo "=== Running All Experiments ==="
echo "Start time: $(date +'%Y-%m-%d %H:%M:%S')"

TRAIN_START=$(date +%s)

bash scripts/run_all_experiments.sh

RET_CODE=$?

TRAIN_END=$(date +%s)
DURATION=$((TRAIN_END - TRAIN_START))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "============================================"
if [ $RET_CODE -eq 0 ]; then
    echo "All experiments finished successfully!"
else
    echo "Experiments failed with code: $RET_CODE"
fi
echo "End time: $(date +'%Y-%m-%d %H:%M:%S')"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "============================================"