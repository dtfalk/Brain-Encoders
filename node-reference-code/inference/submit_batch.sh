#!/bin/bash
#SBATCH --job-name=emnist-batch
#SBATCH --partition=hcn1-gpu
#SBATCH --account=pi-hcn1
#SBATCH --qos=hcn1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/batch_%A_%a.out
#SBATCH --error=logs/batch_%A_%a.err
#
# Distributed usage (optional):
#   sbatch --array=0-3 submit_batch.sh   # 4 workers, each handles 1/4 of videos
#
# Single-process usage:
#   sbatch submit_batch.sh               # one worker generates all videos

# -----------------------------
# MODULES + ENV
# -----------------------------
module load cuda/12.6
module load python/miniforge-25.3.0

eval "$($CONDA_EXE shell.bash hook)"
conda activate superstition-sd

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# Fallback for SLURM (BASH_SOURCE may not work in spool)
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"

# Project root is one level up from inference/
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export TORCH_HOME="${PROJECT_ROOT}/data"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -----------------------------
# WORKING DIR
# -----------------------------
cd "$PROJECT_ROOT"
mkdir -p logs

# -----------------------------
# RUN BATCH
# -----------------------------
echo "Starting batch orchestrator at $(date)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-not set}"
echo "SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-not set}"
python -m inference.run_batch --batch-size 64
echo "Batch orchestrator finished at $(date)"
