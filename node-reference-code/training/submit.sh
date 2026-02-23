#!/bin/bash
#SBATCH --job-name=emnist-ddpm-train
#SBATCH --partition=hcn1-gpu
#SBATCH --account=pi-hcn1
#SBATCH --qos=hcn1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/log_large.out
#SBATCH --error=logs/err_large.err

# -----------------------------
# MODULES + ENV
# -----------------------------
module load cuda/12.6
module load python/miniforge-25.3.0

eval "$($CONDA_EXE shell.bash hook)"
conda activate superstition-sd

# TORCH_HOME points into the project-level data/ directory.
# The EMNIST dataset will be downloaded here on first run.
# NOTE: BASH_SOURCE doesn't work under SLURM (script is copied to spool).
# Use SLURM_SUBMIT_DIR instead, which is the directory where sbatch was invoked.
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export TORCH_HOME="${PROJECT_ROOT}/data"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# -----------------------------
# WORKING DIR
# -----------------------------
cd "$SCRIPT_DIR"

# -----------------------------
# TRAIN (DDP with 4x L40S)
# Metrics, plots, and report are generated automatically
# at the end of training (see train.py).
# -----------------------------
torchrun --standalone --nproc_per_node=4 train.py
