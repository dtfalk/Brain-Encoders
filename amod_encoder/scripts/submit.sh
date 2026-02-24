#!/bin/bash
# =============================================================================
# SLURM submission script — AMOD Encoder Pipeline
# Cluster: Midway3, partition hcn1-gpu, node midway3-0427
# Node:    4× L40S (48 GB VRAM each), 32 CPU cores, 1 TB RAM
#
# Usage (run from Brain-Encoders/amod_encoder/):
#   sbatch --export=STAGE=amygdala,STEP=fit     scripts/submit.sh
#   sbatch --export=STAGE=amygdala,STEP=eval    scripts/submit.sh
#   sbatch --export=STAGE=amygdala,STEP=iaps    scripts/submit.sh
#   sbatch --export=STAGE=amygdala,STEP=betas   scripts/submit.sh
#
#   # Or run the full replication pipeline in one job:
#   sbatch scripts/submit.sh
#
# STAGE choices:  amygdala | subregions | visual_cortex | inferotemporal
# STEP  choices:  fit | eval | iaps | betas | all
# =============================================================================

#SBATCH --job-name=amod-encoder
#SBATCH --partition=hcn1-gpu
#SBATCH --account=pi-hcn1
#SBATCH --qos=hcn1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/amod_%j_%x.out
#SBATCH --error=logs/amod_%j_%x.err
#SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your@email.uchicago.edu

# --------------------------------------------------------------------------
# Environment (matches node-reference-code pattern)
# --------------------------------------------------------------------------
module load cuda/12.6
module load python/miniforge-25.3.0

eval "$($CONDA_EXE shell.bash hook)"
conda activate brain-encoders

# SLURM_SUBMIT_DIR = directory where sbatch was invoked (amod_encoder/)
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export TORCH_HOME="${PROJECT_ROOT}/amod_encoder/data/emonet_weights"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export JOBLIB_TEMP_FOLDER=/tmp

# --------------------------------------------------------------------------
# Working dir
# --------------------------------------------------------------------------
cd "$SCRIPT_DIR"
mkdir -p logs

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
STAGE="${STAGE:-amygdala}"
STEP="${STEP:-all}"

CFG="configs/cluster/${STAGE}.yaml"

echo "============================================================"
echo "SLURM job $SLURM_JOB_ID | node $SLURM_NODELIST"
echo "Stage: $STAGE  |  Step: $STEP  |  Config: $CFG"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "============================================================"

# --------------------------------------------------------------------------
# Pipeline steps
# --------------------------------------------------------------------------
run_fit() {
    echo "[$(date)] Starting: fit"
    amod-encoder fit -c "$CFG"
    echo "[$(date)] Done: fit"
}

run_eval() {
    echo "[$(date)] Starting: eval"
    amod-encoder eval -c "$CFG"
    echo "[$(date)] Done: eval"
}

run_iaps() {
    # Only amygdala and visual_cortex have iaps_csv / oasis_csv
    if [[ "$STAGE" == "amygdala" || "$STAGE" == "visual_cortex" ]]; then
        echo "[$(date)] Starting: predict-iaps-oasis"
        amod-encoder predict-iaps-oasis -c "$CFG"
        echo "[$(date)] Done: predict-iaps-oasis"
    else
        echo "[skip] predict-iaps-oasis not applicable for stage: $STAGE"
    fi
}

run_betas() {
    echo "[$(date)] Starting: export-betas"
    amod-encoder export-betas -c "$CFG"
    echo "[$(date)] Done: export-betas"
}

run_compile() {
    echo "[$(date)] Starting: compile-atanh"
    amod-encoder compile-atanh -c "$CFG"
    echo "[$(date)] Done: compile-atanh"
}

# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
case "$STEP" in
    fit)    run_fit ;;
    eval)   run_eval ;;
    iaps)   run_iaps ;;
    betas)  run_betas ;;
    all)
        run_fit
        run_eval
        run_compile
        run_betas
        run_iaps
        ;;
    *)
        echo "Unknown STEP: $STEP  (choices: fit | eval | iaps | betas | all)"
        exit 1
        ;;
esac

echo "============================================================"
echo "Job complete: $SLURM_JOB_ID"
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,AllocGRES --noheader
echo "============================================================"
