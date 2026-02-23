#!/bin/bash
#SBATCH --job-name=gen-starting-imgs
#SBATCH --account=pi-hcn1
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gen_starting_%j.out
#SBATCH --error=logs/gen_starting_%j.err

# ── Environment ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
source activate superstition-sd 2>/dev/null || conda activate superstition-sd

mkdir -p "$SCRIPT_DIR/logs"

# ── Run ──────────────────────────────────────────────────────────────
python -m inference.generate_starting_images
