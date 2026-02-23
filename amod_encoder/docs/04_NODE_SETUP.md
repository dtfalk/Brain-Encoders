# Node Setup — 4× NVIDIA L40S (Slurm)

This guide covers running the Brain-Encoders pipeline on a compute node
with **4× NVIDIA L40S GPUs** (Ada Lovelace, 48 GB VRAM each) under a
**Slurm** job scheduler.

> **Prerequisites:**
> - Complete the [Data Setup Guide](01_DATA_SETUP.md) first.
> - Read the [General Usage Guide](02_GENERAL_USAGE.md) for pipeline
>   commands and output structure.

---

## 1. Hardware Summary

| Component | Spec |
|-----------|------|
| GPUs | 4× NVIDIA L40S (Ada Lovelace, 48 GB VRAM each, 192 GB total) |
| Architecture | sm_89 |
| CUDA | 12.6 |
| System RAM | 1 TB |
| PyTorch | Stable release (L40S is fully supported) |
| Scheduler | Slurm |

---

## 2. Environment Setup

### Create the Conda Environment

```bash
cd Brain-Encoders/amod_encoder

conda env create -f environment-node.yml
conda activate brain-encoders-node
```

This installs:
- **Python 3.11**
- **PyTorch stable** with CUDA 12.6
  (`--extra-index-url https://download.pytorch.org/whl/cu126`)
- **torchvision ≥ 0.20**
- **timm ≥ 1.0** + **transformers ≥ 4.40**
- **pynvml ≥ 12.0** (GPU monitoring for Rich dashboard)
- **accelerate ≥ 0.30** (multi-GPU support)
- All scientific and neuroimaging dependencies

### Install the Package

```bash
pip install -e ".[all]"
```

### Verify GPU Access

```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.0f} GB)')
"
```

Expected output:

```
CUDA available: True
GPU count: 4
  GPU 0: NVIDIA L40S (48 GB)
  GPU 1: NVIDIA L40S (48 GB)
  GPU 2: NVIDIA L40S (48 GB)
  GPU 3: NVIDIA L40S (48 GB)
```

---

## 3. Data Paths on the Node

On a shared cluster, data typically lives on a network filesystem. Update
your config paths accordingly:

```yaml
paths:
  bids_root: /scratch/your_username/data/ds002837
  osf_fc7_mat: /scratch/your_username/data/osf/500_days_of_summer_fc7_features.mat
  output_dir: /scratch/your_username/output/amygdala
  iaps_csv: /scratch/your_username/data/osf/IAPS_data_amygdala_z.csv
  oasis_csv: /scratch/your_username/data/osf/OASIS_data_amygdala_z.csv
```

> **Tip:** Use `/scratch/` or a fast filesystem for data. NFS home
> directories can be slow for large NIfTI I/O.

---

## 4. Configuration for GPU

### Single-GPU Config

```yaml
compute:
  backend: torch
  device: cuda           # Uses GPU 0
  amp: false             # Set true for faster Ridge (minor precision loss)
```

### Selecting a Specific GPU

```yaml
compute:
  backend: torch
  device: cuda:2         # Use GPU 2 specifically
  amp: false
```

Or set the environment variable before running:

```bash
CUDA_VISIBLE_DEVICES=2 amod-encoder fit --config configs/amod_amygdala.yaml
```

---

## 5. Slurm Job Scripts

### Basic Single-GPU Job

```bash
#!/bin/bash
#SBATCH --job-name=amod-fit
#SBATCH --output=logs/amod-fit-%j.out
#SBATCH --error=logs/amod-fit-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# --- Environment ---
module load cuda/12.6          # adjust to your cluster's module name
conda activate brain-encoders-node

cd /path/to/Brain-Encoders/amod_encoder

# --- Run ---
echo "Starting fit at $(date)"
amod-encoder fit  --config configs/amod_amygdala.yaml
echo "Fit complete at $(date)"

echo "Starting eval at $(date)"
amod-encoder eval --config configs/amod_amygdala.yaml
echo "Eval complete at $(date)"

echo "Starting validate at $(date)"
amod-encoder validate --config configs/amod_amygdala.yaml
echo "All done at $(date)"
```

Submit with:

```bash
mkdir -p logs
sbatch scripts/slurm_fit.sh
```

### Full Pipeline (Amygdala + Subregions)

```bash
#!/bin/bash
#SBATCH --job-name=amod-full
#SBATCH --output=logs/amod-full-%j.out
#SBATCH --error=logs/amod-full-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00

module load cuda/12.6
conda activate brain-encoders-node

cd /path/to/Brain-Encoders/amod_encoder

# Whole amygdala
amod-encoder fit      --config configs/amod_amygdala.yaml
amod-encoder eval     --config configs/amod_amygdala.yaml
amod-encoder validate --config configs/amod_amygdala.yaml

# Subregions
amod-encoder fit  --config configs/amod_subregions.yaml
amod-encoder eval --config configs/amod_subregions.yaml

# Cross-reference subregions against whole amygdala
amod-encoder compile-atanh \
  --config configs/amod_subregions.yaml \
  --parent-mask /scratch/$USER/data/masks/canlab2018_amygdala_combined.nii.gz

# Predictions
amod-encoder predict-iaps-oasis --config configs/amod_amygdala.yaml

# Export
amod-encoder export-betas --config configs/amod_amygdala.yaml
amod-encoder export-betas --config configs/amod_subregions.yaml

echo "Full pipeline complete at $(date)"
```

### Per-Subject Parallel Jobs (Array Job)

For large-scale runs, submit one job per subject to maximise throughput:

```bash
#!/bin/bash
#SBATCH --job-name=amod-sub
#SBATCH --output=logs/amod-sub-%A_%a.out
#SBATCH --error=logs/amod-sub-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --array=1-20

module load cuda/12.6
conda activate brain-encoders-node

cd /path/to/Brain-Encoders/amod_encoder

# Zero-pad subject ID (01, 02, ..., 20)
SUB_ID=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

amod-encoder fit --config configs/amod_amygdala.yaml --subjects $SUB_ID
```

Submit with:

```bash
sbatch scripts/slurm_array.sh
```

Then run `eval` and `validate` once all array jobs finish:

```bash
sbatch --dependency=afterok:<ARRAY_JOB_ID> scripts/slurm_eval.sh
```

---

## 6. Feature Extraction at Scale

The L40S's 48 GB VRAM allows large batch sizes for feature extraction:

```bash
# Extract DINOv2 features from all movie frames
amod-encoder extract-features \
  -c configs/dinov2_extractor.yaml \
  --source /data/500_days_of_summer.mp4 \
  --output output/movie_dinov2.npy \
  --batch-size 128
```

With 48 GB VRAM, batch sizes of 64–256 are comfortable for most vision
models (CLIP, DINOv2, timm ViT variants).

### Multi-GPU Feature Extraction

For very large image sets, you can split across GPUs manually:

```bash
# GPU 0: first half
CUDA_VISIBLE_DEVICES=0 amod-encoder extract-features \
  -c configs/clip.yaml --source /data/images_part1/ \
  --output output/features_part1.npy &

# GPU 1: second half
CUDA_VISIBLE_DEVICES=1 amod-encoder extract-features \
  -c configs/clip.yaml --source /data/images_part2/ \
  --output output/features_part2.npy &

wait
```

---

## 7. GPU Monitoring

The node environment includes `pynvml` for GPU monitoring. You can check
GPU utilisation alongside your jobs:

```bash
# In a separate terminal / tmux pane
watch -n 2 nvidia-smi

# Or use pynvml programmatically
python -c "
import pynvml
pynvml.nvmlInit()
for i in range(pynvml.nvmlDeviceGetCount()):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    util = pynvml.nvmlDeviceGetUtilizationRates(h)
    print(f'GPU {i}: {info.used/1e9:.1f}/{info.total/1e9:.0f} GB, util={util.gpu}%')
"
```

---

## 8. Performance Expectations

On an L40S with `backend: torch`, `device: cuda`:

| Step | Time |
|------|------|
| `fit` (PLS, whole amygdala, 20 subjects) | ~40–60 min (PLS is CPU-bound) |
| `fit` (Ridge, whole amygdala, 20 subjects, GPU) | ~5–10 min |
| `eval` (all subjects) | ~30 sec |
| `extract-features` (CLIP, 10k images, batch=128) | ~2 min |
| `extract-features` (DINOv2-L, 10k images, batch=64) | ~4 min |

> **Note:** PLS models always run on CPU (SIMPLS matches MATLAB exactly).
> GPU acceleration benefits Ridge and feature extraction.

---

## 9. Tips for Cluster Usage

- **Use `/scratch/`** — Write outputs to fast local/scratch storage, not
  NFS home directories.
- **Request enough RAM** — Each BOLD NIfTI is ~1–2 GB in memory.
  `--mem=64G` is safe for all 20 subjects sequentially.
- **Set `--time` conservatively** — A full `fit` + `eval` + `validate` for
  both amygdala and subregions takes ~2–3 hours on CPU, ~1 hour with GPU
  Ridge.
- **Check your CUDA module** — Use `module avail cuda` to find the right
  module name. You need CUDA 12.6+.
- **Logs** — Slurm logs (`logs/*.out`) contain the Rich-formatted console
  output. Use `cat` or `less -R` to view with colour.

---

## 10. Troubleshooting

| Problem | Solution |
|---------|----------|
| `module load cuda/12.6` fails | Use `module avail cuda` to find available versions (12.4+ works for L40S) |
| `torch.cuda.is_available()` is `False` inside Slurm job | Add `module load cuda/12.6` before conda activate in your script |
| Job killed by OOM | Increase `--mem` or reduce subjects per job (use array jobs) |
| `NCCL error` with multi-GPU | Encoding pipeline is single-GPU by design; only `extract-features` benefits from multi-GPU |
| `conda activate` fails in Slurm | Add `source $(conda info --base)/etc/profile.d/conda.sh` before `conda activate` |
| Permission denied on `/scratch/` | Ask your cluster admin for a scratch allocation |
