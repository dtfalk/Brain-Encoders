# EMNIST Diffusion Training Pipeline

Distributed PyTorch training for a pixel-space **class-conditional DDPM** on the EMNIST dataset, using `UNet2DModel` from HuggingFace Diffusers.

- **Author:** David Falk
- **Organization:** APEX Laboratory, The University of Chicago
- **Hardware:** 4× NVIDIA L40S (48 GB each) on SLURM-managed Midway3 HPC

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What This Does](#what-this-does)
3. [Architecture](#architecture)
4. [Training Configuration](#training-configuration)
5. [Model Size Presets](#model-size-presets)
6. [Metrics & Visualization](#metrics--visualization)
7. [Sample Outputs](#sample-outputs)
8. [Subfolder Documentation](#subfolder-documentation)
9. [Key Design Decisions](#key-design-decisions)
10. [Usage](#usage)

---

## Quick Start

```bash
# On SLURM (Midway3)
sbatch submit.sh

# Locally (4 GPUs)
torchrun --standalone --nproc_per_node=4 train.py

# Single GPU (debugging)
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 python train.py
```

---

## What This Does

This pipeline trains a **denoising diffusion probabilistic model (DDPM)** to generate handwritten characters from the EMNIST dataset.

### The Diffusion Process

**Training (forward process):** Gradually add Gaussian noise to real EMNIST images over `T` timesteps until they become pure noise.

**Inference (reverse process):** Start from pure noise and iteratively denoise, guided by a class label, to produce a handwritten character.

```
Training:  clean image  →  noisy image  →  pure noise
                         ε ~ N(0, I) added at each step

Inference: pure noise   →  less noisy   →  clean character
                         UNet predicts ε at each step
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Balanced split** | Equal samples per class (EMNIST raw is heavily skewed toward digits) |
| **Cosine β schedule** | `squaredcos_cap_v2` — better fine-grained detail than linear |
| **EMA weights** | Exponential moving average for smoother, more stable generation |
| **CFG dropout** | 10% label dropout enables classifier-free guidance at inference |
| **DDP training** | PyTorch `DistributedDataParallel` across 4× L40S GPUs |
| **Case merging** | Visually identical uppercase/lowercase pairs merged (e.g., S/s → single class) |

---

## Architecture

```
training/
├── train.py               # Main DDP training script
├── submit.sh              # SLURM submission script (Midway3)
├── pretty_logger.py       # Rich 4-panel live terminal dashboard
│
├── metrics/               # ← Metrics, plots, & report suite
│   ├── __init__.py
│   ├── tracker.py         # MetricsTracker (per-epoch CSV/JSON)
│   ├── plots.py           # 11 publication-quality charts
│   ├── report.py          # Markdown report generator
│   └── README.md          # Detailed metrics documentation
│
└── logs/                  # ← SLURM job logs
    ├── err.err
    ├── log.out
    └── README.md          # Log file descriptions
```

### Module Responsibilities

| File | Purpose |
|------|---------|
| `train.py` | End-to-end training: data loading, model setup, DDP, training loop, EMA, checkpointing |
| `submit.sh` | SLURM job script — 4× L40S, 200G RAM, 32 CPUs, conda activation |
| `pretty_logger.py` | Rich-based live dashboard with 4 panels: log, GPU status, progress bars, epoch history |
| `metrics/tracker.py` | Per-epoch metrics (loss, LR, gradients, throughput, GPU memory, EMA delta) |
| `metrics/plots.py` | 11 matplotlib charts with dark theme, 180 DPI |
| `metrics/report.py` | Markdown training report for HuggingFace model cards |

---

## Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model** | `UNet2DModel` (diffusers) | Pixel-space U-Net for noise prediction |
| **Dataset** | EMNIST Balanced | 47 classes (merged case pairs), equal samples |
| **Epochs** | 50 | Full passes through the dataset |
| **Learning Rate** | 1e-4 | Adam optimizer |
| **Diffusion Steps** | 600 | Number of noise schedule timesteps |
| **β Schedule** | `squaredcos_cap_v2` | Cosine noise schedule |
| **EMA Decay** | 0.999 | Exponential moving average smoothing |
| **CFG Dropout** | 10% | Fraction of labels replaced with null token |
| **DDP Backend** | NCCL | GPU-to-GPU gradient synchronization |

---

## Model Size Presets

| Size | Resolution | Batch Size | Channels | Notes |
|------|-----------|------------|----------|-------|
| `small` | 28×28 | 2048 | (64, 128, 256) | Fast iteration, native EMNIST resolution |
| `medium` | 64×64 | 384 | (64, 128, 256, 256) | Good balance of quality and speed |
| `large` | 96×96 | 96 | (96, 192, 384, 384) | High quality, slower training |
| `xl` | 128×128 | 32 | (128, 256, 512, 512) | Maximum quality, significant GPU memory |

Change the preset in `train.py`:
```python
SIZE_NAME = "small"   # or "medium", "large", "xl"
```

---

## Metrics & Visualization

Training automatically generates comprehensive metrics and plots. See [`metrics/README.md`](metrics/README.md) for full details.

### What Gets Tracked

- **Loss:** mean, std, min, max, median per epoch
- **Gradients:** global L2 norm — detects explosions/vanishing
- **Throughput:** samples/sec and batches/sec
- **GPU Memory:** allocated, reserved, peak (MB)
- **EMA Divergence:** L2 distance between live and shadow weights
- **Timing:** per-epoch and cumulative wall clock

### Generated Artifacts

| Artifact | Format | Description |
|----------|--------|-------------|
| `training_metrics.csv` | CSV | One row per epoch, machine-readable |
| `training_metrics.json` | JSON | Full run snapshot with config |
| `training_report.md` | Markdown | HuggingFace model card |
| 11 plot PNGs | PNG (180 DPI) | Publication-quality charts |

---

## Sample Outputs

> **Note:** Images below are generated after training completes. Placeholders shown until then.

### Training Dashboard

![Training Dashboard](metrics/training_dashboard.png)

*3×3 multi-panel overview: loss curves, learning rate, gradient norms, throughput, GPU memory, EMA divergence, epoch timing, and loss derivative.*

### Loss Curves

| Linear Scale | Log Scale | Distribution (mean ± σ) |
|-------------|-----------|------------------------|
| ![Loss Linear](metrics/loss_linear.png) | ![Loss Log](metrics/loss_log.png) | ![Loss Distribution](metrics/loss_distribution.png) |

*Left: training loss with best-epoch marker (★). Center: log scale for early-epoch detail. Right: mean ± 1σ shaded region.*

### Gradient & EMA Health

| Gradient Norm | EMA Divergence |
|---------------|----------------|
| ![Grad Norm](metrics/grad_norm.png) | ![EMA Divergence](metrics/ema_divergence.png) |

*Left: gradient magnitude per epoch — spikes indicate instability. Right: divergence between live weights and EMA shadow.*

### GPU Utilization

| Memory Usage | Throughput |
|-------------|------------|
| ![GPU Memory](metrics/gpu_memory.png) | ![Throughput](metrics/throughput.png) |

*Left: CUDA memory (allocated/reserved/peak) across epochs. Right: training throughput in samples/sec.*

### Generated Characters (After Inference)

> Below are example outputs from the **inference engine** ([`inference/`](../inference/)) using a trained checkpoint.

| | | | | |
|---|---|---|---|---|
| ![A](../output/A/bw_frames/bw_v000_f0599.png) | ![B](../output/B/bw_frames/bw_v000_f0599.png) | ![C](../output/C/bw_frames/bw_v000_f0599.png) | ![k](../output/k/bw_frames/bw_v000_f0599.png) | ![3](../output/3/bw_frames/bw_v000_f0599.png) |
| A | B | C | k | 3 |

*Final denoised frames showing generated handwritten characters. Each image is the result of 600 diffusion steps from pure noise.*

### Denoising Progression (Video Frames)

| Step 0 (Noise) | Step 100 | Step 300 | Step 500 | Step 599 (Final) |
|----------------|----------|----------|----------|------------------|
| ![s0](../output/A/bw_frames/bw_v000_f0000.png) | ![s100](../output/A/bw_frames/bw_v000_f0100.png) | ![s300](../output/A/bw_frames/bw_v000_f0300.png) | ![s500](../output/A/bw_frames/bw_v000_f0500.png) | ![s599](../output/A/bw_frames/bw_v000_f0599.png) |

*The diffusion reverse process: pure Gaussian noise progressively resolves into a handwritten "A".*

---

## Subfolder Documentation

| Subfolder | README | Description |
|-----------|--------|-------------|
| `metrics/` | [`metrics/README.md`](metrics/README.md) | Per-epoch metrics tracking, 11 plots, markdown report, CSV/JSON persistence |
| `logs/` | [`logs/README.md`](logs/README.md) | SLURM job output logs and archived training logs |

---

## Key Design Decisions

### Why Balanced Split?

Raw EMNIST is heavily skewed — digits have far more samples than letters. Without balancing, the model would disproportionately learn digit classes. The balanced split ensures equal representation, preventing any class from dominating the loss.

### Why Cosine Schedule?

The `squaredcos_cap_v2` schedule adds noise more gradually at both ends of the diffusion process compared to a linear schedule. This preserves fine-grained details (thin strokes, serifs) that are critical for distinguishing similar characters (e.g., `l` vs `I`).

### Why EMA?

Exponential moving average weights produce smoother, more stable generated images. The EMA shadow tracks a running average of model parameters with decay 0.999, dampening the effect of noisy gradient updates.

### Why CFG Dropout?

Classifier-free guidance requires the model to handle both conditional (with class label) and unconditional (null label) predictions. By randomly dropping 10% of labels during training, the model learns to predict noise in both regimes, enabling adjustable guidance strength at inference time.

### Why Case Merging?

Characters like `S/s`, `C/c`, `O/o` are visually identical in handwriting. Training separate classes for them would force the model to learn an arbitrary distinction. Merging them improves sample quality and reduces the class count from 62 to 47.

---

## Usage

### SLURM Submission

```bash
cd training
sbatch submit.sh
```

The `submit.sh` script requests:
- 4× NVIDIA L40S GPUs
- 200 GB RAM, 32 CPUs
- 8-hour time limit
- Conda environment `superstition-sd`

### Monitoring

While training runs, the Rich dashboard shows live progress:
- Training log with color-coded severity
- Per-GPU utilization, temperature, and memory
- Epoch and batch progress bars
- Rolling table of recent epoch summaries

### Output Structure

Checkpoints and metrics are saved to:
```
emnist-ddpm/
└── checkpoints/
    └── small/
        ├── emnist_balanced_cfg_cosine_ema_600_steps_epoch{N}.pt
        └── ...
└── training/
    └── metrics/
        └── output/
            └── balanced_cfg_cosine_ema_600_steps/
                ├── training_metrics.csv
                ├── training_metrics.json
                ├── training_report.md
                └── *.png (11 plot files)
```

---

## Citation

```bibtex
@software{emnist_ddpm_training,
    title = {EMNIST Diffusion Training Pipeline},
    author = {David Falk},
    organization = {APEX Laboratory, The University of Chicago},
    year = {2026},
    url = {https://github.com/your-repo}
}
```
