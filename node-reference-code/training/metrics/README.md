# Training Metrics

Comprehensive metrics collection, CSV/JSON persistence, markdown report generation, and publication-quality plot suite for the EMNIST DDPM training pipeline.

---

## Overview

This module is wired into `train.py` and records per-epoch training metrics. Data flows through three stages:

```
Training Loop → MetricsTracker → CSV / JSON / Plots / Report
```

All metrics are collected **only on rank 0** in distributed training — no redundant writes from worker GPUs.

---

## Files

| File | Purpose | Key Exports |
|------|---------|-------------|
| [`tracker.py`](tracker.py) | `MetricsTracker` — per-epoch metrics collection & CSV/JSON writing | `MetricsTracker`, `EpochMetrics`, `compute_grad_norm()` |
| [`plots.py`](plots.py) | 11 matplotlib charts | `generate_all_plots()` |
| [`report.py`](report.py) | Markdown training report | `generate_report()` |
| [`__init__.py`](__init__.py) | Package exports | `MetricsTracker`, `generate_all_plots`, `generate_report` |

---

## Tracked Metrics (`EpochMetrics`)

| Category | Metrics | Description |
|----------|---------|-------------|
| **Loss** | `loss_mean`, `loss_std`, `loss_min`, `loss_max`, `loss_median` | Full loss distribution per epoch |
| **Learning Rate** | `lr` | Current optimizer LR |
| **Gradient Norm** | `grad_norm` | Global L2 norm of all model gradients |
| **Throughput** | `samples_per_sec`, `batches_per_sec` | Training speed |
| **GPU Memory** | `gpu_allocated_mb`, `gpu_reserved_mb`, `gpu_peak_mb` | CUDA memory usage |
| **EMA** | `ema_delta_norm` | L2 distance between live weights and EMA shadow |
| **Timing** | `epoch_time_sec`, `cumulative_time_sec` | Wall-clock timing |

---

## Generated Plots

All plots use a dark theme (`#0e1117` background) at 180 DPI — suitable for papers, presentations, and HuggingFace model cards.

| # | Plot | Description |
|---|------|-------------|
| 1 | **Loss (Linear)** | Epoch loss with min/max band and best-epoch marker |
| 2 | **Loss (Log Scale)** | Same data, log y-axis for early-epoch detail |
| 3 | **Loss Distribution** | Mean ± 1σ shaded region |
| 4 | **Learning Rate** | LR schedule over epochs |
| 5 | **Gradient Norm** | Gradient magnitude — useful for detecting explosions |
| 6 | **Throughput** | Samples/sec bar chart per epoch |
| 7 | **GPU Memory** | Allocated / reserved / peak memory lines |
| 8 | **EMA Divergence** | Distance between live and EMA weights |
| 9 | **Epoch Timing** | Wall-clock seconds per epoch |
| 10 | **Loss Derivative** | Green (improving) vs red (worsening) per epoch |
| 11 | **Training Dashboard** | 3×3 multi-panel combining key charts |

### Example Outputs

> **Note:** Plots are generated at the end of training. Placeholders shown until you run the pipeline.

#### Training Dashboard
![Training Dashboard](training_dashboard.png)
*3×3 overview: loss, LR, gradients, throughput, GPU memory, EMA divergence, timing, and loss derivative.*

#### Loss Curve
![Loss Linear](loss_linear.png)
*Training loss with best-epoch marker (★) and min/max band.*

#### Gradient Norm
![Gradient Norm](grad_norm.png)
*Global L2 gradient norm per epoch. Spikes indicate potential instability.*

#### GPU Memory
![GPU Memory](gpu_memory.png)
*CUDA memory allocation over training. Useful for capacity planning on multi-GPU nodes.*

---

## Markdown Report (`report.py`)

Generates a complete `training_report.md` with:

- **Summary table** — best loss, total epochs, wall time, GPU peak
- **Hyperparameters** — full JSON config block
- **Per-epoch table** — all metrics in tabular form
- **Embedded plots** — references to generated chart images
- **Artifacts manifest** — list of all output files

Designed for use as a **HuggingFace model card** or supplementary material in papers.

---

## Usage

Metrics are automatically wired into `train.py`. No manual setup needed for standard training runs.

```python
# Programmatic usage (if integrating into a custom loop)
from metrics import MetricsTracker, generate_all_plots, generate_report
from metrics.tracker import compute_grad_norm

tracker = MetricsTracker(output_dir="./my_metrics")

for epoch in range(num_epochs):
    tracker.start_epoch(epoch)
    
    for batch in dataloader:
        loss = train_step(batch)
        grad_norm = compute_grad_norm(model)
        tracker.log_batch(loss=loss.item(), lr=optimizer.param_groups[0]["lr"], grad_norm=grad_norm)
    
    tracker.end_epoch(model=model, ema_model=ema_model)

# Generate outputs
plot_paths = generate_all_plots(tracker, save_dir="./my_metrics")
generate_report(tracker, config=TRAINING_CONFIG, save_dir="./my_metrics", plot_paths=plot_paths)
```

---

## Output Files

After training completes, the metrics directory contains:

```
metrics/
├── training_metrics.csv       # One row per epoch
├── training_metrics.json      # Full run snapshot
├── training_report.md         # Markdown report (HuggingFace model card)
├── loss_linear.png            # Plot
├── loss_log.png               # Plot
├── loss_distribution.png      # Plot
├── learning_rate.png          # Plot
├── grad_norm.png              # Plot
├── throughput.png             # Plot
├── gpu_memory.png             # Plot
├── ema_divergence.png         # Plot
├── epoch_timing.png           # Plot
├── loss_derivative.png        # Plot
└── training_dashboard.png     # Plot (3×3 multi-panel)
```
