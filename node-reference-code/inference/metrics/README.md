# Inference Metrics

Automated metrics collection, CSV/JSON persistence, and publication-quality plot generation for the EMNIST diffusion inference engine.

---

## Overview

This module is wired into `run_inference.py` and records data at two granularities:

| Granularity | File | Description |
|-------------|------|-------------|
| **Per-video** | `video_metrics.csv` | One row per generated video — timing, seed, L2 totals, final image stats, GPU peak |
| **Per-step** | `step_metrics.csv` | One row per diffusion step — latent/pixel deltas, scheduler scalars, wall-clock timing |

A combined JSON snapshot (`inference_metrics.json`) is rewritten after every video completes, providing a single-file archive of the full run.

---

## Files

| File | Purpose |
|------|---------|
| [`tracker.py`](tracker.py) | `InferenceMetricsTracker` — records per-video & per-step metrics, writes CSV/JSON |
| [`plots.py`](plots.py) | `generate_all_inference_plots()` — 9 publication-quality matplotlib charts |
| [`__init__.py`](__init__.py) | Package exports |

---

## Tracked Metrics

### Video-Level (`VideoMetrics`)

| Metric | Field | Unit |
|--------|-------|------|
| Video ID | `vid_id` | int |
| Seed | `seed` | int |
| Class label | `label` / `character` | int / str |
| Generation time | `generation_time_sec` | seconds |
| Total steps | `total_steps` | int |
| Captured frames | `captured_frames` | int |
| Final image mean/std/min/max | `final_mean`, `final_std`, … | float |
| Total latent L2 displacement | `total_latent_l2` | float |
| Total pixel BW L2 displacement | `total_pixel_bw_l2` | float |
| Total pixel RGB L2 displacement | `total_pixel_rgb_l2` | float |
| GPU peak memory | `gpu_mem_peak_mb` | MB |

### Step-Level (`StepMetrics`)

| Metric | Field | Unit |
|--------|-------|------|
| Step index | `step_index` | int |
| Timestep | `timestep` | int |
| Wall clock delta | `wall_dt` | seconds |
| Latent delta L2 | `delta_x_l2` | float |
| Latent delta stats | `delta_x_mean/std/min/max` | float |
| Pixel BW delta L2 | `delta_bw_l2` | float |
| Pixel RGB delta L2 | `delta_rgb_l2` | float |
| α_t / β_t / ᾱ_t | `alpha_t`, `beta_t`, `alpha_cumprod_t` | float |

---

## Generated Plots

All plots use a dark theme (`#0e1117` background) at 180 DPI, suitable for papers and HuggingFace model cards.

| # | Plot | Description |
|---|------|-------------|
| 1 | **Latent L2 per Step** | L2 norm of Δx across diffusion steps (per video, overlaid) |
| 2 | **Pixel BW L2 per Step** | BW pixel delta L2 across steps |
| 3 | **Schedule Overlay** | Dual-axis α_cumprod and β_t vs step index |
| 4 | **Generation Times** | Bar chart of wall-clock time per video |
| 5 | **Final Image Stats** | Grouped bars of mean/std/min/max for each video |
| 6 | **Cumulative L2** | Running sum of latent L2 displacement over steps |
| 7 | **L2 Velocity** | Derivative of cumulative L2 — change acceleration |
| 8 | **Video Summary** | 3-panel overview (latent L2, generation time, GPU memory) |
| 9 | **Inference Dashboard** | 3×3 multi-panel combining key charts |

### Example Outputs

> **Note:** The plots below are generated after running inference. Placeholders are shown until then.

#### Inference Dashboard
![Inference Dashboard](metrics/inference_dashboard.png)
*3×3 overview of all key inference metrics.*

#### Latent L2 per Step
![Latent L2 Per Step](metrics/latent_l2_per_step.png)
*How the latent representation changes at each diffusion step.*

#### Generation Times
![Generation Times](metrics/generation_times.png)
*Wall-clock time for each video — useful for spotting outliers or GPU throttling.*

---

## Usage

Metrics are automatically collected when `run_inference.py` runs. No manual setup needed.

```python
# Programmatic usage (if running outside the CLI)
from inference.metrics import InferenceMetricsTracker, generate_all_inference_plots

tracker = InferenceMetricsTracker(output_dir="./my_metrics", character="A")

# After each video:
tracker.record_video(result=generation_result, vid_id=0, character="A", config=config)

# After all videos:
generate_all_inference_plots(tracker, save_dir="./my_metrics")
```

---

## Output Files

After a run, the metrics directory contains:

```
metrics/
├── video_metrics.csv          # One row per video
├── step_metrics.csv           # One row per diffusion step
├── inference_metrics.json     # Full run snapshot
├── latent_l2_per_step.png     # Plot
├── pixel_bw_l2_per_step.png   # Plot
├── schedule_overlay.png       # Plot
├── generation_times.png       # Plot
├── final_image_stats.png      # Plot
├── cumulative_l2.png          # Plot
├── l2_velocity.png            # Plot
├── video_summary.png          # Plot
└── inference_dashboard.png    # Plot
```
