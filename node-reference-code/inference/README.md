# EMNIST Diffusion Inference Engine

A modular, extensible inference system for class-conditional EMNIST diffusion models with **classifier-free guidance**, **force composition**, and **state-space drift**.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Architecture](#architecture)
4. [Configuration](#configuration)
5. [Force Zoo](#force-zoo)
6. [Drift Zoo](#drift-zoo)
7. [Presets](#presets)
8. [Output Structure](#output-structure)
9. [CLI Reference](#cli-reference)
10. [Extending the System](#extending-the-system)
11. [Mathematical Background](#mathematical-background)

---

## Quick Start

### Minimal Python Usage

```python
from inference import InferenceConfig, generate_sample, load_checkpoint, load_unet, build_scheduler

# Load model
ckpt = load_checkpoint("/path/to/checkpoint.pt")
unet = load_unet(ckpt, device="cuda")
scheduler = build_scheduler(ckpt)

# Configure
config = InferenceConfig(character="A", total_videos=10)

# Generate one sample
result = generate_sample(
    model=unet,
    scheduler=scheduler,
    config=config,
    ckpt_info=ckpt,
)

# result.frames contains all intermediate latents
# result.final_image is the decoded output
```

### CLI Usage

```bash
# Basic generation
python -m inference.run_inference \
    --checkpoint /path/to/model.pt \
    --character A \
    --total-videos 100

# With preset force stack
python -m inference.run_inference \
    --checkpoint /path/to/model.pt \
    --character k \
    --preset subtract \
    --preset-args '{"pos_class": "k", "neg_class": "K", "w_pos": 3.0}'

# SLURM distributed generation
python -m inference.run_inference \
    --checkpoint /path/to/model.pt \
    --character A \
    --total-videos 10000 \
    --distributed
```

---

## Core Concepts

### Score Space vs State Space

The diffusion process operates in two conceptual spaces:

| Space | What it is | What we modify | Tool |
|-------|------------|----------------|------|
| **Score Space** | The UNet's predicted noise (∇ log p) | Classifier-free guidance, force composition | Forces |
| **State Space** | The actual latent x_t being denoised | Direct manipulation of the sample | Drifts |

**Forces** act on the score—they change *what direction* the model thinks is "toward the data."

**Drifts** act on the state—they directly nudge *where the sample is* in latent space.

### Classifier-Free Guidance (CFG)

CFG amplifies class-conditional signal by contrasting conditional vs unconditional predictions:

```
score_guided = score_uncond + guidance_scale * (score_cond - score_uncond)
```

This is automatically handled when `guidance_scale > 1.0`.

### Force Composition

Forces are **stacked** and **composed**. Each force in the stack adds a term to the final score:

```
final_score = base_score + Σ (force_i contribution)
```

Forces can:
- Attract toward a class
- Repel from a class
- Blend between classes
- Change dynamically over time

### The Generation Loop

```
for t in timesteps (T → 0):
    1. Get model prediction: ε = UNet(x_t, t, class_label)
    2. Apply forces: ε' = compose_forces(ε, force_stack, t, ...)
    3. Scheduler step: x_{t-1} = scheduler.step(ε', t, x_t)
    4. Apply drift: x_{t-1} = drift_fn(x_{t-1}, t, ...)
    5. Capture frame data (optional)
```

---

## Architecture

```
inference/
├── __init__.py            # Package exports
├── config.py              # Configuration dataclasses
├── checkpoint.py          # Checkpoint discovery and loading
├── model.py               # UNet construction
├── scheduler.py           # Noise scheduler setup
├── generation_loop.py     # Pure generation loop (no I/O)
├── video_writer.py        # FFmpeg video abstraction
├── metadata.py            # JSON metadata builders
├── logging_utils.py       # Rich 4-panel GPU dashboard
├── run_inference.py       # CLI entry point
│
├── metrics/               # ← Metrics & plotting suite
│   ├── __init__.py
│   ├── tracker.py         # InferenceMetricsTracker (CSV/JSON)
│   ├── plots.py           # 9 publication-quality charts
│   └── README.md          # Detailed metrics documentation
│
├── zoos/                  # ← Composable dynamics
│   ├── __init__.py
│   ├── force_zoo.py       # Score-space force registry
│   ├── drift_zoo.py       # State-space drift registry
│   ├── presets.py         # Ready-made configurations
│   └── README.md          # Zoo documentation & energy landscape guide
│
└── utils/                 # ← Low-level helpers
    ├── __init__.py
    ├── tensors.py         # Tensor conversion & serialization
    ├── stats.py           # Statistical summaries
    ├── attribution.py     # Spatial attribution analysis
    └── README.md          # Utility function reference
```

### Subfolder Documentation

| Subfolder | README | Description |
|-----------|--------|-------------|
| `metrics/` | [`metrics/README.md`](metrics/README.md) | Per-video and per-step metrics tracking, CSV/JSON persistence, 9 publication-quality plots |
| `zoos/` | [`zoos/README.md`](zoos/README.md) | Force & drift registries — composable dynamical steering of the diffusion process |
| `utils/` | [`utils/README.md`](utils/README.md) | Tensor helpers, statistical utilities, interpretability/attribution tools |

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | Hierarchical dataclass configuration with serialization |
| `checkpoint.py` | Find and load `.pt` checkpoints, extract metadata |
| `model.py` | Build `UNet2DModel` from checkpoint parameters |
| `scheduler.py` | Build DDPM/DDIM scheduler, extract constants |
| `generation_loop.py` | **Pure** diffusion loop—no disk I/O, no side effects |
| `video_writer.py` | FFmpeg subprocess wrapper for video encoding |
| `metadata.py` | Build structured JSON for reproducibility |
| `logging_utils.py` | Rich-based 4-panel GPU dashboard with live progress |
| `run_inference.py` | Orchestration, CLI parsing, metrics wiring |

---

## Configuration

Configuration uses **nested dataclasses** for type safety and IDE support.

### InferenceConfig (Top-Level)

```python
@dataclass
class InferenceConfig:
    # Core settings
    character: str = "A"              # Target class character
    total_videos: int = 100           # Number of samples to generate
    checkpoint_path: Optional[str] = None
    output_dir: str = "output"
    
    # Nested configs
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    force: ForceConfig = field(default_factory=ForceConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
```

### CaptureConfig

Controls what intermediate data is saved:

```python
@dataclass
class CaptureConfig:
    save_latents: bool = False        # Save raw latent tensors
    save_frames: bool = True          # Save frame images
    save_deltas: bool = True          # Save frame-to-frame differences
    save_video: bool = True           # Encode MP4 video
    
    latent_format: str = "npy"        # "npy" or "pt"
    frame_format: str = "png"         # "png" or "jpg"
```

### ForceConfig

```python
@dataclass
class ForceConfig:
    guidance_scale: float = 7.5       # CFG scale
    force_stack: List[Dict] = field(default_factory=list)
    global_cfg: Dict = field(default_factory=dict)
    clip_abs: Optional[float] = None  # Clip force magnitudes
```

### DriftConfig

```python
@dataclass
class DriftConfig:
    enabled: bool = False
    function_name: str = "drift_default"
    scale: float = 1.0
    decay: float = 0.99
    # ... additional drift parameters
```

### Creating Configs

```python
# From scratch
config = InferenceConfig(
    character="k",
    total_videos=50,
    capture=CaptureConfig(save_latents=True, save_video=False),
    force=ForceConfig(guidance_scale=10.0),
)

# From dict (e.g., loaded from JSON)
config = InferenceConfig.from_dict(json.load(open("config.json")))

# To dict (for saving)
config_dict = config.to_dict()
```

---

## Force Zoo

Forces modify the **score** (predicted noise) during generation. They're defined in `zoos/force_zoo.py`.

### Force Stack Format

A force stack is a list of dictionaries:

```python
force_stack = [
    {
        "name": "force_attract_class",
        "target_class": "A",
        "weight": 3.0,
    },
    {
        "name": "force_repel_class",
        "target_class": "B",
        "weight": -1.5,
    },
]
```

### Available Forces

| Force | Description | Key Parameters |
|-------|-------------|----------------|
| `force_none` | No-op, returns zero | — |
| `force_attract_class` | Pull toward a class | `target_class`, `weight` |
| `force_repel_class` | Push away from a class | `target_class`, `weight` |
| `force_blend_two` | Interpolate between two classes | `class_a`, `class_b`, `blend_alpha` |
| `force_subtract_two` | Positive class minus negative class | `pos_class`, `neg_class`, `w_pos`, `w_neg` |
| `force_weighted_multi` | Weighted sum of multiple classes | `class_weights: Dict[str, float]` |
| `force_switch_two` | Switch class at timestep | `class_a`, `class_b`, `switch_t` |
| `force_cycle_classes` | Rotate through classes | `classes: List[str]`, `period` |
| `force_oscillate_two` | Sinusoidal blend | `class_a`, `class_b`, `frequency` |
| `force_ramp_attract` | Linear ramp weight | `target_class`, `start_w`, `end_w` |
| `force_user_schedule` | User-defined time schedule | `schedule: List[Dict]` |
| `force_user_custom` | Fully custom function | `fn: Callable` |

### Force Function Signature

All forces have the same signature:

```python
def force_fn(
    model_output: torch.Tensor,      # Current predicted noise
    x_t: torch.Tensor,               # Current latent
    t: int,                          # Current timestep
    t_index: int,                    # Index in timestep sequence
    total_steps: int,                # Total number of steps
    model: UNet2DModel,              # The UNet
    ckpt_info: CheckpointInfo,       # Checkpoint metadata
    class_idx: int,                  # Target class index
    force_cfg: Dict[str, Any],       # Force-specific config
    state: Dict[str, Any],           # Persistent state dict
    **kwargs,
) -> torch.Tensor:                   # Force contribution (same shape as model_output)
```

### Adding a Custom Force

```python
from inference.zoos.force_zoo import FORCE_REGISTRY

def force_my_custom(model_output, x_t, t, t_index, total_steps, 
                    model, ckpt_info, class_idx, force_cfg, state, **kwargs):
    """My custom force that does something special."""
    my_param = force_cfg.get("my_param", 1.0)
    
    # Your logic here
    force = torch.zeros_like(model_output)
    
    return force * my_param

# Register it
FORCE_REGISTRY["force_my_custom"] = force_my_custom
```

---

## Drift Zoo

Drifts modify the **state** (latent sample) after each scheduler step. They're defined in `zoos/drift_zoo.py`.

### Available Drifts

| Drift | Description |
|-------|-------------|
| `drift_none` | No modification |
| `drift_default` | Small Gaussian noise scaled by decay |
| `drift_time_scaled` | Noise magnitude scales with timestep |
| `drift_inverse_time_scaled` | More noise at later (smaller t) steps |
| `drift_oscillatory` | Sinusoidal noise pattern |
| `drift_decay_only` | Pure exponential decay toward zero |
| `drift_clamp` | Clamp values to range |
| `drift_normalize` | Normalize to unit variance |
| `drift_user_custom` | Custom callable |

### Drift Function Signature

```python
def drift_fn(
    x: torch.Tensor,          # Current latent
    t: int,                   # Current timestep
    t_index: int,             # Index in sequence
    total_steps: int,         # Total steps
    drift_cfg: Dict[str, Any], # Drift config
    state: Dict[str, Any],    # Persistent state
    **kwargs,
) -> torch.Tensor:            # Modified latent
```

---

## Presets

Presets are **convenience factories** that return properly configured force stacks. They're in `zoos/presets.py`.

### Available Presets

```python
from inference.zoos.presets import (
    null_wander,      # No guidance, just noise
    attract_class,    # Simple attraction to one class
    repel_class,      # Repel from a class
    blend,            # Blend two classes
    subtract,         # Positive minus negative (contrastive)
    oscillate,        # Oscillate between two classes
    competitive,      # Multiple classes compete
    rotating_all,     # Cycle through all classes
    crystallize,      # Start random, converge to class
    chaos,            # High-frequency oscillation
    ramp_to_class,    # Gradual increase in guidance
    weighted_multi,   # Weighted combination
)
```

### Usage

```python
from inference.zoos.presets import subtract

config = InferenceConfig(
    character="k",
    force=ForceConfig(
        force_stack=subtract("k", "K", w_pos=3.0, w_neg=2.0),
        guidance_scale=7.5,
    ),
)
```

### Preset Examples

```python
# Make lowercase "a" but NOT uppercase "A"
force_stack = subtract("a", "A", w_pos=4.0, w_neg=2.0)

# Blend "a" and "e" at 50/50
force_stack = blend("a", "e", alpha=0.5)

# Oscillate between "0" and "O" with period 20 steps
force_stack = oscillate("0", "O", frequency=0.05)

# Ramp guidance from 1.0 to 10.0 over generation
force_stack = ramp_to_class("k", start_weight=1.0, end_weight=10.0)
```

---

## Output Structure

Generated outputs follow a consistent directory structure:

```
output/
└── {character}/
    ├── metadata.json           # Run metadata (config, environment, etc.)
    ├── index.json              # Index of all generated videos
    │
    ├── videos/
    │   ├── v000.mp4
    │   ├── v001.mp4
    │   └── ...
    │
    ├── frames/                 # Frames (if enabled)
    │   ├── v000_f0000.png
    │   ├── v000_f0001.png
    │   └── ...
    │
    ├── deltas/                 # Frame differences (if enabled)
    │   ├── v000_d0001.npy
    │   └── ...
    │
    └── latents/                # Raw latent tensors (if enabled)
        ├── v000_t0999.npy
        └── ...
```

### Metadata Format

`metadata.json` contains:

```json
{
    "run_id": "uuid-string",
    "timestamp_start": "2026-02-17T10:30:00Z",
    "timestamp_end": "2026-02-17T10:45:00Z",
    
    "config": { /* full InferenceConfig as dict */ },
    
    "checkpoint": {
        "path": "/path/to/model.pt",
        "classes": ["0", "1", ..., "z"],
        "size": 28,
        "channels": 1
    },
    
    "environment": {
        "hostname": "compute-node-01",
        "gpu": "NVIDIA A100",
        "cuda_version": "12.1",
        "torch_version": "2.1.0",
        "slurm_job_id": "12345678",
        "slurm_array_task_id": "0"
    },
    
    "scheduler": {
        "num_train_timesteps": 1000,
        "beta_start": 0.0001,
        "beta_end": 0.02
    }
}
```

---

## CLI Reference

```bash
python -m inference.run_inference [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--checkpoint PATH` | Path to model checkpoint |
| `--character CHAR` | Target character class |

### Generation Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--total-videos N` | 100 | Number of samples to generate |
| `--num-inference-steps N` | 50 | Denoising steps |
| `--guidance-scale F` | 7.5 | CFG scale |
| `--seed N` | None | Random seed (None = random) |

### Output Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir PATH` | output | Output directory |
| `--save-latents` | False | Save raw latent tensors |
| `--save-video` | True | Encode MP4 videos |
| `--no-video` | — | Disable video encoding |
| `--fps N` | 30 | Video framerate |

### Force/Drift Options

| Argument | Description |
|----------|-------------|
| `--preset NAME` | Use a preset force stack |
| `--preset-args JSON` | JSON args for preset |
| `--force-stack JSON` | Custom force stack as JSON |
| `--drift NAME` | Drift function name |
| `--drift-scale F` | Drift magnitude scale |

### Distributed Options

| Argument | Description |
|----------|-------------|
| `--distributed` | Enable SLURM array sharding |
| `--rank N` | Manual rank override |
| `--world-size N` | Manual world size override |

### Example Commands

```bash
# Basic generation
python -m inference.run_inference \
    --checkpoint models/emnist.pt \
    --character k \
    --total-videos 50

# High guidance, save everything
python -m inference.run_inference \
    --checkpoint models/emnist.pt \
    --character A \
    --guidance-scale 12.0 \
    --save-latents \
    --total-videos 100

# Contrastive: k but not K
python -m inference.run_inference \
    --checkpoint models/emnist.pt \
    --character k \
    --preset subtract \
    --preset-args '{"pos_class": "k", "neg_class": "K", "w_pos": 4.0, "w_neg": 2.0}'

# Distributed across SLURM array
#SBATCH --array=0-9
python -m inference.run_inference \
    --checkpoint models/emnist.pt \
    --character A \
    --total-videos 1000 \
    --distributed
```

---

## Extending the System

### Adding a New Force

1. Define the function in `zoos/force_zoo.py`:

```python
def force_my_new_force(model_output, x_t, t, t_index, total_steps,
                       model, ckpt_info, class_idx, force_cfg, state, **kwargs):
    # Your implementation
    return force_contribution
```

2. Register it:

```python
FORCE_REGISTRY["force_my_new_force"] = force_my_new_force
```

3. Use it:

```python
force_stack = [{"name": "force_my_new_force", "my_param": 2.0}]
```

### Adding a New Drift

Same pattern in `zoos/drift_zoo.py`:

```python
def drift_my_drift(x, t, t_index, total_steps, drift_cfg, state, **kwargs):
    return modified_x

DRIFT_REGISTRY["drift_my_drift"] = drift_my_drift
```

### Adding a New Preset

In `zoos/presets.py`:

```python
def my_preset(class_a: str, class_b: str, **kwargs) -> List[Dict]:
    return [
        {"name": "force_attract_class", "target_class": class_a, "weight": 2.0},
        {"name": "force_repel_class", "target_class": class_b, "weight": 1.0},
    ]
```

### Custom Frame Callback

For custom per-frame processing without modifying the generation loop:

```python
from inference.generation_loop import generate_sample, FrameData

def my_callback(frame: FrameData) -> None:
    print(f"Step {frame.t_index}: norm={frame.latent.norm():.4f}")
    # Custom processing, logging, visualization, etc.

result = generate_sample(
    model=unet,
    scheduler=scheduler,
    config=config,
    ckpt_info=ckpt,
    frame_callback=my_callback,
)
```

---

## Mathematical Background

### Diffusion Model Basics

A diffusion model learns to reverse a noising process. Given data x₀, we add noise:

$$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon$$

where $\epsilon \sim \mathcal{N}(0, I)$ and $\bar\alpha_t$ is a noise schedule.

The model learns to predict $\epsilon$ from $x_t$:

$$\epsilon_\theta(x_t, t, c) \approx \epsilon$$

where $c$ is the conditioning (class label).

### Score Function Interpretation

The predicted noise relates to the **score** (gradient of log probability):

$$\nabla_{x_t} \log p(x_t) \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar\alpha_t}}$$

This is why we call force modifications "score-space" operations.

### Classifier-Free Guidance

CFG combines conditional and unconditional predictions:

$$\tilde\epsilon = \epsilon_\theta(x_t, t, \varnothing) + s \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing))$$

where $s$ is the guidance scale. Higher $s$ means stronger conditioning.

### Force Composition

Forces add perturbations to the score:

$$\tilde\epsilon_{\text{final}} = \tilde\epsilon + \sum_i w_i \cdot F_i(x_t, t, \ldots)$$

For example, `force_attract_class` computes:

$$F_{\text{attract}}(c) = \epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)$$

This is the "direction toward class $c$" in score space.

### Contrastive Subtraction

The `subtract` preset implements:

$$\tilde\epsilon = \epsilon_\theta(x_t, t, \varnothing) + w_+ \cdot (\epsilon_\theta(x_t, t, c_+) - \epsilon_\theta(x_t, t, \varnothing)) - w_- \cdot (\epsilon_\theta(x_t, t, c_-) - \epsilon_\theta(x_t, t, \varnothing))$$

This generates samples that are **like** $c_+$ but **unlike** $c_-$.

### State-Space Drift

After the scheduler step produces $x_{t-1}$, drift modifies it directly:

$$x'_{t-1} = D(x_{t-1}, t, \ldots)$$

Common drifts add noise or apply decay. This operates in "pixel space" (well, latent space) rather than score space.

---

## Tips and Tricks

### Choosing Guidance Scale

- **1.0**: No guidance, unconditional generation
- **3.0-5.0**: Mild guidance, more diversity
- **7.5**: Standard starting point
- **10.0-15.0**: Strong guidance, less diversity, more "on-class"
- **>20.0**: Often oversaturated/artifacted

### Force Weight Intuition

- **0.0**: No effect
- **1.0**: Baseline strength (similar magnitude to CFG)
- **2.0-3.0**: Strong attraction/repulsion
- **>5.0**: Usually too strong, causes artifacts

### Debugging Generation

```python
# Enable verbose frame capture
config = InferenceConfig(
    capture=CaptureConfig(
        save_latents=True,
        latent_format="npy",
    ),
)

# After generation, inspect
import numpy as np
latent = np.load("output/A/latents/v000_t0500.npy")
print(f"Latent stats at t=500: mean={latent.mean():.4f}, std={latent.std():.4f}")
```

### Performance Tips

1. **Batch generation**: The loop generates one sample at a time. For throughput, run multiple processes.
2. **Disable video encoding** if you only need frames: `--no-video`
3. **Use SLURM arrays** for large-scale generation: `--distributed`
4. **Reduce inference steps** for quick tests: `--num-inference-steps 20`

---

## Sample Outputs

> **Note:** The images and videos below are generated after running inference. Placeholders are shown until you run the pipeline.

### Generated Videos

Videos show the full denoising trajectory from pure noise → final character, captured at every diffusion step.

| Character | Video |
|-----------|-------|
| A | ![Video of 'A'](../output/A/videos/video_v000.mp4) |
| k | ![Video of 'k'](../output/k/videos/video_v000.mp4) |

*Each frame is one diffusion step. Early frames are noisy; later frames converge to the target character.*

### Generated Frames (Snapshots)

| Step 0 (Pure Noise) | Step 150 | Step 300 | Step 600 (Final) |
|---------------------|----------|----------|------------------|
| ![step 0](../output/A/frames/frame_v000_f0000.png) | ![step 150](../output/A/frames/frame_v000_f0150.png) | ![step 300](../output/A/frames/frame_v000_f0300.png) | ![step 600](../output/A/frames/frame_v000_f0599.png) |

*Progression from random noise to a recognizable handwritten character.*

### Inference Metrics Dashboard

![Inference Dashboard](metrics/inference_dashboard.png)

*3×3 multi-panel dashboard showing latent L2, pixel delta, schedule overlay, generation times, and more.*

### Per-Step Latent Dynamics

| Latent L2 per Step | Cumulative L2 | L2 Velocity |
|--------------------|---------------|-------------|
| ![Latent L2](metrics/latent_l2_per_step.png) | ![Cumulative L2](metrics/cumulative_l2.png) | ![L2 Velocity](metrics/l2_velocity.png) |

*How the latent representation evolves during denoising. The velocity plot shows acceleration/deceleration of change.*

### Generation Timing

![Generation Times](metrics/generation_times.png)

*Wall-clock time per video. Useful for identifying GPU throttling or scheduling anomalies on SLURM.*

---

## License

MIT License. See repository root for details.

---

## Citation

If you use this code in research, please cite:

```bibtex
@software{emnist_diffusion_inference,
    title = {EMNIST Diffusion Inference Engine},
    author = {David Falk},
    organization = {APEX Laboratory, The University of Chicago},
    year = {2026},
    url = {https://github.com/your-repo}
}
```
