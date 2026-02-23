"""
EMNIST Diffusion Inference Engine
=================================

A modular, composable, research-grade inference system for class-conditional
DDPM on EMNIST (or similar pixel-space diffusion models).

This package provides:
- Clean configuration system via dataclasses
- Automatic checkpoint discovery and validation
- Model and scheduler construction from checkpoints
- A pure generation loop with composable force and drift dynamics
- Video writing via ffmpeg subprocess
- Rich metadata capture for research reproducibility
- Attribution utilities for interpretability research

Mathematical Foundations
------------------------

**Score Space vs. State Space**

The diffusion process operates in two conceptual spaces:

1. **State Space (x-space)**: The raw latent image x_t at timestep t.
   Drift functions modify x directly after each scheduler step.

2. **Score Space (pred-space)**: The model's predicted noise / score ε_θ(x_t, t).
   Force functions modify the predicted score before the scheduler step.

**Force Composition**

Forces are additive modifications to the unconditional score:

    pred = pred_uncond + Σ_i force_i(x, t, pred_uncond, ...)

Each force_i can implement class guidance, repulsion, blending, oscillation,
or any custom dynamics in the score manifold.

**Drift Composition**

Drift functions are applied after the scheduler step:

    x_{t-1} = scheduler.step(pred, t, x_t).prev_sample
    x_{t-1} = drift(x_{t-1}, t, step_index, total_steps, cfg)

This allows for state-space perturbations independent of the score dynamics.

**Energy Landscape Interpretation**

Each class defines a basin in the score manifold. Force functions can be
interpreted as gradient fields:

- **Attract**: Pull toward a class basin (positive gravity)
- **Repel**: Push away from a class basin (negative gravity)
- **Blend**: Weighted combination of multiple basins
- **Oscillate**: Time-varying oscillation between basins

Architecture
------------

::

    inference/
    ├── __init__.py            # This file - package exports
    ├── config.py              # Configuration dataclasses
    ├── checkpoint.py          # Checkpoint discovery and loading
    ├── model.py               # UNet construction and weight loading
    ├── scheduler.py           # DDPMScheduler setup and diagnostics
    ├── generation_loop.py     # Pure diffusion generation loop
    ├── video_writer.py        # FFmpeg video writing abstraction
    ├── metadata.py            # Metadata capture and JSON building
    ├── logging_utils.py       # Logging utilities
    ├── zoos/
    │   ├── __init__.py        # Zoo exports
    │   ├── force_zoo.py       # Score-space force registry
    │   ├── drift_zoo.py       # State-space drift registry
    │   └── presets.py         # Ready-made force stacks
    ├── utils/
    │   ├── __init__.py        # Utility exports
    │   ├── tensors.py         # Tensor manipulation helpers
    │   ├── stats.py           # Statistical utilities
    │   └── attribution.py     # Attribution and interpretability
    └── run_inference.py       # Main entry point

Usage
-----

Basic usage::

    from inference.config import InferenceConfig
    from inference.run_inference import run_inference
    
    config = InferenceConfig(
        character="k",
        total_videos=10,
    )
    run_inference(config)

With custom force stack::

    from inference.zoos.presets import subtract
    
    config = InferenceConfig(
        character="k",
        force_stack=subtract("k", "m"),
    )
    run_inference(config)

Author
------
David Falk
APEX Laboratory, The University of Chicago
2026
"""

__version__ = "2.0.0"

# Core configuration
from inference.config import (
    InferenceConfig,
    CaptureConfig,
    VideoConfig,
    DriftConfig,
    ForceConfig,
    EnvironmentConfig,
)

# Checkpoint handling
from inference.checkpoint import (
    CheckpointInfo,
    discover_checkpoint,
    load_checkpoint,
    validate_character,
)

# Model construction
from inference.model import build_unet, load_unet

# Scheduler
from inference.scheduler import build_scheduler, get_scheduler_constants

# Generation
from inference.generation_loop import GenerationResult, generate_sample

# Video writing
from inference.video_writer import VideoWriter

# Metadata
from inference.metadata import build_run_metadata, update_end_timestamp

# Logging
from inference.logging_utils import PrettyLogger, get_logger

# Package-level exports
__all__ = [
    # Version
    "__version__",
    # Config
    "InferenceConfig",
    "CaptureConfig",
    "VideoConfig",
    "DriftConfig",
    "ForceConfig",
    "EnvironmentConfig",
    # Checkpoint
    "CheckpointInfo",
    "discover_checkpoint",
    "load_checkpoint",
    "validate_character",
    # Model
    "build_unet",
    "load_unet",
    # Scheduler
    "build_scheduler",
    "get_scheduler_constants",
    # Generation
    "GenerationResult",
    "generate_sample",
    # Video
    "VideoWriter",
    # Metadata
    "build_run_metadata",
    "update_end_timestamp",
    # Logging
    "PrettyLogger",
    "get_logger",
]
