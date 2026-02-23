"""
Generation Loop
===============

Pure diffusion generation loop with composable dynamics.

This module contains the core generation logic, separated from:
- Video writing
- File system operations
- Metadata capture
- Logging

The generation loop accepts:
- Model and scheduler
- Configuration
- Force stack and drift function

And returns a structured GenerationResult with captured data.

Design Philosophy
-----------------

The generation loop should be **pure** in the sense that it:
- Does not perform I/O directly
- Returns all results as data structures
- Is testable in isolation
- Is composable with different output handlers

The caller is responsible for:
- Writing videos
- Saving metadata
- Logging progress
- File organization

Core Algorithm
--------------

```python
for step_index, t in enumerate(scheduler.timesteps):
    # Unconditional prediction
    pred_uncond = model(x, t, class_labels=null_label)
    
    # Optional conditional prediction for CFG
    if guidance_scale > 0:
        pred_cond = model(x, t, class_labels=target_label)
        pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    else:
        pred = pred_uncond
    
    # Compose forces
    pred = compose_forces(x, t, pred_uncond, force_stack, ...)
    
    # Scheduler step
    x = scheduler.step(pred, t, x).prev_sample
    
    # Apply drift
    x = drift(x, t, step_index, ...)
    
    # Capture frame data
    capture_frame(...)
```

"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from inference.config import InferenceConfig, CaptureConfig, StartingImageConfig
from inference.scheduler import get_timestep_scalars
from inference.utils.tensors import scheduler_step_to_dict
from inference.utils.stats import tensor_stats
from inference.utils.attribution import top_coords_2d, compute_active_mask
from inference.zoos.force_zoo import compose_forces
from inference.zoos.drift_zoo import DriftFn, drift_none


# =============================================================================
# Starting-Image Bank
# =============================================================================

def initialize_x(
    starting_cfg: StartingImageConfig,
    shape: tuple,
    device: str,
    generator: torch.Generator,
    vid_index: int = 0,
) -> torch.Tensor:
    """
    Create the initial latent ``x_T``.

    If ``starting_cfg.enabled`` is *False*, falls back to standard
    Gaussian noise.  Otherwise loads a pre-selected tensor from the
    starting-image bank.

    Args:
        starting_cfg: Starting-image configuration.
        shape: Desired tensor shape, e.g. ``(1, 1, h, w)``.
        device: Target device.
        generator: Torch RNG (used for noise fallback and random selection).
        vid_index: Current video index (used for sequential selection).

    Returns:
        Tensor of shape *shape* on *device*.
    """
    if not starting_cfg.enabled or not starting_cfg.bank_dir:
        return torch.randn(shape, device=device, generator=generator)

    bank_dir = starting_cfg.bank_dir
    if not os.path.isdir(bank_dir):
        raise FileNotFoundError(
            f"Starting-image bank directory not found: {bank_dir}"
        )

    # Collect .pt files sorted for reproducibility
    files = sorted(f for f in os.listdir(bank_dir) if f.endswith(".pt"))
    if not files:
        raise FileNotFoundError(
            f"No .pt files found in starting-image bank: {bank_dir}"
        )

    # Select file
    sel = starting_cfg.selection
    if sel == "fixed":
        idx = starting_cfg.fixed_index % len(files)
    elif sel == "sequential":
        idx = vid_index % len(files)
    else:  # "random"
        idx = int(torch.randint(len(files), (1,), generator=generator).item())

    path = os.path.join(bank_dir, files[idx])
    x = torch.load(path, map_location=device, weights_only=True)

    # Sanity-check shape
    if x.shape != shape:
        raise ValueError(
            f"Starting image shape {x.shape} != expected {shape} "
            f"(file: {path})"
        )
    return x


# =============================================================================
# Result Data Structures
# =============================================================================

@dataclass
class FrameData:
    """
    Captured data for a single frame.
    
    Contains all per-step captured data in a structured format.
    """
    k: int                              # Frame index (0, 1, 2, ...)
    step_index: int                     # Step index in scheduler
    timestep: int                       # Timestep value
    wall_dt: float                      # Wall clock time since start
    
    # Pixel data
    frame_u8: Optional[np.ndarray] = None  # Grayscale frame (H, W) uint8
    latent_u8: Optional[np.ndarray] = None  # Latent-resolution frame
    
    # Delta data
    delta_pixel: Optional[np.ndarray] = None   # Pixel delta (H, W) int16
    
    # Latent data
    delta_x: Optional[torch.Tensor] = None  # Latent delta (1, 1, h, w)
    
    # Statistics
    stats_delta_x: Optional[Dict[str, float]] = None
    stats_pixel: Optional[Dict[str, float]] = None
    stats_delta_pixel: Optional[Dict[str, float]] = None
    
    # Attribution
    active_latent_mask: Optional[torch.Tensor] = None
    active_pixel_mask: Optional[torch.Tensor] = None
    top_latent_coords: Optional[List[Tuple[int, int, float]]] = None
    top_pixel_coords: Optional[List[Tuple[int, int, float]]] = None
    
    # Scheduler diagnostics
    scheduler_step_out: Optional[Dict[str, Any]] = None
    alpha_t: Optional[float] = None
    beta_t: Optional[float] = None
    alpha_cumprod_t: Optional[float] = None


@dataclass
class GenerationResult:
    """
    Complete result of a generation run.
    
    Contains all captured data and metadata for a single video.
    """
    # Initial state
    x_init: torch.Tensor                # Initial noise (1, 1, h, w)
    seed: int                           # Random seed used
    label: int                          # Target class label
    
    # Final state
    x_final: torch.Tensor               # Final generated image
    
    # Per-frame data
    frames: List[FrameData] = field(default_factory=list)
    
    # Timing
    total_time: float = 0.0             # Total generation time
    
    # Latent tensors (stacked)
    delta_x_stack: Optional[torch.Tensor] = None    # (n_frames, 1, h, w)
    
    # L2 metrics
    delta_x_l2: Optional[List[float]] = None
    delta_pixel_l2: Optional[List[float]] = None
    
    # L2 maps
    delta_x_l2_map: Optional[torch.Tensor] = None   # (n_frames, h, w)
    delta_pixel_l2_map: Optional[torch.Tensor] = None  # (n_frames, H, W)


# =============================================================================
# Generation Loop
# =============================================================================

def generate_sample(
    model: Any,
    scheduler: Any,
    config: InferenceConfig,
    device: str,
    seed: int,
    label: int,
    null_class_index: int,
    classes: Sequence[str],
    force_stack: Optional[Sequence[Union[Dict[str, Any], Any]]] = None,
    drift_fn: Optional[DriftFn] = None,
    drift_cfg: Optional[Dict[str, Any]] = None,
    frame_callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> GenerationResult:
    """
    Run the diffusion generation loop.
    
    This is the core generation function. It performs the full diffusion
    process and returns a GenerationResult with all captured data.
    
    Args:
        model: UNet2DModel for noise prediction.
        scheduler: DDPM/DDIM scheduler.
        config: Inference configuration.
        device: Compute device ("cuda" or "cpu").
        seed: Random seed for reproducibility.
        label: Target class label index.
        null_class_index: Null class index for unconditional prediction.
        classes: List of class names.
        force_stack: Force specifications for score-space dynamics.
        drift_fn: Optional drift function for state-space dynamics.
        drift_cfg: Configuration for drift function.
        frame_callback: Optional callback(step, total, frame_u8) for each frame.
        progress_callback: Optional callback(step, total) for progress updates.
        
    Returns:
        GenerationResult with all captured data.
    """
    import os

    capture = config.capture
    
    # Get model dimensions from scheduler
    img = int(model.config.sample_size)
    upscale = config.video.upscale
    H = W = img * upscale
    
    # Setup generator
    gen = torch.Generator(device).manual_seed(seed)
    
    # Initialize latent (from bank or random noise)
    shape = (1, 1, img, img)
    x = initialize_x(
        starting_cfg=config.starting_image,
        shape=shape,
        device=device,
        generator=gen,
        vid_index=seed - config.base_seed,
    )
    x_init = x.detach().clone().cpu()
    
    # Labels
    y_target = torch.tensor([label], device=device, dtype=torch.int64)
    y_null = torch.tensor([null_class_index], device=device, dtype=torch.int64)
    
    # Timing
    t0 = time.time()
    
    # Step counting
    n_steps = len(scheduler.timesteps)
    stride = max(1, capture.save_every)
    
    # Determine which steps to save
    save_indices = set(range(0, n_steps, stride))
    save_indices.add(n_steps - 1)  # Always save final step
    
    # Preallocate result lists
    frames: List[FrameData] = []
    delta_x_list: List[torch.Tensor] = []
    delta_x_l2_list: List[float] = []
    delta_pixel_l2_list: List[float] = []
    
    # Previous frames for delta computation
    prev_frame: Optional[np.ndarray] = None
    
    # Force/drift setup
    if force_stack is None:
        force_stack = config.force.force_stack
    if drift_fn is None:
        if config.drift.enabled:
            from inference.zoos.drift_zoo import resolve_drift_fn
            drift_fn = resolve_drift_fn(config.drift.function_name)
        else:
            drift_fn = drift_none
    if drift_cfg is None:
        drift_cfg = config.drift.to_dict()
    
    # Force state (for caching)
    force_state: Dict[str, Any] = {}
    
    # Guidance scale
    guidance_scale = config.force.guidance_scale
    
    # Global force config
    force_cfg_global = dict(config.force.global_cfg)
    if config.force.clip_abs is not None:
        force_cfg_global["clip_abs"] = config.force.clip_abs
    if config.force.clip_norm is not None:
        force_cfg_global["clip_norm"] = config.force.clip_norm
    
    # Main loop
    frame_idx = 0
    for step_index, t in enumerate(scheduler.timesteps):
        should_save = step_index in save_indices
        
        with torch.no_grad():
            x_before = x.detach().clone()
            
            # Unconditional prediction
            pred_uncond = model(x, t, class_labels=y_null).sample
            
            # Conditional prediction (for CFG)
            if guidance_scale > 0:
                pred_cond = model(x, t, class_labels=y_target).sample
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                pred = pred_uncond
            
            # Clear force state cache at each step
            force_state.clear()
            
            # Compose forces
            if force_stack:
                pred = compose_forces(
                    x=x,
                    t=t,
                    step_index=step_index,
                    total_steps=n_steps,
                    unet=model,
                    pred_uncond=pred_uncond,
                    classes=classes,
                    stack=force_stack,
                    cfg_global=force_cfg_global,
                    state=force_state,
                )
            
            # Scheduler step
            step_out = scheduler.step(pred, t, x)
            x = step_out.prev_sample
            
            # Apply drift
            if drift_fn is not None:
                x = drift_fn(x, t, step_index, n_steps, drift_cfg)
            
            # Compute delta for latent capture
            if should_save and capture.save_latents:
                delta_x = (x.detach() - x_before).cpu().to(capture.delta_dtype)
                delta_x_list.append(delta_x)
                delta_x_l2 = float(torch.linalg.vector_norm(delta_x.float()).item())
                delta_x_l2_list.append(delta_x_l2)
        
        # Skip non-save steps
        if not should_save:
            if progress_callback is not None:
                progress_callback(step_index + 1, n_steps)
            continue
        
        # Wall time
        wall_dt = time.time() - t0
        
        # Convert to image
        pixel_float = ((x[0, 0] + 1) / 2).clamp(0, 1).detach().cpu().numpy()
        latent_frame = (pixel_float * 255).round().astype(np.uint8)
        
        pil = Image.fromarray(latent_frame, mode="L").resize((W, H), Image.NEAREST)
        frame_u8 = np.array(pil, dtype=np.uint8)
        
        # Compute pixel deltas
        if prev_frame is not None:
            delta_pixel = frame_u8.astype(np.int16) - prev_frame.astype(np.int16)
            delta_pixel_l2 = float(np.linalg.norm(delta_pixel.astype(np.float32)))
        else:
            delta_pixel = np.zeros_like(frame_u8, dtype=np.int16)
            delta_pixel_l2 = 0.0
        
        delta_pixel_l2_list.append(delta_pixel_l2)
        
        # Call frame callback
        if frame_callback is not None:
            frame_callback(step_index, n_steps, frame_u8)
        
        # Get timestep value
        try:
            t_val = int(t.item()) if hasattr(t, "item") else int(t)
        except Exception:
            t_val = step_index
        
        # Create frame data
        frame = FrameData(
            k=frame_idx,
            step_index=step_index,
            timestep=t_val,
            wall_dt=wall_dt,
        )
        
        # Populate based on capture settings
        if capture.save_pixels:
            frame.frame_u8 = frame_u8
            frame.latent_u8 = latent_frame
            frame.delta_pixel = delta_pixel
            frame.stats_pixel = tensor_stats(torch.from_numpy(frame_u8).float())
            frame.stats_delta_pixel = tensor_stats(torch.from_numpy(delta_pixel).float())
            
            # Pixel attribution
            frame.active_pixel_mask = compute_active_mask(
                torch.from_numpy(delta_pixel.astype(np.float32)),
                float(capture.active_pixel_threshold),
            )
            frame.top_pixel_coords = top_coords_2d(
                torch.from_numpy(np.abs(delta_pixel).astype(np.float32)),
                capture.top_n_coords,
            )
        
        if capture.save_latents and delta_x_list:
            frame.delta_x = delta_x_list[-1]
            frame.stats_delta_x = tensor_stats(delta_x_list[-1].float())
            
            # Latent attribution
            dx_abs = delta_x_list[-1][0, 0].abs()
            frame.active_latent_mask = compute_active_mask(
                dx_abs, capture.active_latent_threshold
            )
            frame.top_latent_coords = top_coords_2d(dx_abs, capture.top_n_coords)
        
        # Scheduler diagnostics
        frame.scheduler_step_out = scheduler_step_to_dict(step_out)
        scalars = get_timestep_scalars(scheduler, t_val)
        frame.alpha_t = scalars.get("alpha_t")
        frame.beta_t = scalars.get("beta_t")
        frame.alpha_cumprod_t = scalars.get("alpha_cumprod_t")
        
        frames.append(frame)
        frame_idx += 1
        
        # Update previous frames
        prev_frame = frame_u8
        
        # Progress callback
        if progress_callback is not None:
            progress_callback(step_index + 1, n_steps)
    
    # Total time
    total_time = time.time() - t0
    
    # Stack delta_x tensors
    delta_x_stack = None
    if delta_x_list:
        delta_x_stack = torch.cat(delta_x_list, dim=0)
    
    # Build result
    result = GenerationResult(
        x_init=x_init,
        seed=seed,
        label=label,
        x_final=x.detach().cpu(),
        frames=frames,
        total_time=total_time,
        delta_x_stack=delta_x_stack,
        delta_x_l2=delta_x_l2_list if delta_x_l2_list else None,
        delta_pixel_l2=delta_pixel_l2_list if delta_pixel_l2_list else None,
    )
    
    return result


# =============================================================================
# Batched Generation Loop
# =============================================================================

# Type alias for the batched frame callback.
# Receives (step_index, total_steps, list_of_frame_u8_arrays).
BatchedFrameCallback = Callable[[int, int, List[np.ndarray]], None]


def generate_sample_batched(
    model: Any,
    scheduler: Any,
    config: InferenceConfig,
    device: str,
    seeds: List[int],
    label: int,
    null_class_index: int,
    classes: Sequence[str],
    force_stack: Optional[Sequence[Union[Dict[str, Any], Any]]] = None,
    drift_fn: Optional[DriftFn] = None,
    drift_cfg: Optional[Dict[str, Any]] = None,
    frame_callback: Optional[BatchedFrameCallback] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[GenerationResult]:
    """
    Run the diffusion generation loop for **N** trajectories simultaneously.

    This function is functionally identical to :func:`generate_sample` but
    batches ``N = len(seeds)`` independent trajectories into a single UNet
    forward pass per diffusion step.  This dramatically improves GPU
    utilisation for small models (e.g. EMNIST 1×28×28 or 1×64×64).

    Each trajectory gets its own deterministic random seed, so results are
    **reproducible** and **identical** to calling :func:`generate_sample`
    with the same seed one-at-a-time (the only mathematical difference is
    floating-point ordering, which is negligible).

    Args:
        model: UNet2DModel for noise prediction.
        scheduler: DDPM/DDIM scheduler.
        config: Inference configuration.
        device: Compute device ("cuda" or "cpu").
        seeds: List of N random seeds — one per trajectory.
        label: Target class label index (shared by the whole batch).
        null_class_index: Null class index for unconditional prediction.
        classes: List of class names.
        force_stack: Force specifications for score-space dynamics.
        drift_fn: Optional drift function for state-space dynamics.
        drift_cfg: Configuration for drift function.
        frame_callback: Optional callback(step, total, List[frame_u8]) for each step.
        progress_callback: Optional callback(step, total) for progress updates.

    Returns:
        List of *N* :class:`GenerationResult`, one per seed, in the same
        order as *seeds*.
    """
    import os

    N = len(seeds)
    if N == 0:
        return []

    capture = config.capture

    # Model dimensions
    img = int(model.config.sample_size)
    upscale = config.video.upscale
    H = W = img * upscale

    # -----------------------------------------------------------------
    # Initialise N latents — each with its own seed for reproducibility
    # -----------------------------------------------------------------

    x_parts: List[torch.Tensor] = []
    for s in seeds:
        gen = torch.Generator(device).manual_seed(s)
        xi = initialize_x(
            starting_cfg=config.starting_image,
            shape=(1, 1, img, img),
            device=device,
            generator=gen,
            vid_index=s - config.base_seed,
        )
        x_parts.append(xi)

    x = torch.cat(x_parts, dim=0)  # (N, 1, img, img)
    x_init = x.detach().clone().cpu()  # save before mutation

    # Labels — same class for all N trajectories
    y_target = torch.tensor([label] * N, device=device, dtype=torch.int64)
    y_null = torch.tensor([null_class_index] * N, device=device, dtype=torch.int64)

    # Timing
    t0 = time.time()

    # Step counting
    n_steps = len(scheduler.timesteps)
    stride = max(1, capture.save_every)
    save_indices = set(range(0, n_steps, stride))
    save_indices.add(n_steps - 1)

    # Per-trajectory accumulators
    frames_per: List[List[FrameData]] = [[] for _ in range(N)]
    delta_x_lists: List[List[torch.Tensor]] = [[] for _ in range(N)]
    delta_x_l2_lists: List[List[float]] = [[] for _ in range(N)]
    delta_pixel_l2_lists: List[List[float]] = [[] for _ in range(N)]
    frame_idx_per: List[int] = [0] * N
    prev_frames: List[Optional[np.ndarray]] = [None] * N

    # Force / drift setup (shared across batch)
    if force_stack is None:
        force_stack = config.force.force_stack
    if drift_fn is None:
        if config.drift.enabled:
            from inference.zoos.drift_zoo import resolve_drift_fn
            drift_fn = resolve_drift_fn(config.drift.function_name)
        else:
            drift_fn = drift_none
    if drift_cfg is None:
        drift_cfg = config.drift.to_dict()

    force_state: Dict[str, Any] = {}
    guidance_scale = config.force.guidance_scale

    force_cfg_global = dict(config.force.global_cfg)
    if config.force.clip_abs is not None:
        force_cfg_global["clip_abs"] = config.force.clip_abs
    if config.force.clip_norm is not None:
        force_cfg_global["clip_norm"] = config.force.clip_norm

    # =================================================================
    # Main diffusion loop — single batched pass
    # =================================================================

    for step_index, t in enumerate(scheduler.timesteps):
        should_save = step_index in save_indices

        with torch.no_grad():
            x_before = x.detach().clone()

            # Unconditional prediction — (N, 1, img, img)
            pred_uncond = model(x, t, class_labels=y_null).sample

            # Conditional prediction for CFG
            if guidance_scale > 0:
                pred_cond = model(x, t, class_labels=y_target).sample
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                pred = pred_uncond

            # Forces — operate element-wise across the batch
            force_state.clear()
            if force_stack:
                pred = compose_forces(
                    x=x, t=t,
                    step_index=step_index, total_steps=n_steps,
                    unet=model, pred_uncond=pred_uncond,
                    classes=classes, stack=force_stack,
                    cfg_global=force_cfg_global, state=force_state,
                )

            # Scheduler step — handles batch dim natively
            step_out = scheduler.step(pred, t, x)
            x = step_out.prev_sample

            # Drift — element-wise, batch-safe
            if drift_fn is not None:
                x = drift_fn(x, t, step_index, n_steps, drift_cfg)

            # Per-trajectory latent deltas
            if should_save and capture.save_latents:
                delta_x_batch = (x.detach() - x_before).cpu().to(capture.delta_dtype)
                for i in range(N):
                    dxi = delta_x_batch[i : i + 1]  # keep (1,1,h,w) shape
                    delta_x_lists[i].append(dxi)
                    delta_x_l2_lists[i].append(
                        float(torch.linalg.vector_norm(dxi.float()).item())
                    )

        # Skip non-save steps
        if not should_save:
            if progress_callback is not None:
                progress_callback(step_index + 1, n_steps)
            continue

        wall_dt = time.time() - t0

        # -----------------------------------------------------------------
        # Convert each trajectory to pixel-space and build FrameData
        # -----------------------------------------------------------------
        frame_u8_list: List[np.ndarray] = []

        for i in range(N):
            pixel_float = ((x[i, 0] + 1) / 2).clamp(0, 1).detach().cpu().numpy()
            latent_frame = (pixel_float * 255).round().astype(np.uint8)

            pil = Image.fromarray(latent_frame, mode="L").resize((W, H), Image.NEAREST)
            frame_u8 = np.array(pil, dtype=np.uint8)

            # Pixel deltas
            if prev_frames[i] is not None:
                delta_pixel = frame_u8.astype(np.int16) - prev_frames[i].astype(np.int16)
                delta_pixel_l2 = float(np.linalg.norm(delta_pixel.astype(np.float32)))
            else:
                delta_pixel = np.zeros_like(frame_u8, dtype=np.int16)
                delta_pixel_l2 = 0.0

            delta_pixel_l2_lists[i].append(delta_pixel_l2)

            # Timestep
            try:
                t_val = int(t.item()) if hasattr(t, "item") else int(t)
            except Exception:
                t_val = step_index

            # Build frame
            frame = FrameData(
                k=frame_idx_per[i],
                step_index=step_index,
                timestep=t_val,
                wall_dt=wall_dt,
            )

            if capture.save_pixels:
                frame.frame_u8 = frame_u8
                frame.latent_u8 = latent_frame
                frame.delta_pixel = delta_pixel
                frame.stats_pixel = tensor_stats(torch.from_numpy(frame_u8).float())
                frame.stats_delta_pixel = tensor_stats(torch.from_numpy(delta_pixel).float())
                frame.active_pixel_mask = compute_active_mask(
                    torch.from_numpy(delta_pixel.astype(np.float32)),
                    float(capture.active_pixel_threshold),
                )
                frame.top_pixel_coords = top_coords_2d(
                    torch.from_numpy(np.abs(delta_pixel).astype(np.float32)),
                    capture.top_n_coords,
                )

            if capture.save_latents and delta_x_lists[i]:
                frame.delta_x = delta_x_lists[i][-1]
                frame.stats_delta_x = tensor_stats(delta_x_lists[i][-1].float())
                dx_abs = delta_x_lists[i][-1][0, 0].abs()
                frame.active_latent_mask = compute_active_mask(
                    dx_abs, capture.active_latent_threshold
                )
                frame.top_latent_coords = top_coords_2d(dx_abs, capture.top_n_coords)

            # Scheduler diagnostics (same for all trajectories in batch)
            frame.scheduler_step_out = scheduler_step_to_dict(step_out)
            scalars = get_timestep_scalars(scheduler, t_val)
            frame.alpha_t = scalars.get("alpha_t")
            frame.beta_t = scalars.get("beta_t")
            frame.alpha_cumprod_t = scalars.get("alpha_cumprod_t")

            frames_per[i].append(frame)
            frame_idx_per[i] += 1
            prev_frames[i] = frame_u8
            frame_u8_list.append(frame_u8)

        # Batched frame callback
        if frame_callback is not None:
            frame_callback(step_index, n_steps, frame_u8_list)

        if progress_callback is not None:
            progress_callback(step_index + 1, n_steps)

    # =================================================================
    # Build per-trajectory results
    # =================================================================

    total_time = time.time() - t0

    results: List[GenerationResult] = []
    for i in range(N):
        delta_x_stack = None
        if delta_x_lists[i]:
            delta_x_stack = torch.cat(delta_x_lists[i], dim=0)

        results.append(GenerationResult(
            x_init=x_init[i : i + 1],
            seed=seeds[i],
            label=label,
            x_final=x[i : i + 1].detach().cpu(),
            frames=frames_per[i],
            total_time=total_time / N,  # approximate per-video time
            delta_x_stack=delta_x_stack,
            delta_x_l2=delta_x_l2_lists[i] if delta_x_l2_lists[i] else None,
            delta_pixel_l2=delta_pixel_l2_lists[i] if delta_pixel_l2_lists[i] else None,
        ))

    return results


def build_latent_payload(
    result: GenerationResult,
    config: InferenceConfig,
    label: int,
) -> Dict[str, Any]:
    """
    Build the latent payload dictionary for saving.
    
    Args:
        result: Generation result.
        config: Inference configuration.
        label: Target label.
        
    Returns:
        Dictionary suitable for torch.save().
    """
    frames = result.frames
    n = len(frames)
    
    return {
        "index": {
            "k": [f.k for f in frames],
            "si": [f.step_index for f in frames],
            "t": [f.timestep for f in frames],
            "t_index": [f.step_index for f in frames],
            "wall_dt": [f.wall_dt for f in frames],
            "y": [label] * n,
        },
        "tensors": {
            "x_init": result.x_init,
            "delta_x": result.delta_x_stack,
            "y": torch.tensor([label], dtype=torch.int64),
        },
        "derived": {
            "delta_x_l2": result.delta_x_l2,
            "delta_x_l2_map": result.delta_x_l2_map,
            "active_latent_mask": torch.stack([
                f.active_latent_mask for f in frames if f.active_latent_mask is not None
            ], dim=0) if frames and frames[0].active_latent_mask is not None else None,
            "top_latent_coords": [f.top_latent_coords for f in frames],
            "alpha_t": [f.alpha_t for f in frames],
            "beta_t": [f.beta_t for f in frames],
            "alpha_cumprod_t": [f.alpha_cumprod_t for f in frames],
            "scheduler_step_out": [f.scheduler_step_out for f in frames],
        },
        "stats": {
            "delta_x": [f.stats_delta_x for f in frames],
        },
    }


def build_pixel_payload(
    result: GenerationResult,
    config: InferenceConfig,
) -> Dict[str, Any]:
    """
    Build the pixel payload dictionary for saving.
    
    Args:
        result: Generation result.
        config: Inference configuration.
        
    Returns:
        Dictionary suitable for torch.save().
    """
    frames = result.frames
    
    return {
        "index": {
            "k": [f.k for f in frames],
            "si": [f.step_index for f in frames],
            "t": [f.timestep for f in frames],
            "t_index": [f.step_index for f in frames],
            "wall_dt": [f.wall_dt for f in frames],
        },
        "tensors": {
            "latent_u8": torch.stack([
                torch.from_numpy(f.latent_u8) for f in frames 
                if f.latent_u8 is not None
            ], dim=0) if frames and frames[0].latent_u8 is not None else None,
            "frame_u8": torch.stack([
                torch.from_numpy(f.frame_u8) for f in frames if f.frame_u8 is not None
            ], dim=0) if frames and frames[0].frame_u8 is not None else None,
            "delta_pixel": torch.stack([
                torch.from_numpy(f.delta_pixel) for f in frames if f.delta_pixel is not None
            ], dim=0) if frames and frames[0].delta_pixel is not None else None,
        },
        "derived": {
            "delta_pixel_l2": result.delta_pixel_l2,
            "delta_pixel_l2_map": result.delta_pixel_l2_map,
            "active_pixel_mask": torch.stack([
                f.active_pixel_mask for f in frames if f.active_pixel_mask is not None
            ], dim=0) if frames and frames[0].active_pixel_mask is not None else None,
            "top_pixel_coords": [f.top_pixel_coords for f in frames],
        },
        "stats": {
            "frame_u8": [f.stats_pixel for f in frames],
            "delta_pixel": [f.stats_delta_pixel for f in frames],
        },
    }
