"""
Configuration System for EMNIST Diffusion Inference
====================================================

This module defines a hierarchical configuration system using Python dataclasses.
All inference parameters are encapsulated in explicitly typed, documented objects.

Design Principles:
- No hidden globals
- Explicit data flow
- Serializable for reproducibility
- Nested dataclasses for logical grouping

Configuration Hierarchy:
- InferenceConfig (top-level)
  ├── CaptureConfig (what data to save)
  ├── VideoConfig (video encoding parameters)
  ├── DriftConfig (state-space drift settings)
  ├── ForceConfig (score-space force settings)
  └── EnvironmentConfig (runtime environment)

"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch


# =============================================================================
# Capture Configuration
# =============================================================================

@dataclass
class CaptureConfig:
    """
    Configuration for data capture during generation.
    
    Controls what intermediate data is saved during the diffusion process.
    Different archive modes can enable/disable expensive captures.
    
    Attributes:
        archive_mode: Capture tier ("tier1" = minimal, "tier2" = full).
        delta_dtype: Data type for delta tensors (torch.float16 saves memory).
        save_deltas: Save per-frame delta arrays as .npy files.
        save_frames: Save per-frame PNG images (expensive).
        save_every: Frame stride (1 = every step, 2 = every other, etc.).
        save_meta: Save JSON run metadata.
        save_latents: Save latent tensors (.pt files).
        save_pixels: Save pixel tensors (.pt files).
        save_l2_maps: Save L2 norm maps for visualization.
        save_x_before_after: Save x before/after each step.
        active_latent_threshold: Threshold for active latent mask.
        active_pixel_threshold: Threshold for active pixel mask.
        top_n_coords: Number of top coordinates to extract for attribution.
    """
    
    archive_mode: str = "tier1"
    delta_dtype: torch.dtype = torch.float16
    
    save_deltas: bool = True
    save_frames: bool = False
    save_every: int = 1
    save_meta: bool = True
    save_latents: bool = True
    save_pixels: bool = False  # Derived from archive_mode by default
    save_l2_maps: bool = False  # Derived from archive_mode by default
    save_x_before_after: bool = False  # Derived from archive_mode by default
    
    # Attribution thresholds
    active_latent_threshold: float = 1e-4
    active_pixel_threshold: int = 1
    top_n_coords: int = 1000
    
    def __post_init__(self) -> None:
        """
        Apply archive mode defaults after initialization.
        """
        # In tier1 mode, disable expensive captures unless explicitly set
        if self.archive_mode == "tier1":
            # Don't override if user explicitly set them
            pass
        else:
            # tier2 or higher: enable full capture
            self.save_pixels = True
            self.save_l2_maps = True
            self.save_x_before_after = True


# =============================================================================
# Video Configuration
# =============================================================================

@dataclass
class VideoConfig:
    """
    Configuration for video encoding.
    
    Controls ffmpeg subprocess parameters for video output.
    
    Attributes:
        fps: Frames per second in output video.
        upscale: Upscale factor from latent to pixel resolution.
        codec: Video codec (default: libx264).
        pixel_format: Output pixel format (default: yuv420p).
        write_video: Write output video.
    """
    
    fps: int = 60
    upscale: int = 8
    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    write_video: bool = True


# =============================================================================
# Drift Configuration
# =============================================================================

@dataclass
class DriftConfig:
    """
    Configuration for state-space drift.
    
    Drift functions operate on x (the latent state) after each scheduler step.
    They modify the trajectory in state space, independent of score-space forces.
    
    Mathematical Interpretation:
    After scheduler step: x_{t-1} = scheduler_step(pred, t, x_t)
    After drift: x_{t-1} = drift(x_{t-1}, t, ...)
    
    Drift is applied in state-space (x), not score-space (pred).
    
    Attributes:
        enabled: Whether drift is active.
        function_name: Name of drift function from drift_zoo.
        noise_scale: Noise magnitude for stochastic drift.
        decay: Decay factor applied to x (0 = no decay, 1 = full decay).
        custom_cfg: Additional configuration passed to drift function.
    """
    
    enabled: bool = False
    function_name: str = "default"
    noise_scale: float = 0.03
    decay: float = 0.0
    custom_cfg: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for passing to drift functions.
        """
        cfg = {
            "noise_scale": self.noise_scale,
            "decay": self.decay,
        }
        cfg.update(self.custom_cfg)
        return cfg


# =============================================================================
# Force Configuration
# =============================================================================

@dataclass 
class ForceConfig:
    """
    Configuration for score-space forces.
    
    Forces operate on pred (the predicted noise / score) before each scheduler step.
    They modify the trajectory in score space, enabling class guidance, repulsion,
    blending, and complex multi-class dynamics.
    
    Mathematical Interpretation:
    Unconditional: pred_uncond = model(x_t, t, null_label)
    Conditional: pred_cond = model(x_t, t, target_label)
    Force to class: force = pred_cond - pred_uncond
    Final pred: pred = pred_uncond + Σ_i force_i
    
    Forces are additive in score space.
    
    Attributes:
        guidance_scale: Classifier-free guidance scale (0 = no guidance).
        force_stack: List of force specifications to compose.
        global_cfg: Global configuration merged into all force cfgs.
        clip_abs: Absolute value clipping for force magnitude.
        clip_norm: Norm clipping for force magnitude.
        enable_cache: Cache class predictions to avoid redundant UNet calls.
    """
    
    guidance_scale: float = 0.0
    force_stack: List[Dict[str, Any]] = field(default_factory=list)
    global_cfg: Dict[str, Any] = field(default_factory=dict)
    clip_abs: Optional[float] = None
    clip_norm: Optional[float] = None
    enable_cache: bool = False


# =============================================================================
# Environment Configuration
# =============================================================================

@dataclass
class EnvironmentConfig:
    """
    Runtime environment configuration.
    
    Contains paths, SLURM settings, and device configuration.
    These are typically read from environment variables or defaults.
    
    Attributes:
        cache_root: Root directory for checkpoints.
        output_root: Root directory for output files.
        log_dir: Directory for log files.
        variation: Training variation name for checkpoint discovery.
        device: Compute device ("cuda" or "cpu").
        slurm_rank: SLURM array task ID (for distributed sharding).
        slurm_world: SLURM array task count (total workers).
        batch_size: Number of videos to generate concurrently on one GPU.
                    Higher values improve GPU utilisation at the cost of
                    memory.  Default 1 = sequential (original behaviour).
    """
    
    # Default: project-level checkpoints/ directory (self-contained)
    cache_root: str = ""
    output_root: str = ""
    log_dir: str = ""
    variation: str = "balanced_cfg_cosine_ema"
    device: str = "cuda"
    slurm_rank: int = 0
    slurm_world: int = 1
    batch_size: int = 1
    
    def __post_init__(self) -> None:
        """
        Read environment variables for SLURM settings and resolve defaults.
        """
        self.slurm_rank = int(os.environ.get("SLURM_ARRAY_TASK_ID", self.slurm_rank))
        self.slurm_world = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", self.slurm_world))
        
        # Default cache_root: project-level checkpoints/ directory
        if not self.cache_root:
            self.cache_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "checkpoints",
            )
        
        # Default device selection
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"


# =============================================================================
# Starting-Image Bank Configuration
# =============================================================================

@dataclass
class StartingImageConfig:
    """
    Configuration for starting-image bank.

    When enabled, the generation loop loads a pre-selected latent tensor
    from *bank_dir* instead of sampling pure Gaussian noise.

    Attributes:
        enabled: Use starting-image bank instead of random noise.
        bank_dir: Directory containing ``.pt`` files (each shape 1,1,h,w).
        selection: How to pick an image: "random", "sequential", "fixed".
        fixed_index: Index to use when selection="fixed".
    """

    enabled: bool = False
    bank_dir: str = ""
    selection: str = "random"   # "random" | "sequential" | "fixed"
    fixed_index: int = 0


# =============================================================================
# Post-Hoc Scoring Configuration
# =============================================================================

@dataclass
class ScoringConfig:
    """
    Configuration for post-hoc evidence-accumulation scoring.

    When enabled, each video's diffusion trajectory is scored frame-by-frame
    against the final generated image using the listed scorer functions.

    Attributes:
        enabled: Run post-hoc scoring after generation.
        score_functions: List of scorer names from the scoring registry
                         (e.g. ["pearson_pixel", "pearson_latent"]).
    """

    enabled: bool = False
    score_functions: List[str] = field(default_factory=lambda: ["pearson_pixel", "pearson_latent"])


# =============================================================================
# Main Inference Configuration
# =============================================================================

@dataclass
class InferenceConfig:
    """
    Top-level configuration for EMNIST diffusion inference.
    
    This is the main configuration object passed to the inference engine.
    It contains all parameters needed to run generation, including:
    - What character to generate
    - How many videos to produce
    - Diffusion parameters (steps, seed, etc.)
    - Nested configurations for capture, video, drift, force, environment
    
    Attributes:
        split: Dataset split (e.g., "balanced").
        size_name: Model size preset ("small", "medium", "large", "xl").
        character: Character to generate (must exist in checkpoint classes).
        total_videos: Number of videos to generate.
        steps: Number of diffusion timesteps.
        eta: DDIM eta parameter (0 = deterministic).
        base_seed: Base random seed (vid_id is added for per-video seed).
        experiment_name: Optional experiment name for output organization.
        
        capture: CaptureConfig for data saving.
        video: VideoConfig for video encoding.
        drift: DriftConfig for state-space drift.
        force: ForceConfig for score-space forces.
        env: EnvironmentConfig for runtime settings.
    """
    
    # Model selection
    split: str = "balanced"
    size_name: str = "medium"
    
    # Checkpoint selection
    checkpoint_path: Optional[str] = None   # Direct path to a .pt checkpoint (overrides auto-discovery)
    checkpoint_epoch: Optional[int] = None  # Specific epoch number to load (None = latest)
    
    # Generation target
    character: str = "k"
    
    # Generation parameters
    total_videos: int = 105
    steps: int = 1000
    eta: float = 0.0
    base_seed: int = 1000
    
    # Experiment organization
    experiment_name: Optional[str] = None
    
    # Nested configurations
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    force: ForceConfig = field(default_factory=ForceConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    starting_image: StartingImageConfig = field(default_factory=StartingImageConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    
    def __post_init__(self) -> None:
        """
        Apply derived settings and validation.
        """
        # Auto-generate experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = f"{self.character}_{self.size_name}"
    
    def get_output_base(self) -> str:
        """
        Get the base output directory for this configuration.
        """
        if self.env.output_root:
            return os.path.join(self.env.output_root, self.size_name, self.character)
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output",
            self.size_name,
            self.character,
        )
    
    def get_checkpoint_search_dir(self) -> str:
        """
        Get the directory to search for checkpoints.
        
        Checkpoints live at: <cache_root>/<size_name>/
        """
        return os.path.join(
            self.env.cache_root,
            self.size_name,
        )
    
    def get_video_ids(self) -> List[int]:
        """
        Get the video IDs assigned to this worker (SLURM sharding).
        """
        return list(range(self.env.slurm_rank, self.total_videos, self.env.slurm_world))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize configuration to a JSON-compatible dictionary.
        """
        return {
            "split": self.split,
            "size_name": self.size_name,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_epoch": self.checkpoint_epoch,
            "character": self.character,
            "total_videos": self.total_videos,
            "steps": self.steps,
            "eta": self.eta,
            "base_seed": self.base_seed,
            "experiment_name": self.experiment_name,
            "capture": {
                "archive_mode": self.capture.archive_mode,
                "save_deltas": self.capture.save_deltas,
                "save_frames": self.capture.save_frames,
                "save_every": self.capture.save_every,
                "save_meta": self.capture.save_meta,
                "save_latents": self.capture.save_latents,
                "save_pixels": self.capture.save_pixels,
                "save_l2_maps": self.capture.save_l2_maps,
                "active_latent_threshold": self.capture.active_latent_threshold,
                "active_pixel_threshold": self.capture.active_pixel_threshold,
                "top_n_coords": self.capture.top_n_coords,
            },
            "video": {
                "fps": self.video.fps,
                "upscale": self.video.upscale,
                "codec": self.video.codec,
                "pixel_format": self.video.pixel_format,
            },
            "drift": self.drift.to_dict() if self.drift.enabled else None,
            "force": {
                "guidance_scale": self.force.guidance_scale,
                "force_stack": self.force.force_stack,
            },
            "starting_image": {
                "enabled": self.starting_image.enabled,
                "bank_dir": self.starting_image.bank_dir,
                "selection": self.starting_image.selection,
                "fixed_index": self.starting_image.fixed_index,
            },
            "scoring": {
                "enabled": self.scoring.enabled,
                "score_functions": self.scoring.score_functions,
            },
        }


# =============================================================================
# Configuration Builders
# =============================================================================

def config_from_dict(d: Dict[str, Any]) -> InferenceConfig:
    """
    Reconstruct InferenceConfig from a dictionary.
    
    Useful for loading saved configurations.
    
    Args:
        d: Dictionary with configuration values.
        
    Returns:
        InferenceConfig instance.
    """
    capture = CaptureConfig(
        archive_mode=d.get("capture", {}).get("archive_mode", "tier1"),
        save_deltas=d.get("capture", {}).get("save_deltas", True),
        save_frames=d.get("capture", {}).get("save_frames", False),
        save_every=d.get("capture", {}).get("save_every", 1),
        save_meta=d.get("capture", {}).get("save_meta", True),
        save_latents=d.get("capture", {}).get("save_latents", True),
        save_pixels=d.get("capture", {}).get("save_pixels", False),
        save_l2_maps=d.get("capture", {}).get("save_l2_maps", False),
        active_latent_threshold=d.get("capture", {}).get("active_latent_threshold", 1e-4),
        active_pixel_threshold=d.get("capture", {}).get("active_pixel_threshold", 1),
        top_n_coords=d.get("capture", {}).get("top_n_coords", 1000),
    )
    
    video = VideoConfig(
        fps=d.get("video", {}).get("fps", 60),
        upscale=d.get("video", {}).get("upscale", 8),
        codec=d.get("video", {}).get("codec", "libx264"),
        pixel_format=d.get("video", {}).get("pixel_format", "yuv420p"),
    )
    
    drift_cfg = d.get("drift", {}) or {}
    drift = DriftConfig(
        enabled=bool(drift_cfg),
        function_name=drift_cfg.get("function_name", "default"),
        noise_scale=drift_cfg.get("noise_scale", 0.03),
        decay=drift_cfg.get("decay", 0.0),
    )
    
    force_cfg = d.get("force", {}) or {}
    force = ForceConfig(
        guidance_scale=force_cfg.get("guidance_scale", 0.0),
        force_stack=force_cfg.get("force_stack", []),
    )
    
    si_cfg = d.get("starting_image", {}) or {}
    starting_image = StartingImageConfig(
        enabled=si_cfg.get("enabled", False),
        bank_dir=si_cfg.get("bank_dir", ""),
        selection=si_cfg.get("selection", "random"),
        fixed_index=si_cfg.get("fixed_index", 0),
    )

    sc_cfg = d.get("scoring", {}) or {}
    scoring = ScoringConfig(
        enabled=sc_cfg.get("enabled", False),
        score_functions=sc_cfg.get("score_functions", ["pearson_pixel", "pearson_latent"]),
    )

    return InferenceConfig(
        split=d.get("split", "balanced"),
        size_name=d.get("size_name", "medium"),
        character=d.get("character", "k"),
        total_videos=d.get("total_videos", 105),
        steps=d.get("steps", 1000),
        eta=d.get("eta", 0.0),
        base_seed=d.get("base_seed", 1000),
        experiment_name=d.get("experiment_name"),
        capture=capture,
        video=video,
        drift=drift,
        force=force,
        starting_image=starting_image,
        scoring=scoring,
    )
