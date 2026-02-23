"""
Inference Metrics Tracker
=========================

Collects per-video and per-step metrics during diffusion inference.
Writes CSV and JSON artifacts for downstream analysis and visualization.

Tracked Metrics (per video):
    - Generation wall time
    - Seed, label, video ID
    - Final image statistics (mean, std, min, max)
    - Total L2 displacement in latent and pixel space
    - Number of captured frames
    - GPU memory at generation time

Tracked Metrics (per step — appended to step-level CSV):
    - Timestep value and index
    - Latent delta L2 norm
    - Pixel delta L2 norm
    - Latent statistics (mean, std, min, max)
    - Wall clock delta

Author: David Falk
Organization: APEX Laboratory, The University of Chicago
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from inference.generation_loop import GenerationResult, FrameData


# =========================================================================
# Per-Video Summary Record
# =========================================================================

@dataclass
class VideoMetrics:
    """Metrics for a single generated video."""

    vid_id: int = 0
    seed: int = 0
    label: int = 0
    character: str = ""

    # Timing
    generation_time_sec: float = 0.0
    wall_start: float = 0.0
    wall_end: float = 0.0

    # Step counts
    total_steps: int = 0
    captured_frames: int = 0

    # Final image stats
    final_mean: float = 0.0
    final_std: float = 0.0
    final_min: float = 0.0
    final_max: float = 0.0

    # L2 displacement totals
    total_latent_l2: float = 0.0
    total_pixel_l2: float = 0.0

    # GPU
    gpu_mem_peak_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================================
# Per-Step Record
# =========================================================================

@dataclass
class StepMetrics:
    """Metrics for a single diffusion step."""

    vid_id: int = 0
    step_index: int = 0
    timestep: int = 0
    wall_dt: float = 0.0

    # Latent delta
    delta_x_l2: float = 0.0
    delta_x_mean: float = 0.0
    delta_x_std: float = 0.0
    delta_x_min: float = 0.0
    delta_x_max: float = 0.0

    # Pixel delta
    delta_pixel_l2: float = 0.0

    # Scheduler
    alpha_t: float = 0.0
    beta_t: float = 0.0
    alpha_cumprod_t: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================================
# CSV Column Orders
# =========================================================================

VIDEO_CSV_COLUMNS = [
    "vid_id", "seed", "label", "character",
    "generation_time_sec",
    "total_steps", "captured_frames",
    "final_mean", "final_std", "final_min", "final_max",
    "total_latent_l2", "total_pixel_l2",
    "gpu_mem_peak_mb",
]

STEP_CSV_COLUMNS = [
    "vid_id", "step_index", "timestep", "wall_dt",
    "delta_x_l2", "delta_x_mean", "delta_x_std", "delta_x_min", "delta_x_max",
    "delta_pixel_l2",
    "alpha_t", "beta_t", "alpha_cumprod_t",
]


# =========================================================================
# Inference Metrics Tracker
# =========================================================================

class InferenceMetricsTracker:
    """
    Collects inference metrics and writes to disk.

    Parameters
    ----------
    output_dir : str
        Directory for metrics files.
    character : str
        Target character being generated.
    config_dict : dict | None
        Inference config to embed in JSON output.
    """

    def __init__(
        self,
        output_dir: str,
        character: str = "",
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.output_dir = output_dir
        self.character = character
        self.config_dict = config_dict or {}

        os.makedirs(output_dir, exist_ok=True)

        self.video_csv_path = os.path.join(output_dir, "video_metrics.csv")
        self.step_csv_path = os.path.join(output_dir, "step_metrics.csv")
        self.json_path = os.path.join(output_dir, "inference_metrics.json")

        self.video_history: List[VideoMetrics] = []
        self.step_history: List[StepMetrics] = []

        self._start_time = time.time()

        self._write_csv_headers()

    def _write_csv_headers(self) -> None:
        with open(self.video_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(VIDEO_CSV_COLUMNS)
        with open(self.step_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(STEP_CSV_COLUMNS)

    # ---- Record a completed video ----------------------------------------

    def record_video(
        self,
        result: "GenerationResult",
        vid_id: int,
        character: str,
        config: Optional[Any] = None,
    ) -> VideoMetrics:
        """
        Record metrics for a completed video generation.

        Parameters
        ----------
        result : GenerationResult
            Output of generate_sample().
        vid_id : int
            Video ID number.
        character : str
            Target character.
        config : InferenceConfig | None
            Config (for step count).

        Returns
        -------
        VideoMetrics
        """
        import torch

        m = VideoMetrics(
            vid_id=vid_id,
            seed=result.seed,
            label=result.label,
            character=character,
            generation_time_sec=result.total_time,
            captured_frames=len(result.frames),
        )

        # Final image stats
        x = result.x_final.float()
        m.final_mean = float(x.mean().item())
        m.final_std = float(x.std().item())
        m.final_min = float(x.min().item())
        m.final_max = float(x.max().item())

        # L2 totals
        if result.delta_x_l2:
            m.total_latent_l2 = sum(result.delta_x_l2)
        if result.delta_pixel_l2:
            m.total_pixel_l2 = sum(result.delta_pixel_l2)
        # GPU
        if torch.cuda.is_available():
            m.gpu_mem_peak_mb = torch.cuda.max_memory_allocated() / 1e6

        # Step count
        if config is not None:
            m.total_steps = config.steps

        self.video_history.append(m)
        self._append_video_csv(m)

        # Record per-step metrics from frames
        for f in result.frames:
            step_m = StepMetrics(
                vid_id=vid_id,
                step_index=f.step_index,
                timestep=f.timestep,
                wall_dt=f.wall_dt,
                alpha_t=f.alpha_t or 0.0,
                beta_t=f.beta_t or 0.0,
                alpha_cumprod_t=f.alpha_cumprod_t or 0.0,
            )
            if f.stats_delta_x:
                step_m.delta_x_mean = f.stats_delta_x.get("mean", 0.0)
                step_m.delta_x_std = f.stats_delta_x.get("std", 0.0)
                step_m.delta_x_min = f.stats_delta_x.get("min", 0.0)
                step_m.delta_x_max = f.stats_delta_x.get("max", 0.0)
            self.step_history.append(step_m)
            self._append_step_csv(step_m)

        # Update L2 values from result lists
        if result.delta_x_l2 and len(result.delta_x_l2) == len(result.frames):
            for i, f in enumerate(result.frames):
                idx = len(self.step_history) - len(result.frames) + i
                if idx >= 0:
                    self.step_history[idx].delta_x_l2 = result.delta_x_l2[i]
        if result.delta_pixel_l2 and len(result.delta_pixel_l2) == len(result.frames):
            for i, f in enumerate(result.frames):
                idx = len(self.step_history) - len(result.frames) + i
                if idx >= 0:
                    self.step_history[idx].delta_pixel_l2 = result.delta_pixel_l2[i]
        # Rewrite step CSV with corrected L2 values
        self._rewrite_step_csv()

        # Write JSON
        self._write_json()

        return m

    # ---- CSV I/O ---------------------------------------------------------

    def _append_video_csv(self, m: VideoMetrics) -> None:
        with open(self.video_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([getattr(m, col) for col in VIDEO_CSV_COLUMNS])

    def _append_step_csv(self, m: StepMetrics) -> None:
        with open(self.step_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([getattr(m, col) for col in STEP_CSV_COLUMNS])

    def _rewrite_step_csv(self) -> None:
        """Rewrite the full step CSV (needed after L2 correction)."""
        with open(self.step_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(STEP_CSV_COLUMNS)
            for m in self.step_history:
                writer.writerow([getattr(m, col) for col in STEP_CSV_COLUMNS])

    def _write_json(self) -> None:
        data = {
            "character": self.character,
            "config": self.config_dict,
            "total_videos": len(self.video_history),
            "total_wall_time_sec": time.time() - self._start_time,
            "videos": [m.to_dict() for m in self.video_history],
            "summary": self.summary(),
        }
        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=2)

    # ---- Summary ---------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Aggregate summary across all recorded videos."""
        if not self.video_history:
            return {}
        times = [m.generation_time_sec for m in self.video_history]
        latent_l2s = [m.total_latent_l2 for m in self.video_history]
        return {
            "num_videos": len(self.video_history),
            "total_wall_sec": time.time() - self._start_time,
            "avg_gen_time_sec": sum(times) / len(times),
            "min_gen_time_sec": min(times),
            "max_gen_time_sec": max(times),
            "avg_total_latent_l2": sum(latent_l2s) / len(latent_l2s) if latent_l2s else 0.0,
            "gpu_peak_mb": max(m.gpu_mem_peak_mb for m in self.video_history),
        }
