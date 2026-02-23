"""
Metrics Tracker
===============

Tracks per-epoch and per-batch training metrics with automatic CSV and JSON
persistence. Designed for distributed training — only rank 0 should call
write methods.

Tracked Metrics:
    - Loss (mean, min, max, std, median per epoch)
    - Learning rate (supports schedulers)
    - Gradient norm (global L2)
    - Throughput (samples/sec, batches/sec)
    - GPU memory (allocated, reserved, peak)
    - EMA delta norm (drift between EMA and live weights)
    - Wall clock timing
    - Cumulative statistics

CSV columns are written incrementally (one row per epoch) so partial runs
are always recoverable.

Author: David Falk
Organization: APEX Laboratory, The University of Chicago
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import torch


# =========================================================================
# Per-Epoch Metrics Record
# =========================================================================

@dataclass
class EpochMetrics:
    """Single epoch's collected metrics."""

    epoch: int = 0

    # Loss statistics
    loss_mean: float = 0.0
    loss_std: float = 0.0
    loss_min: float = float("inf")
    loss_max: float = float("-inf")
    loss_median: float = 0.0

    # Learning rate
    lr: float = 0.0

    # Gradient norm (global L2 across all params)
    grad_norm: float = 0.0

    # Throughput
    epoch_time_sec: float = 0.0
    samples_per_sec: float = 0.0
    batches_per_sec: float = 0.0
    num_batches: int = 0
    num_samples: int = 0

    # GPU memory (bytes → MB for display, stored as MB)
    gpu_mem_allocated_mb: float = 0.0
    gpu_mem_reserved_mb: float = 0.0
    gpu_mem_peak_mb: float = 0.0

    # EMA
    ema_delta_norm: float = 0.0

    # Wall clock
    wall_clock_sec: float = 0.0
    cumulative_time_sec: float = 0.0
    cumulative_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================================
# CSV Column Order
# =========================================================================

CSV_COLUMNS = [
    "epoch",
    "loss_mean", "loss_std", "loss_min", "loss_max", "loss_median",
    "lr", "grad_norm",
    "epoch_time_sec", "samples_per_sec", "batches_per_sec",
    "num_batches", "num_samples",
    "gpu_mem_allocated_mb", "gpu_mem_reserved_mb", "gpu_mem_peak_mb",
    "ema_delta_norm",
    "wall_clock_sec", "cumulative_time_sec", "cumulative_samples",
]


# =========================================================================
# Metrics Tracker
# =========================================================================

class MetricsTracker:
    """
    Collects per-batch scalars, computes epoch-level statistics, and
    writes CSV + JSON artifacts.

    Parameters
    ----------
    output_dir : str
        Directory where metrics files are written.
    model_name : str
        Used in filenames (e.g. "balanced_cfg_cosine_ema_600_steps").
    csv_filename : str | None
        Override CSV filename (default: metrics_{model_name}.csv).
    json_filename : str | None
        Override JSON filename for full history dump.
    """

    def __init__(
        self,
        output_dir: str,
        model_name: str = "training",
        csv_filename: Optional[str] = None,
        json_filename: Optional[str] = None,
    ) -> None:
        self.output_dir = output_dir
        self.model_name = model_name

        os.makedirs(output_dir, exist_ok=True)

        self.csv_path = os.path.join(
            output_dir,
            csv_filename or f"metrics_{model_name}.csv",
        )
        self.json_path = os.path.join(
            output_dir,
            json_filename or f"metrics_{model_name}.json",
        )

        # Epoch-level history
        self.history: List[EpochMetrics] = []

        # Per-batch accumulators (reset each epoch)
        self._batch_losses: List[float] = []
        self._batch_grad_norms: List[float] = []
        self._batch_count: int = 0
        self._batch_sample_count: int = 0

        # Timing
        self._epoch_start: float = 0.0
        self._training_start: float = time.time()
        self._cumulative_samples: int = 0

        # Write CSV header
        self._write_csv_header()

    # ---- CSV header -------------------------------------------------------

    def _write_csv_header(self) -> None:
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

    # ---- Per-batch interface ----------------------------------------------

    def start_epoch(self) -> None:
        """Call at the start of each epoch."""
        self._batch_losses.clear()
        self._batch_grad_norms.clear()
        self._batch_count = 0
        self._batch_sample_count = 0
        self._epoch_start = time.time()
        # Reset peak memory tracking for this epoch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def log_batch(
        self,
        loss: float,
        batch_size: int,
        grad_norm: Optional[float] = None,
    ) -> None:
        """
        Record a single mini-batch.

        Parameters
        ----------
        loss : float
            Scalar loss for this batch.
        batch_size : int
            Number of samples in this batch.
        grad_norm : float | None
            Global L2 gradient norm (optional).
        """
        self._batch_losses.append(loss)
        self._batch_count += 1
        self._batch_sample_count += batch_size
        if grad_norm is not None:
            self._batch_grad_norms.append(grad_norm)

    # ---- End of epoch -----------------------------------------------------

    def end_epoch(
        self,
        epoch: int,
        lr: float,
        ema_unet: Optional[Any] = None,
        live_unet: Optional[Any] = None,
    ) -> EpochMetrics:
        """
        Finalize epoch metrics, write to CSV, and return the record.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        lr : float
            Current learning rate.
        ema_unet : nn.Module | None
            EMA model (for delta norm calculation).
        live_unet : nn.Module | None
            Training model (unwrapped, i.e. unet.module for DDP).

        Returns
        -------
        EpochMetrics
        """
        dt = time.time() - self._epoch_start
        n = len(self._batch_losses)

        # Sort for median / min / max
        sorted_losses = sorted(self._batch_losses) if n else [0.0]
        mid = n // 2

        m = EpochMetrics(
            epoch=epoch,
            loss_mean=sum(self._batch_losses) / n if n else 0.0,
            loss_std=_std(self._batch_losses) if n > 1 else 0.0,
            loss_min=sorted_losses[0] if n else 0.0,
            loss_max=sorted_losses[-1] if n else 0.0,
            loss_median=sorted_losses[mid] if n else 0.0,
            lr=lr,
            grad_norm=(
                sum(self._batch_grad_norms) / len(self._batch_grad_norms)
                if self._batch_grad_norms else 0.0
            ),
            epoch_time_sec=dt,
            samples_per_sec=self._batch_sample_count / dt if dt > 0 else 0.0,
            batches_per_sec=self._batch_count / dt if dt > 0 else 0.0,
            num_batches=self._batch_count,
            num_samples=self._batch_sample_count,
        )

        # GPU memory
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            m.gpu_mem_allocated_mb = torch.cuda.memory_allocated(dev) / 1e6
            m.gpu_mem_reserved_mb = torch.cuda.memory_reserved(dev) / 1e6
            m.gpu_mem_peak_mb = torch.cuda.max_memory_allocated(dev) / 1e6

        # EMA delta norm
        if ema_unet is not None and live_unet is not None:
            m.ema_delta_norm = _param_delta_norm(ema_unet, live_unet)

        # Cumulative
        self._cumulative_samples += self._batch_sample_count
        m.wall_clock_sec = time.time() - self._training_start
        m.cumulative_time_sec = m.wall_clock_sec
        m.cumulative_samples = self._cumulative_samples

        # Save
        self.history.append(m)
        self._append_csv(m)
        self._write_json()

        return m

    # ---- Persistence -------------------------------------------------------

    def _append_csv(self, m: EpochMetrics) -> None:
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([getattr(m, col) for col in CSV_COLUMNS])

    def _write_json(self) -> None:
        with open(self.json_path, "w") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "num_epochs": len(self.history),
                    "epochs": [m.to_dict() for m in self.history],
                },
                f,
                indent=2,
            )

    # ---- Accessors ---------------------------------------------------------

    def get_loss_history(self) -> List[float]:
        return [m.loss_mean for m in self.history]

    def get_lr_history(self) -> List[float]:
        return [m.lr for m in self.history]

    def get_throughput_history(self) -> List[float]:
        return [m.samples_per_sec for m in self.history]

    def get_gpu_mem_history(self) -> List[float]:
        return [m.gpu_mem_peak_mb for m in self.history]

    def get_grad_norm_history(self) -> List[float]:
        return [m.grad_norm for m in self.history]

    def get_epoch_list(self) -> List[int]:
        return [m.epoch for m in self.history]

    def summary(self) -> Dict[str, Any]:
        """Return a summary dictionary of the full training run."""
        if not self.history:
            return {}
        best = min(self.history, key=lambda m: m.loss_mean)
        last = self.history[-1]
        return {
            "total_epochs": len(self.history),
            "total_samples": self._cumulative_samples,
            "total_time_sec": last.wall_clock_sec,
            "total_time_human": _fmt_time(last.wall_clock_sec),
            "best_epoch": best.epoch,
            "best_loss": best.loss_mean,
            "final_loss": last.loss_mean,
            "final_lr": last.lr,
            "avg_throughput_sps": (
                sum(m.samples_per_sec for m in self.history)
                / len(self.history)
            ),
            "peak_gpu_mem_mb": max(
                m.gpu_mem_peak_mb for m in self.history
            ) if self.history else 0.0,
        }


# =========================================================================
# Helpers
# =========================================================================

def _std(vals: List[float]) -> float:
    """Population standard deviation."""
    n = len(vals)
    if n < 2:
        return 0.0
    mu = sum(vals) / n
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / n)


def _param_delta_norm(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
) -> float:
    """L2 norm of the difference between two models' parameters."""
    total = 0.0
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        total += float(
            torch.linalg.vector_norm(
                pa.detach().float() - pb.detach().float()
            ).item() ** 2
        )
    return math.sqrt(total)


def _fmt_time(seconds: float) -> str:
    """Format seconds to human-readable H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def compute_grad_norm(model: torch.nn.Module) -> float:
    """
    Compute global L2 gradient norm across all parameters.

    Call after loss.backward() but before optimizer.step().
    Safe for mixed-precision (casts to float32 internally).
    """
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().float().norm().item() ** 2)
    return math.sqrt(total)
