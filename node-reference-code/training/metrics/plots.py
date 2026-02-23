"""
Training Plots
==============

Publication-quality training visualizations generated from MetricsTracker
history. All plots use matplotlib with a clean, consistent style suitable
for papers and HuggingFace model cards.

Generates the following plots:
    1.  Loss curve (linear)
    2.  Loss curve (log scale)
    3.  Loss distribution per epoch (box/violin)
    4.  Learning rate schedule
    5.  Gradient norm over epochs
    6.  Throughput (samples/sec)
    7.  GPU memory usage
    8.  EMA vs live weight divergence
    9.  Epoch timing
    10. Loss rate of change (derivative)
    11. Combined dashboard (multi-panel overview)

Author: David Falk
Organization: APEX Laboratory, The University of Chicago
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for SLURM
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

if TYPE_CHECKING:
    from metrics.tracker import MetricsTracker

# =========================================================================
# Style
# =========================================================================

STYLE = {
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "lines.linewidth": 1.8,
    "font.family": "monospace",
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0e1117",
}

ACCENT = "#58a6ff"
ACCENT2 = "#f78166"
ACCENT3 = "#7ee787"
ACCENT4 = "#d2a8ff"
ACCENT5 = "#ffa657"
FILL_ALPHA = 0.15


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


# =========================================================================
# Individual Plots
# =========================================================================

def plot_loss_linear(tracker: MetricsTracker, save_dir: str) -> str:
    """Training loss (linear scale) with min/max band."""
    _apply_style()
    epochs = tracker.get_epoch_list()
    means = tracker.get_loss_history()
    mins = [m.loss_min for m in tracker.history]
    maxs = [m.loss_max for m in tracker.history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(epochs, mins, maxs, alpha=FILL_ALPHA, color=ACCENT)
    ax.plot(epochs, means, color=ACCENT, label="Mean loss")
    ax.plot(epochs, mins, color=ACCENT3, linewidth=0.8, alpha=0.5, label="Min batch loss")
    ax.plot(epochs, maxs, color=ACCENT2, linewidth=0.8, alpha=0.5, label="Max batch loss")

    # Best epoch marker
    best_idx = int(np.argmin(means))
    ax.axvline(epochs[best_idx], color=ACCENT3, linestyle="--", alpha=0.4)
    ax.scatter([epochs[best_idx]], [means[best_idx]], color=ACCENT3, s=60, zorder=5,
               label=f"Best (epoch {epochs[best_idx]}, {means[best_idx]:.6f})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.3)
    ax.grid(True)

    path = os.path.join(save_dir, "loss_linear.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_loss_log(tracker: MetricsTracker, save_dir: str) -> str:
    """Training loss (log scale)."""
    _apply_style()
    epochs = tracker.get_epoch_list()
    means = tracker.get_loss_history()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(epochs, means, color=ACCENT, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log)")
    ax.set_title("Training Loss (Log Scale)")
    ax.grid(True, which="both", alpha=0.3)

    path = os.path.join(save_dir, "loss_log.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_loss_distribution(tracker: MetricsTracker, save_dir: str) -> str:
    """Loss mean +/- std as error bars."""
    _apply_style()
    epochs = tracker.get_epoch_list()
    means = tracker.get_loss_history()
    stds = [m.loss_std for m in tracker.history]

    fig, ax = plt.subplots(figsize=(10, 5))
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.fill_between(epochs, means_arr - stds_arr, means_arr + stds_arr,
                     alpha=FILL_ALPHA, color=ACCENT4)
    ax.plot(epochs, means, color=ACCENT4, label="Mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Loss Distribution (mean ± std)")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True)

    path = os.path.join(save_dir, "loss_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_learning_rate(tracker: MetricsTracker, save_dir: str) -> str:
    """Learning rate over epochs."""
    _apply_style()
    epochs = tracker.get_epoch_list()
    lrs = tracker.get_lr_history()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, lrs, color=ACCENT5, marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(-4, -4))
    ax.grid(True)

    path = os.path.join(save_dir, "learning_rate.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_grad_norm(tracker: MetricsTracker, save_dir: str) -> str:
    """Gradient norm over epochs."""
    _apply_style()
    epochs = tracker.get_epoch_list()
    norms = tracker.get_grad_norm_history()

    if all(n == 0.0 for n in norms):
        return ""  # Skip if not tracked

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, norms, color=ACCENT2, marker="^", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Global L2 Grad Norm")
    ax.set_title("Gradient Norm")
    ax.grid(True)

    path = os.path.join(save_dir, "grad_norm.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_throughput(tracker: MetricsTracker, save_dir: str) -> str:
    """Training throughput (samples/sec)."""
    _apply_style()
    epochs = tracker.get_epoch_list()
    sps = tracker.get_throughput_history()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(epochs, sps, color=ACCENT3, alpha=0.7, width=0.8)
    ax.axhline(np.mean(sps), color=ACCENT, linestyle="--", alpha=0.6,
               label=f"Avg: {np.mean(sps):.0f} samp/s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Samples / sec")
    ax.set_title("Training Throughput")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True, axis="y")

    path = os.path.join(save_dir, "throughput.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_gpu_memory(tracker: MetricsTracker, save_dir: str) -> str:
    """GPU memory usage over epochs."""
    _apply_style()
    epochs = tracker.get_epoch_list()
    allocated = [m.gpu_mem_allocated_mb for m in tracker.history]
    reserved = [m.gpu_mem_reserved_mb for m in tracker.history]
    peak = tracker.get_gpu_mem_history()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, peak, color=ACCENT2, label="Peak allocated", marker="v", markersize=3)
    ax.plot(epochs, allocated, color=ACCENT, label="Current allocated", marker="o", markersize=3)
    ax.plot(epochs, reserved, color=ACCENT4, label="Reserved", marker="s", markersize=3, alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("GPU Memory Usage")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True)

    path = os.path.join(save_dir, "gpu_memory.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_ema_divergence(tracker: MetricsTracker, save_dir: str) -> str:
    """EMA vs live model parameter divergence."""
    _apply_style()
    epochs = tracker.get_epoch_list()
    deltas = [m.ema_delta_norm for m in tracker.history]

    if all(d == 0.0 for d in deltas):
        return ""

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, deltas, color=ACCENT4, marker="D", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Parameter Delta")
    ax.set_title("EMA ↔ Live Weight Divergence")
    ax.grid(True)

    path = os.path.join(save_dir, "ema_divergence.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_epoch_timing(tracker: MetricsTracker, save_dir: str) -> str:
    """Per-epoch wall clock time."""
    _apply_style()
    epochs = tracker.get_epoch_list()
    times = [m.epoch_time_sec for m in tracker.history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(epochs, times, color=ACCENT5, alpha=0.7, width=0.8)
    ax.axhline(np.mean(times), color=ACCENT, linestyle="--", alpha=0.6,
               label=f"Avg: {np.mean(times):.1f}s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (sec)")
    ax.set_title("Epoch Duration")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True, axis="y")

    path = os.path.join(save_dir, "epoch_timing.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_loss_derivative(tracker: MetricsTracker, save_dir: str) -> str:
    """Rate of change of loss (finite difference)."""
    _apply_style()
    means = tracker.get_loss_history()
    if len(means) < 2:
        return ""

    deltas = [means[i] - means[i - 1] for i in range(1, len(means))]
    epochs = tracker.get_epoch_list()[1:]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [ACCENT3 if d < 0 else ACCENT2 for d in deltas]
    ax.bar(epochs, deltas, color=colors, alpha=0.7, width=0.8)
    ax.axhline(0, color="#8b949e", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Δ Loss")
    ax.set_title("Loss Rate of Change (green = improving)")
    ax.grid(True, axis="y")

    path = os.path.join(save_dir, "loss_derivative.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_dashboard(tracker: MetricsTracker, save_dir: str) -> str:
    """Combined multi-panel dashboard."""
    _apply_style()
    epochs = tracker.get_epoch_list()
    if not epochs:
        return ""

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.30)

    means = tracker.get_loss_history()
    mins = [m.loss_min for m in tracker.history]
    maxs = [m.loss_max for m in tracker.history]
    stds = [m.loss_std for m in tracker.history]
    lrs = tracker.get_lr_history()
    norms = tracker.get_grad_norm_history()
    sps = tracker.get_throughput_history()
    peak_mem = tracker.get_gpu_mem_history()
    times = [m.epoch_time_sec for m in tracker.history]
    ema_d = [m.ema_delta_norm for m in tracker.history]

    # 1. Loss linear
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(epochs, mins, maxs, alpha=FILL_ALPHA, color=ACCENT)
    ax1.plot(epochs, means, color=ACCENT)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.grid(True)

    # 2. Loss log
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(epochs, means, color=ACCENT)
    ax2.set_title("Loss (log)")
    ax2.set_xlabel("Epoch")
    ax2.grid(True, which="both", alpha=0.3)

    # 3. Loss distribution
    ax3 = fig.add_subplot(gs[0, 2])
    arr_m = np.array(means)
    arr_s = np.array(stds)
    ax3.fill_between(epochs, arr_m - arr_s, arr_m + arr_s, alpha=FILL_ALPHA, color=ACCENT4)
    ax3.plot(epochs, means, color=ACCENT4)
    ax3.set_title("Loss ± std")
    ax3.set_xlabel("Epoch")
    ax3.grid(True)

    # 4. LR
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, lrs, color=ACCENT5, marker="s", markersize=2)
    ax4.set_title("Learning Rate")
    ax4.set_xlabel("Epoch")
    ax4.grid(True)

    # 5. Grad norm
    ax5 = fig.add_subplot(gs[1, 1])
    if any(n != 0 for n in norms):
        ax5.plot(epochs, norms, color=ACCENT2)
    ax5.set_title("Grad Norm")
    ax5.set_xlabel("Epoch")
    ax5.grid(True)

    # 6. Throughput
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(epochs, sps, color=ACCENT3, alpha=0.7, width=0.8)
    ax6.set_title("Throughput (samp/s)")
    ax6.set_xlabel("Epoch")
    ax6.grid(True, axis="y")

    # 7. GPU mem
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(epochs, peak_mem, color=ACCENT2)
    ax7.set_title("Peak GPU Mem (MB)")
    ax7.set_xlabel("Epoch")
    ax7.grid(True)

    # 8. EMA divergence
    ax8 = fig.add_subplot(gs[2, 1])
    if any(d != 0 for d in ema_d):
        ax8.plot(epochs, ema_d, color=ACCENT4)
    ax8.set_title("EMA Divergence")
    ax8.set_xlabel("Epoch")
    ax8.grid(True)

    # 9. Epoch timing
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.bar(epochs, times, color=ACCENT5, alpha=0.7, width=0.8)
    ax9.set_title("Epoch Duration (s)")
    ax9.set_xlabel("Epoch")
    ax9.grid(True, axis="y")

    fig.suptitle(
        f"{tracker.model_name} — Training Dashboard",
        fontsize=15,
        color="#c9d1d9",
        y=0.98,
    )

    path = os.path.join(save_dir, "dashboard.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# =========================================================================
# Public Entry Point
# =========================================================================

def generate_all_plots(
    tracker: MetricsTracker,
    save_dir: Optional[str] = None,
) -> List[str]:
    """
    Generate all training plots and save to disk.

    Parameters
    ----------
    tracker : MetricsTracker
        Populated tracker with at least one epoch of data.
    save_dir : str | None
        Directory for plot images. Defaults to tracker.output_dir / plots.

    Returns
    -------
    list[str]
        Paths of generated plot files.
    """
    if save_dir is None:
        save_dir = os.path.join(tracker.output_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    if not tracker.history:
        return []

    generators = [
        plot_loss_linear,
        plot_loss_log,
        plot_loss_distribution,
        plot_learning_rate,
        plot_grad_norm,
        plot_throughput,
        plot_gpu_memory,
        plot_ema_divergence,
        plot_epoch_timing,
        plot_loss_derivative,
        plot_dashboard,
    ]

    paths: List[str] = []
    for gen in generators:
        try:
            p = gen(tracker, save_dir)
            if p:
                paths.append(p)
        except Exception as e:
            print(f"[metrics/plots] Warning: {gen.__name__} failed: {e}")

    return paths
