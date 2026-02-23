"""
Inference Plots
===============

Publication-quality visualizations for diffusion inference runs.

Generates the following plots:
    1.  Latent delta L2 vs. diffusion step (per video)
    2.  Pixel delta L2 vs. step
    3.  Alpha/beta schedule overlay
    4.  Generation time per video
    5.  Final image statistics across videos
    6.  Latent displacement heatmap (time × coord)
    7.  Cumulative L2 displacement
    8.  Step-level dashboard (multi-panel)
    9.  Video-level summary bar chart
    10. L2 velocity (derivative)

Author: David Falk
Organization: APEX Laboratory, The University of Chicago
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

if TYPE_CHECKING:
    from inference.metrics.tracker import InferenceMetricsTracker

# =========================================================================
# Style (matches training plots for visual consistency)
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
# Helpers
# =========================================================================

def _get_steps_for_video(tracker: "InferenceMetricsTracker", vid_id: int):
    """Get step metrics for a specific video."""
    return [s for s in tracker.step_history if s.vid_id == vid_id]


# =========================================================================
# Individual Plots
# =========================================================================

def plot_latent_l2_per_step(tracker: "InferenceMetricsTracker", save_dir: str) -> str:
    """Latent delta L2 norm vs. diffusion step for each video."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    for vm in tracker.video_history:
        steps = _get_steps_for_video(tracker, vm.vid_id)
        if not steps:
            continue
        idxs = [s.step_index for s in steps]
        l2s = [s.delta_x_l2 for s in steps]
        ax.plot(idxs, l2s, alpha=0.4, linewidth=0.8)

    # Mean line if multiple videos
    if len(tracker.video_history) > 1:
        all_vids = {}
        for s in tracker.step_history:
            all_vids.setdefault(s.step_index, []).append(s.delta_x_l2)
        mean_idxs = sorted(all_vids.keys())
        mean_vals = [np.mean(all_vids[i]) for i in mean_idxs]
        ax.plot(mean_idxs, mean_vals, color=ACCENT, linewidth=2, label="Mean")
        ax.legend(fontsize=9, framealpha=0.3)

    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Latent Δx L2")
    ax.set_title("Latent Delta L2 Norm per Step")
    ax.grid(True)

    path = os.path.join(save_dir, "latent_l2_per_step.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_pixel_l2_per_step(tracker: "InferenceMetricsTracker", save_dir: str) -> str:
    """Pixel delta L2 vs. step."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    for vm in tracker.video_history:
        steps = _get_steps_for_video(tracker, vm.vid_id)
        if not steps:
            continue
        idxs = [s.step_index for s in steps]
        l2s = [s.delta_pixel_l2 for s in steps]
        ax.plot(idxs, l2s, alpha=0.4, linewidth=0.8)

    if len(tracker.video_history) > 1:
        all_vids = {}
        for s in tracker.step_history:
            all_vids.setdefault(s.step_index, []).append(s.delta_pixel_l2)
        mean_idxs = sorted(all_vids.keys())
        mean_vals = [np.mean(all_vids[i]) for i in mean_idxs]
        ax.plot(mean_idxs, mean_vals, color=ACCENT3, linewidth=2, label="Mean")
        ax.legend(fontsize=9, framealpha=0.3)

    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Pixel Δ L2")
    ax.set_title("Pixel Delta L2 Norm per Step")
    ax.grid(True)

    path = os.path.join(save_dir, "pixel_l2_per_step.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_schedule_overlay(tracker: "InferenceMetricsTracker", save_dir: str) -> str:
    """Alpha and beta schedule from recorded steps."""
    _apply_style()

    # Use first video's steps
    if not tracker.video_history:
        return ""
    steps = _get_steps_for_video(tracker, tracker.video_history[0].vid_id)
    if not steps:
        return ""

    idxs = [s.step_index for s in steps]
    alphas = [s.alpha_cumprod_t for s in steps]
    betas = [s.beta_t for s in steps]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(idxs, alphas, color=ACCENT, label="ᾱ_t (alpha_cumprod)")
    ax1.set_xlabel("Diffusion Step")
    ax1.set_ylabel("ᾱ_t", color=ACCENT)
    ax1.tick_params(axis="y", labelcolor=ACCENT)

    ax2 = ax1.twinx()
    ax2.plot(idxs, betas, color=ACCENT2, label="β_t", alpha=0.7)
    ax2.set_ylabel("β_t", color=ACCENT2)
    ax2.tick_params(axis="y", labelcolor=ACCENT2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9, framealpha=0.3)

    ax1.set_title("Noise Schedule")
    ax1.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "schedule_overlay.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_generation_times(tracker: "InferenceMetricsTracker", save_dir: str) -> str:
    """Generation time per video."""
    _apply_style()
    if not tracker.video_history:
        return ""

    vids = [m.vid_id for m in tracker.video_history]
    times = [m.generation_time_sec for m in tracker.video_history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(vids, times, color=ACCENT5, alpha=0.7)
    ax.axhline(np.mean(times), color=ACCENT, linestyle="--", alpha=0.6,
               label=f"Avg: {np.mean(times):.1f}s")
    ax.set_xlabel("Video ID")
    ax.set_ylabel("Time (sec)")
    ax.set_title("Generation Time per Video")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True, axis="y")

    path = os.path.join(save_dir, "generation_times.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_final_image_stats(tracker: "InferenceMetricsTracker", save_dir: str) -> str:
    """Final image mean/std across videos."""
    _apply_style()
    if not tracker.video_history:
        return ""

    vids = [m.vid_id for m in tracker.video_history]
    means = [m.final_mean for m in tracker.video_history]
    stds = [m.final_std for m in tracker.video_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(vids, means, color=ACCENT, alpha=0.7)
    ax1.set_xlabel("Video ID")
    ax1.set_ylabel("Mean pixel value")
    ax1.set_title("Final Image Mean")
    ax1.grid(True, axis="y")

    ax2.bar(vids, stds, color=ACCENT4, alpha=0.7)
    ax2.set_xlabel("Video ID")
    ax2.set_ylabel("Std pixel value")
    ax2.set_title("Final Image Std")
    ax2.grid(True, axis="y")

    fig.tight_layout()
    path = os.path.join(save_dir, "final_image_stats.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_cumulative_l2(tracker: "InferenceMetricsTracker", save_dir: str) -> str:
    """Cumulative L2 displacement over steps."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    for vm in tracker.video_history:
        steps = _get_steps_for_video(tracker, vm.vid_id)
        if not steps:
            continue
        idxs = [s.step_index for s in steps]
        l2s = [s.delta_x_l2 for s in steps]
        cum = np.cumsum(l2s)
        ax.plot(idxs, cum, alpha=0.4, linewidth=0.8)

    if len(tracker.video_history) > 1:
        all_vids = {}
        for s in tracker.step_history:
            all_vids.setdefault(s.step_index, []).append(s.delta_x_l2)
        mean_idxs = sorted(all_vids.keys())
        mean_vals = [np.mean(all_vids[i]) for i in mean_idxs]
        cum_mean = np.cumsum(mean_vals)
        ax.plot(mean_idxs, cum_mean, color=ACCENT, linewidth=2, label="Mean cumulative")
        ax.legend(fontsize=9, framealpha=0.3)

    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Cumulative Latent L2")
    ax.set_title("Cumulative L2 Displacement")
    ax.grid(True)

    path = os.path.join(save_dir, "cumulative_l2.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_l2_velocity(tracker: "InferenceMetricsTracker", save_dir: str) -> str:
    """Rate of change of L2 (derivative)."""
    _apply_style()
    if not tracker.video_history:
        return ""

    steps = _get_steps_for_video(tracker, tracker.video_history[0].vid_id)
    if len(steps) < 2:
        return ""

    l2s = [s.delta_x_l2 for s in steps]
    velocity = [l2s[i] - l2s[i - 1] for i in range(1, len(l2s))]
    idxs = [s.step_index for s in steps][1:]

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = [ACCENT3 if v > 0 else ACCENT2 for v in velocity]
    ax.bar(idxs, velocity, color=colors, alpha=0.6, width=1.0)
    ax.axhline(0, color="#8b949e", linewidth=0.8)
    ax.set_xlabel("Diffusion Step")
    ax.set_ylabel("Δ L2")
    ax.set_title("L2 Velocity (rate of change)")
    ax.grid(True, axis="y")

    path = os.path.join(save_dir, "l2_velocity.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_video_summary(tracker: "InferenceMetricsTracker", save_dir: str) -> str:
    """Multi-metric summary per video."""
    _apply_style()
    if not tracker.video_history:
        return ""

    vids = [m.vid_id for m in tracker.video_history]
    latent_l2 = [m.total_latent_l2 for m in tracker.video_history]
    times = [m.generation_time_sec for m in tracker.video_history]
    gpu_mem = [m.gpu_mem_peak_mb for m in tracker.video_history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(vids, latent_l2, color=ACCENT, alpha=0.7)
    axes[0].set_title("Total Latent L2")
    axes[0].set_xlabel("Video ID")
    axes[0].grid(True, axis="y")

    axes[1].bar(vids, times, color=ACCENT5, alpha=0.7)
    axes[1].set_title("Gen Time (s)")
    axes[1].set_xlabel("Video ID")
    axes[1].grid(True, axis="y")

    axes[2].bar(vids, gpu_mem, color=ACCENT2, alpha=0.7)
    axes[2].set_title("GPU Peak (MB)")
    axes[2].set_xlabel("Video ID")
    axes[2].grid(True, axis="y")

    fig.tight_layout()
    path = os.path.join(save_dir, "video_summary.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_inference_dashboard(tracker: "InferenceMetricsTracker", save_dir: str) -> str:
    """Combined dashboard of inference metrics."""
    _apply_style()
    if not tracker.video_history or not tracker.step_history:
        return ""

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.30)

    # 1. Latent L2 per step
    ax1 = fig.add_subplot(gs[0, 0])
    for vm in tracker.video_history[:5]:
        steps = _get_steps_for_video(tracker, vm.vid_id)
        if steps:
            ax1.plot([s.step_index for s in steps], [s.delta_x_l2 for s in steps],
                     alpha=0.5, linewidth=0.8)
    ax1.set_title("Latent Δ L2")
    ax1.set_xlabel("Step")
    ax1.grid(True)

    # 2. Pixel L2 per step
    ax2 = fig.add_subplot(gs[0, 1])
    for vm in tracker.video_history[:5]:
        steps = _get_steps_for_video(tracker, vm.vid_id)
        if steps:
            ax2.plot([s.step_index for s in steps], [s.delta_pixel_l2 for s in steps],
                     alpha=0.5, linewidth=0.8)
    ax2.set_title("Pixel Δ L2")
    ax2.set_xlabel("Step")
    ax2.grid(True)

    # 3. Schedule
    ax3 = fig.add_subplot(gs[0, 2])
    steps0 = _get_steps_for_video(tracker, tracker.video_history[0].vid_id)
    if steps0:
        ax3.plot([s.step_index for s in steps0], [s.alpha_cumprod_t for s in steps0], color=ACCENT)
        ax3r = ax3.twinx()
        ax3r.plot([s.step_index for s in steps0], [s.beta_t for s in steps0], color=ACCENT2, alpha=0.7)
    ax3.set_title("Schedule (ᾱ, β)")
    ax3.grid(True, alpha=0.3)

    # 4. Generation times
    ax4 = fig.add_subplot(gs[1, 0])
    vids = [m.vid_id for m in tracker.video_history]
    gts = [m.generation_time_sec for m in tracker.video_history]
    ax4.bar(vids, gts, color=ACCENT5, alpha=0.7)
    ax4.set_title("Gen Time (s)")
    ax4.grid(True, axis="y")

    # 5. Final image stats
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(vids, [m.final_mean for m in tracker.video_history], color=ACCENT, alpha=0.7)
    ax5.set_title("Final Mean")
    ax5.grid(True, axis="y")

    # 6. Total latent L2
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(vids, [m.total_latent_l2 for m in tracker.video_history], color=ACCENT4, alpha=0.7)
    ax6.set_title("Total Latent L2")
    ax6.grid(True, axis="y")

    # 7. Cumulative L2
    ax7 = fig.add_subplot(gs[2, 0])
    for vm in tracker.video_history[:5]:
        steps = _get_steps_for_video(tracker, vm.vid_id)
        if steps:
            cum = np.cumsum([s.delta_x_l2 for s in steps])
            ax7.plot([s.step_index for s in steps], cum, alpha=0.5, linewidth=0.8)
    ax7.set_title("Cumulative L2")
    ax7.set_xlabel("Step")
    ax7.grid(True)

    # 8. GPU memory
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.bar(vids, [m.gpu_mem_peak_mb for m in tracker.video_history], color=ACCENT2, alpha=0.7)
    ax8.set_title("GPU Peak (MB)")
    ax8.grid(True, axis="y")

    # 9. L2 velocity
    ax9 = fig.add_subplot(gs[2, 2])
    if steps0 and len(steps0) > 1:
        l2s = [s.delta_x_l2 for s in steps0]
        vel = [l2s[i] - l2s[i-1] for i in range(1, len(l2s))]
        ax9.bar([s.step_index for s in steps0][1:], vel,
                color=[ACCENT3 if v > 0 else ACCENT2 for v in vel], alpha=0.6, width=1)
    ax9.set_title("L2 Velocity")
    ax9.set_xlabel("Step")
    ax9.grid(True, axis="y")

    fig.suptitle(
        f"Inference Dashboard — {tracker.character}",
        fontsize=15, color="#c9d1d9", y=0.98,
    )

    path = os.path.join(save_dir, "inference_dashboard.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# =========================================================================
# Public Entry Point
# =========================================================================

def generate_all_inference_plots(
    tracker: "InferenceMetricsTracker",
    save_dir: Optional[str] = None,
) -> List[str]:
    """
    Generate all inference plots and save to disk.

    Parameters
    ----------
    tracker : InferenceMetricsTracker
        Populated tracker with at least one video.
    save_dir : str | None
        Directory for plots. Defaults to tracker.output_dir/plots.

    Returns
    -------
    list[str]
        Paths of generated plot files.
    """
    if save_dir is None:
        save_dir = os.path.join(tracker.output_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)

    if not tracker.video_history:
        return []

    generators = [
        plot_latent_l2_per_step,
        plot_pixel_l2_per_step,
        plot_schedule_overlay,
        plot_generation_times,
        plot_final_image_stats,
        plot_cumulative_l2,
        plot_l2_velocity,
        plot_video_summary,
        plot_inference_dashboard,
    ]

    paths: List[str] = []
    for gen in generators:
        try:
            p = gen(tracker, save_dir)
            if p:
                paths.append(p)
        except Exception as e:
            print(f"[inference/plots] Warning: {gen.__name__} failed: {e}")

    return paths
