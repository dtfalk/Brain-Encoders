"""
Post-Hoc Scoring & Evidence-Accumulation Analysis
===================================================

Reconstructs the full diffusion trajectory from tier-1 data
(``x_init`` + ``delta_x_stack``), scores every intermediate frame
against the **final** generated frame with one or more registered
scorers, and saves results + plots.

Typical usage (called automatically when ``ScoringConfig.enabled`` is True)::

    from inference.posthoc_analysis import run_posthoc_analysis

    run_posthoc_analysis(
        vid_root="output/small/k/vid_000",
        score_fn_names=["pearson_pixel", "pearson_latent"],
    )

Produces:
- ``meta/scores_vid_NNN.csv`` — per-step scores
- ``meta/scores_vid_NNN.pt``  — raw score tensors
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from inference.scoring import get_scorer, SCORER_DIM_LABELS


# ─────────────────────────────────────────────────────────────────────
# Trajectory Reconstruction
# ─────────────────────────────────────────────────────────────────────

def reconstruct_trajectory(
    x_init: torch.Tensor,
    delta_x_stack: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct the full latent trajectory from tier-1 data.

    Args:
        x_init: Initial noise tensor, shape ``(1, 1, h, w)``.
        delta_x_stack: Per-step deltas, shape ``(N, 1, h, w)`` in float16.

    Returns:
        Tensor of shape ``(N, 1, h, w)`` where ``trajectory[k]`` is the
        latent state after applying the first ``k+1`` deltas.
    """
    cumulative = torch.cumsum(delta_x_stack.float(), dim=0)
    return x_init.float() + cumulative  # (N, 1, h, w)


def latent_to_pixel(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a latent tensor to pixel values in [0, 255] uint8.

    Args:
        x: Latent tensor of shape ``(1, 1, h, w)`` or ``(h, w)``.

    Returns:
        ``torch.uint8`` tensor of shape ``(h, w)``.
    """
    if x.dim() == 4:
        x = x[0, 0]
    elif x.dim() == 3:
        x = x[0]
    px = ((x.float() + 1) / 2).clamp(0, 1) * 255
    return px.round().to(torch.uint8)


# ─────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────

def compute_scores(
    trajectory: torch.Tensor,
    score_fn_names: Sequence[str],
) -> Dict[str, List[float]]:
    """
    Score every frame in *trajectory* against the last frame.

    Args:
        trajectory: ``(N, 1, h, w)`` latent trajectory.
        score_fn_names: Names of registered scorers to apply.

    Returns:
        ``{scorer_name: [score_0, score_1, …, score_{N-1}]}``
    """
    n_frames = trajectory.shape[0]
    ref_latent = trajectory[-1]          # (1, h, w)
    ref_pixel = latent_to_pixel(ref_latent)    # (h, w)

    results: Dict[str, List[float]] = {name: [] for name in score_fn_names}

    for k in range(n_frames):
        frame_latent = trajectory[k]     # (1, h, w)
        frame_pixel = latent_to_pixel(frame_latent)  # (h, w)

        for name in score_fn_names:
            scorer = get_scorer(name)
            # Route to the right space based on scorer name convention
            if "latent" in name:
                score = scorer(frame_latent, ref_latent)
            else:
                score = scorer(frame_pixel, ref_pixel)
            results[name].append(score)

    return results


# ─────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────

def save_scores_csv(
    scores: Dict[str, List[float]],
    path: str,
) -> None:
    """Write scores to a CSV file with columns: step, scorer1, scorer2, …"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    names = sorted(scores.keys())
    n = len(next(iter(scores.values())))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + names)
        for k in range(n):
            row = [k] + [scores[name][k] for name in names]
            writer.writerow(row)


def save_scores_pt(
    scores: Dict[str, List[float]],
    path: str,
) -> None:
    """Save scores as a ``.pt`` file (dict of float tensors)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {name: torch.tensor(vals) for name, vals in scores.items()}
    torch.save(payload, path)


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_evidence_curves(
    all_scores: Dict[str, torch.Tensor],
    out_dir: str,
    title_prefix: str = "",
) -> List[str]:
    """
    Plot mean ± std evidence-accumulation curves across multiple videos.

    Args:
        all_scores: ``{scorer_name: (n_videos, n_steps)}`` tensors.
        out_dir: Directory to write PNG files into.
        title_prefix: Optional prefix for plot titles (e.g. "small / k").

    Returns:
        List of saved file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []

    for name, data in all_scores.items():
        # data shape: (n_videos, n_steps)
        mean = data.mean(dim=0).numpy()
        std = data.std(dim=0).numpy()
        steps = np.arange(len(mean))
        dim_label = SCORER_DIM_LABELS.get(name, "")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, mean, linewidth=1.5, label="mean")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.25, label="±1 std")
        ax.set_xlabel("Diffusion step")
        ax.set_ylabel(name)
        title = f"{title_prefix}  {name}" if title_prefix else name
        if dim_label:
            title += f"\n[{dim_label}]"
        ax.set_title(title, fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fname = f"evidence_{name}.png"
        fpath = os.path.join(out_dir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(fpath)

    return saved


def plot_single_video_evidence(
    scores: Dict[str, List[float]],
    out_dir: str,
    vid_tag: str,
    title_prefix: str = "",
) -> List[str]:
    """
    Plot per-video evidence curves (one PNG per scorer).

    Args:
        scores: ``{scorer_name: [score_0, …, score_{N-1}]}``
        out_dir: Directory to write PNG files.
        vid_tag: Video identifier, e.g. ``"vid_000"``.
        title_prefix: Optional prefix for the plot title.

    Returns:
        List of saved file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []

    for name, vals in scores.items():
        steps = np.arange(len(vals))
        dim_label = SCORER_DIM_LABELS.get(name, "")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, vals, linewidth=1.0)
        ax.set_xlabel("Diffusion step")
        ax.set_ylabel(name)
        title = f"{title_prefix}  {vid_tag}  {name}" if title_prefix else f"{vid_tag}  {name}"
        if dim_label:
            title += f"\n[{dim_label}]"
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)

        fname = f"evidence_{name}_{vid_tag}.png"
        fpath = os.path.join(out_dir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(fpath)

    return saved


# ─────────────────────────────────────────────────────────────────────
# High-level Entry Point
# ─────────────────────────────────────────────────────────────────────

def run_posthoc_for_video(
    vid_root: str,
    score_fn_names: Sequence[str],
) -> Optional[Dict[str, List[float]]]:
    """
    Run post-hoc scoring for a single video directory.

    Looks for ``meta/latent_vid_*.pt`` inside *vid_root*, reconstructs
    the trajectory, and saves scores alongside the latent file.

    Returns:
        Score dict if successful, ``None`` if latent file not found.
    """
    meta_dir = os.path.join(vid_root, "meta")
    if not os.path.isdir(meta_dir):
        return None

    # Find latent .pt file
    latent_files = [
        f for f in os.listdir(meta_dir)
        if f.startswith("latent_") and f.endswith(".pt")
    ]
    if not latent_files:
        return None

    latent_path = os.path.join(meta_dir, latent_files[0])
    payload = torch.load(latent_path, map_location="cpu", weights_only=False)

    x_init = payload["tensors"]["x_init"]        # (1, 1, h, w)
    delta_x = payload["tensors"]["delta_x"]       # (N, 1, h, w)
    if delta_x is None:
        return None

    trajectory = reconstruct_trajectory(x_init, delta_x)
    scores = compute_scores(trajectory, score_fn_names)

    # Derive vid_tag from directory name
    vid_tag = os.path.basename(vid_root)
    csv_path = os.path.join(meta_dir, f"scores_{vid_tag}.csv")
    pt_path = os.path.join(meta_dir, f"scores_{vid_tag}.pt")
    save_scores_csv(scores, csv_path)
    save_scores_pt(scores, pt_path)

    # Per-video evidence plots
    plot_single_video_evidence(
        scores=scores,
        out_dir=meta_dir,
        vid_tag=vid_tag,
    )

    return scores


def run_posthoc_analysis(
    out_base: str,
    score_fn_names: Sequence[str],
    size_name: str = "",
    character: str = "",
) -> List[str]:
    """
    Run post-hoc scoring for *all* videos under *out_base* and
    generate aggregate evidence-accumulation plots.

    Args:
        out_base: Character-level output dir, e.g.
                  ``output/small/k/``.
        score_fn_names: Scorer names to use.
        size_name: For plot titles.
        character: For plot titles.

    Returns:
        List of saved plot file paths.
    """
    # Discover video directories
    if not os.path.isdir(out_base):
        return []

    vid_dirs = sorted([
        os.path.join(out_base, d)
        for d in os.listdir(out_base)
        if d.startswith("vid_") and os.path.isdir(os.path.join(out_base, d))
    ])

    if not vid_dirs:
        return []

    # Score each video
    all_scores_lists: Dict[str, List[List[float]]] = {
        name: [] for name in score_fn_names
    }
    for vdir in vid_dirs:
        scores = run_posthoc_for_video(vdir, score_fn_names)
        if scores is None:
            continue
        for name in score_fn_names:
            if name in scores:
                all_scores_lists[name].append(scores[name])

    if not any(all_scores_lists.values()):
        return []

    # Stack into tensors (n_videos, n_steps) — handle unequal lengths
    all_scores_tensors: Dict[str, torch.Tensor] = {}
    for name, lists in all_scores_lists.items():
        if not lists:
            continue
        min_len = min(len(l) for l in lists)
        truncated = [l[:min_len] for l in lists]
        all_scores_tensors[name] = torch.tensor(truncated)

    # Plot
    metrics_dir = os.path.join(out_base, "metrics")
    prefix = f"{size_name} / {character}" if size_name and character else ""
    return plot_evidence_curves(all_scores_tensors, metrics_dir, title_prefix=prefix)
