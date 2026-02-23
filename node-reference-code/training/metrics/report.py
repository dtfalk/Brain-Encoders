"""
Training Report Generator
==========================

Generates a Markdown summary report of the training run with embedded
metrics, hyperparameters, and references to saved plots. Suitable for
HuggingFace model cards and research documentation.

Author: David Falk
Organization: APEX Laboratory, The University of Chicago
"""

from __future__ import annotations

import json
import os
import platform
import socket
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from metrics.tracker import MetricsTracker


def generate_report(
    tracker: "MetricsTracker",
    config: Dict[str, Any],
    save_dir: Optional[str] = None,
    plot_paths: Optional[List[str]] = None,
) -> str:
    """
    Generate a Markdown training report.

    Parameters
    ----------
    tracker : MetricsTracker
        Populated metrics tracker.
    config : dict
        Training hyperparameters / configuration dictionary.
    save_dir : str | None
        Where to write the report. Defaults to tracker.output_dir.
    plot_paths : list[str] | None
        Paths to generated plots (for image links in the report).

    Returns
    -------
    str
        Path to the saved Markdown file.
    """
    if save_dir is None:
        save_dir = tracker.output_dir
    os.makedirs(save_dir, exist_ok=True)

    report_path = os.path.join(save_dir, f"training_report_{tracker.model_name}.md")
    summary = tracker.summary()

    lines: List[str] = []
    _h = lines.append

    # ---- Header -----------------------------------------------------------
    _h(f"# Training Report: {tracker.model_name}")
    _h("")
    _h(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _h(f"**Host:** {socket.gethostname()}")
    _h(f"**Platform:** {platform.platform()}")
    _h(f"**PyTorch:** {torch.__version__}")
    if torch.cuda.is_available():
        _h(f"**GPU:** {torch.cuda.get_device_name(0)} × {torch.cuda.device_count()}")
    _h("")

    # ---- Summary ----------------------------------------------------------
    _h("## Summary")
    _h("")
    _h("| Metric | Value |")
    _h("|--------|-------|")
    for k, v in summary.items():
        if isinstance(v, float):
            _h(f"| {k} | {v:.6f} |")
        else:
            _h(f"| {k} | {v} |")
    _h("")

    # ---- Configuration ----------------------------------------------------
    _h("## Hyperparameters")
    _h("")
    _h("```json")
    _h(json.dumps(config, indent=2, default=str))
    _h("```")
    _h("")

    # ---- Per-Epoch Table --------------------------------------------------
    _h("## Per-Epoch Metrics")
    _h("")
    _h("| Epoch | Loss (mean) | Loss (std) | Loss (min) | Loss (max) | LR | Grad Norm | Samples/s | GPU Peak MB | EMA Δ | Time (s) |")
    _h("|-------|-------------|------------|------------|------------|-----|-----------|-----------|-------------|-------|----------|")
    for m in tracker.history:
        _h(
            f"| {m.epoch:3d} "
            f"| {m.loss_mean:.6f} "
            f"| {m.loss_std:.6f} "
            f"| {m.loss_min:.6f} "
            f"| {m.loss_max:.6f} "
            f"| {m.lr:.2e} "
            f"| {m.grad_norm:.4f} "
            f"| {m.samples_per_sec:.0f} "
            f"| {m.gpu_mem_peak_mb:.0f} "
            f"| {m.ema_delta_norm:.4f} "
            f"| {m.epoch_time_sec:.1f} |"
        )
    _h("")

    # ---- Plots ------------------------------------------------------------
    if plot_paths:
        _h("## Training Curves")
        _h("")
        for p in plot_paths:
            name = os.path.splitext(os.path.basename(p))[0].replace("_", " ").title()
            rel = os.path.relpath(p, save_dir)
            _h(f"### {name}")
            _h(f"![{name}]({rel})")
            _h("")

    # ---- Files manifest ---------------------------------------------------
    _h("## Saved Artifacts")
    _h("")
    _h(f"- **CSV metrics:** `{os.path.basename(tracker.csv_path)}`")
    _h(f"- **JSON metrics:** `{os.path.basename(tracker.json_path)}`")
    _h(f"- **Report:** `{os.path.basename(report_path)}`")
    if plot_paths:
        _h(f"- **Plots:** {len(plot_paths)} images in `plots/`")
    _h("")

    # ---- Footer -----------------------------------------------------------
    _h("---")
    _h("*APEX Laboratory — The University of Chicago*")
    _h("")

    content = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(content)

    return report_path
