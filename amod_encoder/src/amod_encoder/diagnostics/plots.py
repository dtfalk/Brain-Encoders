"""
Paper-Matched Plotting System for Brain-Encoders
=================================================

Produces publication-quality figures that mirror every figure from:

    Jang & Kragel (2024). "Understanding human amygdala function
    with artificial neural networks."

Each plot function corresponds to a specific paper figure and can be
used both for exact replication and for extension experiments with
different ROIs, feature extractors, or datasets.

Design Principles (follows node-reference-code patterns):
    - Dark theme consistent with node-reference-code style
    - One function per plot, each returns the saved path
    - ``generate_all_validation_plots()`` aggregator with error handling
    - ``matplotlib.use("Agg")`` for Slurm compatibility
    - Every plot is ROI-agnostic: pass in data + labels, not hardcoded names

Figure Mapping:
    +------+-----------------------------------------+----------------------------+
    | Fig  | Paper Content                           | Function                   |
    +------+-----------------------------------------+----------------------------+
    |  3a  | Group t-stat maps (voxelwise)            | plot_tstat_map()           |
    |  3b  | Amygdala parcellation rendering           | (needs brain surface)      |
    |  3c  | Violin: predictive perf per subregion     | plot_subregion_violins()   |
    |  4   | Valence × arousal surface plots           | plot_valence_arousal()     |
    |  6a  | t-SNE of predicted activations            | plot_tsne()                |
    |  6b  | Optimal clustering solution               | plot_clustering()          |
    |  6c  | Normalized confusion matrix               | plot_confusion_matrix()    |
    +------+-----------------------------------------+----------------------------+
    | T1   | Valence/arousal effects table             | table_regression_effects() |
    +------+-----------------------------------------+----------------------------+

Author: Brain-Encoders Project
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Slurm compatibility

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from amod_encoder.utils.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# Dark Theme — matches node-reference-code style
# =============================================================================

STYLE: Dict[str, Any] = {
    "figure.facecolor":  "#0e1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "text.color":        "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "grid.color":        "#21262d",
    "grid.alpha":        0.6,
    "lines.linewidth":   1.8,
    "font.family":       "monospace",
    "savefig.dpi":       200,
    "savefig.facecolor": "#0e1117",
    "savefig.bbox":      "tight",
}

# Named accent colours
ACCENT   = "#58a6ff"   # Blue
ACCENT2  = "#f78166"   # Orange-red
ACCENT3  = "#7ee787"   # Green
ACCENT4  = "#d2a8ff"   # Purple
ACCENT5  = "#ffa657"   # Orange
ACCENT6  = "#ff7b72"   # Red
FILL_ALPHA = 0.15

# Default ROI colour map (ROI-agnostic — pass your own if you want)
DEFAULT_ROI_COLORS = {
    "LB":   "#58a6ff",
    "SF":   "#ffa657",
    "CM":   "#f78166",
    "AStr": "#7ee787",
    "amygdala":      "#d2a8ff",
    "visual_cortex": "#8b949e",
    "IT":            "#ff7b72",
}


def _apply_style() -> None:
    """Apply the dark theme to matplotlib."""
    plt.rcParams.update(STYLE)


def _get_roi_color(roi_name: str) -> str:
    """Get colour for an ROI, falling back to accent blue."""
    return DEFAULT_ROI_COLORS.get(roi_name, ACCENT)


def _ensure_dir(save_dir: str | Path) -> Path:
    """Create save directory if needed, return Path."""
    d = Path(save_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# Figure 3c — Violin: Predictive Performance per Subregion
# =============================================================================

def plot_subregion_violins(
    data: Dict[str, np.ndarray],
    save_dir: str | Path,
    title: str = "Encoding Model Performance by Region",
    ylabel: str = "Mean Voxelwise Correlation (Fisher's Z)",
    filename: str = "fig3c_subregion_violins.png",
) -> str:
    """
    Violin plot of per-subject predictive performance for each ROI.

    Mirrors **Figure 3c** from Jang & Kragel (2024): violin + individual
    dots + SEM error bars, one violin per subregion.

    This function is ROI-agnostic — pass any dict of region_name → values.

    Args:
        data:     Dict mapping ROI name → 1D array of per-subject values
                  (e.g. mean Fisher's Z across voxels, one per subject).
        save_dir: Output directory for the plot.
        title:    Plot title.
        ylabel:   Y-axis label.
        filename: Output filename.

    Returns:
        Absolute path to the saved figure.
    """
    _apply_style()
    save_dir = _ensure_dir(save_dir)

    roi_names = list(data.keys())
    roi_values = [data[r] for r in roi_names]
    n_rois = len(roi_names)
    colors = [_get_roi_color(r) for r in roi_names]

    fig, ax = plt.subplots(figsize=(max(6, n_rois * 1.5), 5))

    # Violin plot
    parts = ax.violinplot(roi_values, positions=range(n_rois), showmedians=False,
                          showextrema=False, widths=0.7)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.3)
        pc.set_edgecolor(colors[i])

    # Individual data points (jittered)
    rng = np.random.default_rng(42)
    for i, vals in enumerate(roi_values):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=colors[i], s=20, alpha=0.7, zorder=3, edgecolors="none")

    # Mean + SEM bar
    for i, vals in enumerate(roi_values):
        mean = np.nanmean(vals)
        sem = np.nanstd(vals, ddof=1) / np.sqrt(len(vals))
        ax.errorbar(i, mean, yerr=sem, fmt="D", color="white", markersize=6,
                    capsize=4, capthick=1.5, elinewidth=1.5, zorder=4)

    ax.set_xticks(range(n_rois))
    ax.set_xticklabels(roi_names, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(axis="y", alpha=0.3)

    path = str(save_dir / filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info("plot_saved | path=%s", path)
    return path


# =============================================================================
# Figure 4 — Valence × Arousal Surface Plots
# =============================================================================

def plot_valence_arousal(
    valence: np.ndarray,
    arousal: np.ndarray,
    predicted_activation: np.ndarray,
    roi_name: str,
    save_dir: str | Path,
    filename: Optional[str] = None,
    n_bins: int = 10,
) -> str:
    """
    Surface plot of predicted activation as a function of valence and arousal.

    Mirrors **Figure 4** from Jang & Kragel (2024): 3D surface with
    valence on x-axis, arousal on y-axis, predicted activation on z.

    Args:
        valence:              Valence ratings, shape (N,).
        arousal:              Arousal ratings, shape (N,).
        predicted_activation: Predicted responses, shape (N,).
        roi_name:             ROI name for title.
        save_dir:             Output directory.
        filename:             Output filename (auto-generated if None).
        n_bins:               Number of bins per axis for smoothing.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    save_dir = _ensure_dir(save_dir)
    if filename is None:
        filename = f"fig4_valence_arousal_{roi_name}.png"

    # Bin data for surface smoothing
    val_bins = np.linspace(np.nanmin(valence), np.nanmax(valence), n_bins + 1)
    aro_bins = np.linspace(np.nanmin(arousal), np.nanmax(arousal), n_bins + 1)
    surface = np.full((n_bins, n_bins), np.nan)

    for i in range(n_bins):
        for j in range(n_bins):
            mask = (
                (valence >= val_bins[i]) & (valence < val_bins[i + 1]) &
                (arousal >= aro_bins[j]) & (arousal < aro_bins[j + 1])
            )
            if mask.sum() > 0:
                surface[j, i] = np.nanmean(predicted_activation[mask])

    # Create meshgrid for surface
    val_centers = (val_bins[:-1] + val_bins[1:]) / 2
    aro_centers = (aro_bins[:-1] + aro_bins[1:]) / 2
    V, A = np.meshgrid(val_centers, aro_centers)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#161b22")

    # Interpolate NaN for smoother surface
    from scipy.interpolate import griddata
    valid = ~np.isnan(surface)
    if valid.sum() >= 4:
        points = np.column_stack([V[valid], A[valid]])
        values = surface[valid]
        surface_interp = griddata(points, values, (V, A), method="cubic")
        ax.plot_surface(V, A, surface_interp, cmap="coolwarm", alpha=0.8,
                        edgecolor="none", antialiased=True)
    else:
        ax.plot_surface(V, A, np.nan_to_num(surface), cmap="coolwarm", alpha=0.8)

    ax.set_xlabel("Valence", fontsize=10, labelpad=8)
    ax.set_ylabel("Arousal", fontsize=10, labelpad=8)
    ax.set_zlabel("Predicted Activation", fontsize=10, labelpad=8)
    ax.set_title(f"Valence × Arousal — {roi_name}", fontsize=12, pad=15)

    path = str(save_dir / filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info("plot_saved | path=%s", path)
    return path


# =============================================================================
# Figure 6a — t-SNE Visualization
# =============================================================================

def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray | list[str],
    save_dir: str | Path,
    title: str = "t-SNE of Predicted Activations",
    filename: str = "fig6a_tsne.png",
    perplexity: float = 30.0,
    label_colors: Optional[Dict[str, str]] = None,
) -> str:
    """
    t-SNE visualization of predicted activations, coloured by target ROI.

    Mirrors **Figure 6a** from Jang & Kragel (2024).

    Args:
        features:     Feature matrix, shape (N, D).
        labels:       Category label for each sample (ROI target name).
        save_dir:     Output directory.
        title:        Plot title.
        filename:     Output filename.
        perplexity:   t-SNE perplexity parameter.
        label_colors: Optional mapping of label → hex colour.

    Returns:
        Path to saved figure.
    """
    from sklearn.manifold import TSNE

    _apply_style()
    save_dir = _ensure_dir(save_dir)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                init="pca", learning_rate="auto")
    coords = tsne.fit_transform(features)

    unique_labels = sorted(set(labels))
    if label_colors is None:
        label_colors = {l: _get_roi_color(l) for l in unique_labels}

    fig, ax = plt.subplots(figsize=(8, 7))

    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=label_colors.get(label, ACCENT), label=label,
                   s=25, alpha=0.7, edgecolors="none")

    ax.legend(fontsize=9, loc="best", framealpha=0.3,
              edgecolor="#30363d", facecolor="#161b22")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)

    path = str(save_dir / filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info("plot_saved | path=%s", path)
    return path


# =============================================================================
# Figure 6c — Normalized Confusion Matrix
# =============================================================================

def plot_confusion_matrix(
    true_labels: np.ndarray | list,
    pred_labels: np.ndarray | list,
    class_names: list[str],
    save_dir: str | Path,
    title: str = "Classification Confusion Matrix",
    filename: str = "fig6c_confusion_matrix.png",
    normalize: bool = True,
) -> str:
    """
    Normalized confusion matrix for multi-class classification.

    Mirrors **Figure 6c** from Jang & Kragel (2024): 7-way classification
    matrix (amy, IT, VC, AStr, CM, LB, SF).

    Args:
        true_labels:  Ground-truth labels.
        pred_labels:  Predicted labels.
        class_names:  Ordered list of class names.
        save_dir:     Output directory.
        title:        Plot title.
        filename:     Output filename.
        normalize:    If True, normalize rows to proportions.

    Returns:
        Path to saved figure.
    """
    from sklearn.metrics import confusion_matrix

    _apply_style()
    save_dir = _ensure_dir(save_dir)

    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype(float) / row_sums
    else:
        cm_norm = cm.astype(float)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.9),
                                    max(5, len(class_names) * 0.8)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1 if normalize else None)

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "#c9d1d9"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=10, rotation=45, ha="right")
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    path = str(save_dir / filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info("plot_saved | path=%s", path)
    return path


# =============================================================================
# Regression Effects — Valence/Arousal/Interaction Barplot
# =============================================================================

def plot_regression_effects(
    effects: Dict[str, Dict[str, float]],
    save_dir: str | Path,
    title: str = "Regression Effects: Valence, Arousal, Interaction",
    filename: str = "regression_effects.png",
) -> str:
    """
    Grouped bar chart of regression coefficients (beta, t, p, d).

    Mirrors **Table 1** from Jang & Kragel (2024), visualised as bars
    with significance markers.

    Args:
        effects: Nested dict {roi_name: {predictor: beta_value, ...}}.
                 Predictors should include 'valence', 'arousal', 'interaction'.
        save_dir:  Output directory.
        title:     Plot title.
        filename:  Output filename.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    save_dir = _ensure_dir(save_dir)

    roi_names = list(effects.keys())
    predictors = list(next(iter(effects.values())).keys())
    n_rois = len(roi_names)
    n_pred = len(predictors)

    x = np.arange(n_rois)
    width = 0.8 / n_pred
    pred_colors = [ACCENT, ACCENT2, ACCENT4, ACCENT3, ACCENT5, ACCENT6]

    fig, ax = plt.subplots(figsize=(max(8, n_rois * 2), 5))

    for i, pred in enumerate(predictors):
        vals = [effects[r].get(pred, 0.0) for r in roi_names]
        offset = (i - n_pred / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=pred,
               color=pred_colors[i % len(pred_colors)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, fontsize=11)
    ax.set_ylabel("β coefficient", fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=9, framealpha=0.3, edgecolor="#30363d", facecolor="#161b22")
    ax.grid(axis="y", alpha=0.3)

    path = str(save_dir / filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info("plot_saved | path=%s", path)
    return path


# =============================================================================
# Piecewise Regression — Negative / Neutral / Positive
# =============================================================================

def plot_piecewise_regression(
    valence_ranges: Dict[str, Dict[str, float]],
    save_dir: str | Path,
    roi_name: str = "amygdala",
    filename: Optional[str] = None,
) -> str:
    """
    Bar chart of piecewise regression betas for negative/neutral/positive valence.

    Args:
        valence_ranges: Dict of {"negative": {"beta": ..., "se": ..., "p": ...}, ...}.
        save_dir:       Output directory.
        roi_name:       ROI name for title.
        filename:       Output filename.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    save_dir = _ensure_dir(save_dir)
    if filename is None:
        filename = f"piecewise_regression_{roi_name}.png"

    categories = list(valence_ranges.keys())
    betas = [valence_ranges[c].get("beta", 0.0) for c in categories]
    ses = [valence_ranges[c].get("se", 0.0) for c in categories]
    pvals = [valence_ranges[c].get("p", 1.0) for c in categories]
    colors = [ACCENT6, "#8b949e", ACCENT3]  # red, gray, green

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(categories, betas, yerr=ses, capsize=5,
                  color=colors[:len(categories)], alpha=0.8,
                  edgecolor="#30363d", linewidth=0.8)

    # Significance markers
    for i, p in enumerate(pvals):
        if p < 0.001:
            marker = "***"
        elif p < 0.01:
            marker = "**"
        elif p < 0.05:
            marker = "*"
        else:
            marker = "n.s."
        y = betas[i] + ses[i] + 0.001
        ax.text(i, y, marker, ha="center", fontsize=10, color="#c9d1d9")

    ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.set_ylabel("β coefficient", fontsize=12)
    ax.set_title(f"Piecewise Valence Regression — {roi_name}", fontsize=12, pad=10)
    ax.grid(axis="y", alpha=0.3)

    path = str(save_dir / filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info("plot_saved | path=%s", path)
    return path


# =============================================================================
# On-Target Selectivity — Artificial Stimuli
# =============================================================================

def plot_on_target_selectivity(
    selectivity: Dict[str, Dict[str, float]],
    save_dir: str | Path,
    filename: str = "on_target_selectivity.png",
) -> str:
    """
    Bar chart of on-target vs off-target selectivity per ROI.

    Shows beta (on-target - off-target), SE, and p-values.

    Args:
        selectivity: {roi: {"beta": ..., "se": ..., "t": ..., "p": ..., "d": ...}}.
        save_dir:    Output directory.
        filename:    Output filename.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    save_dir = _ensure_dir(save_dir)

    rois = list(selectivity.keys())
    betas = [selectivity[r]["beta"] for r in rois]
    ses = [selectivity[r].get("se", 0.0) for r in rois]
    pvals = [selectivity[r].get("p", 1.0) for r in rois]
    colors = [_get_roi_color(r) for r in rois]

    fig, ax = plt.subplots(figsize=(max(6, len(rois) * 1.5), 5))
    bars = ax.bar(rois, betas, yerr=ses, capsize=5,
                  color=colors, alpha=0.8, edgecolor="#30363d", linewidth=0.8)

    for i, p in enumerate(pvals):
        if p < 0.001:
            marker = "***"
        elif p < 0.01:
            marker = "**"
        elif p < 0.05:
            marker = "*"
        else:
            marker = "n.s."
        y = betas[i] + ses[i] + 0.002
        ax.text(i, y, marker, ha="center", fontsize=10, color="#c9d1d9")

    ax.axhline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    ax.set_ylabel("On-target β", fontsize=12)
    ax.set_title("On-Target Selectivity of Artificial Stimuli", fontsize=12, pad=10)
    ax.grid(axis="y", alpha=0.3)

    path = str(save_dir / filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info("plot_saved | path=%s", path)
    return path


# =============================================================================
# Dashboard — Multi-panel summary (like node-reference-code)
# =============================================================================

def plot_validation_dashboard(
    subregion_data: Dict[str, np.ndarray],
    regression_effects: Optional[Dict[str, Dict[str, float]]] = None,
    piecewise_data: Optional[Dict[str, Dict[str, float]]] = None,
    save_dir: str | Path = "output/plots",
    filename: str = "validation_dashboard.png",
) -> str:
    """
    Multi-panel validation dashboard combining key results.

    Layout (GridSpec 2×2):
        [0,0] Subregion violin plot
        [0,1] Regression effects bar chart
        [1,0] Piecewise regression
        [1,1] Summary statistics text panel

    Args:
        subregion_data:     {roi: array of per-subject values}.
        regression_effects: {roi: {predictor: beta}}.
        piecewise_data:     {valence_range: {beta, se, p}}.
        save_dir:           Output directory.
        filename:           Output filename.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    save_dir = _ensure_dir(save_dir)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # --- Panel 0,0: Subregion violins ---
    ax0 = fig.add_subplot(gs[0, 0])
    roi_names = list(subregion_data.keys())
    roi_values = [subregion_data[r] for r in roi_names]
    colors = [_get_roi_color(r) for r in roi_names]

    parts = ax0.violinplot(roi_values, positions=range(len(roi_names)),
                           showmedians=False, showextrema=False, widths=0.7)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.3)
        pc.set_edgecolor(colors[i])

    rng = np.random.default_rng(42)
    for i, vals in enumerate(roi_values):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax0.scatter(np.full(len(vals), i) + jitter, vals,
                    color=colors[i], s=15, alpha=0.6, edgecolors="none")
        mean = np.nanmean(vals)
        sem = np.nanstd(vals, ddof=1) / np.sqrt(len(vals))
        ax0.errorbar(i, mean, yerr=sem, fmt="D", color="white", markersize=5,
                     capsize=3, capthick=1.2, elinewidth=1.2, zorder=4)

    ax0.set_xticks(range(len(roi_names)))
    ax0.set_xticklabels(roi_names, fontsize=9)
    ax0.set_ylabel("Fisher's Z", fontsize=10)
    ax0.set_title("Predictive Performance by Region", fontsize=11)
    ax0.grid(axis="y", alpha=0.3)

    # --- Panel 0,1: Regression effects ---
    ax1 = fig.add_subplot(gs[0, 1])
    if regression_effects:
        re_rois = list(regression_effects.keys())
        predictors = list(next(iter(regression_effects.values())).keys())
        x = np.arange(len(re_rois))
        width = 0.8 / len(predictors)
        pred_cols = [ACCENT, ACCENT2, ACCENT4, ACCENT3, ACCENT5]
        for i, pred in enumerate(predictors):
            vals = [regression_effects[r].get(pred, 0) for r in re_rois]
            offset = (i - len(predictors) / 2 + 0.5) * width
            ax1.bar(x + offset, vals, width, label=pred,
                    color=pred_cols[i % len(pred_cols)], alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(re_rois, fontsize=9)
        ax1.legend(fontsize=7, framealpha=0.3, edgecolor="#30363d", facecolor="#161b22")
        ax1.axhline(0, color="#8b949e", linewidth=0.6, linestyle="--")
    ax1.set_ylabel("β", fontsize=10)
    ax1.set_title("Regression Effects (Valence / Arousal / Interaction)", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # --- Panel 1,0: Piecewise regression ---
    ax2 = fig.add_subplot(gs[1, 0])
    if piecewise_data:
        cats = list(piecewise_data.keys())
        betas = [piecewise_data[c].get("beta", 0) for c in cats]
        ses = [piecewise_data[c].get("se", 0) for c in cats]
        pw_colors = [ACCENT6, "#8b949e", ACCENT3]
        ax2.bar(cats, betas, yerr=ses, capsize=4,
                color=pw_colors[:len(cats)], alpha=0.8, edgecolor="#30363d")
        ax2.axhline(0, color="#8b949e", linewidth=0.6, linestyle="--")
    ax2.set_ylabel("β", fontsize=10)
    ax2.set_title("Piecewise Valence Regression", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    # --- Panel 1,1: Summary stats ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    summary_lines = ["Summary Statistics", "=" * 30, ""]
    for name, vals in subregion_data.items():
        m = np.nanmean(vals)
        s = np.nanstd(vals, ddof=1) / np.sqrt(len(vals))
        summary_lines.append(f"{name:<12}  mean={m:+.4f}  SEM={s:.4f}  N={len(vals)}")
    summary_lines.append("")
    summary_lines.append(f"Regions: {len(roi_names)}")
    ax3.text(0.05, 0.95, "\n".join(summary_lines), transform=ax3.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             color="#c9d1d9")

    fig.suptitle("Brain-Encoders — Validation Dashboard", fontsize=14,
                 color="#c9d1d9", y=0.98)

    path = str(save_dir / filename)
    fig.savefig(path)
    plt.close(fig)
    logger.info("dashboard_saved | path=%s", path)
    return path


# =============================================================================
# Aggregator — generate_all_validation_plots()
# =============================================================================

def generate_all_validation_plots(
    subregion_data: Dict[str, np.ndarray],
    save_dir: str | Path,
    regression_effects: Optional[Dict[str, Dict[str, float]]] = None,
    piecewise_data: Optional[Dict[str, Dict[str, float]]] = None,
    selectivity_data: Optional[Dict[str, Dict[str, float]]] = None,
    tsne_features: Optional[np.ndarray] = None,
    tsne_labels: Optional[list] = None,
    confusion_true: Optional[list] = None,
    confusion_pred: Optional[list] = None,
    confusion_classes: Optional[list[str]] = None,
    valence_arousal_data: Optional[list[Dict]] = None,
) -> List[str]:
    """
    Generate all available validation plots in one call.

    Follows the node-reference-code pattern: each generator is called
    in a try/except block so one failure doesn't block the rest.

    Args:
        subregion_data:     Required. {roi: array}.
        save_dir:           Required. Output directory.
        regression_effects: Optional. For regression bar chart.
        piecewise_data:     Optional. For piecewise regression plot.
        selectivity_data:   Optional. For on-target selectivity.
        tsne_features:      Optional. For t-SNE plot.
        tsne_labels:        Optional. Category labels for t-SNE.
        confusion_true:     Optional. Ground truth for confusion matrix.
        confusion_pred:     Optional. Predictions for confusion matrix.
        confusion_classes:  Optional. Class names.
        valence_arousal_data: Optional. List of dicts for Fig 4 surface plots.

    Returns:
        List of paths to all generated plots.
    """
    paths: List[str] = []

    generators = [
        ("subregion_violins",
         lambda: plot_subregion_violins(subregion_data, save_dir)),
        ("validation_dashboard",
         lambda: plot_validation_dashboard(
             subregion_data, regression_effects, piecewise_data, save_dir)),
    ]

    if regression_effects:
        generators.append(
            ("regression_effects",
             lambda: plot_regression_effects(regression_effects, save_dir)))

    if piecewise_data:
        generators.append(
            ("piecewise_regression",
             lambda: plot_piecewise_regression(piecewise_data, save_dir)))

    if selectivity_data:
        generators.append(
            ("on_target_selectivity",
             lambda: plot_on_target_selectivity(selectivity_data, save_dir)))

    if tsne_features is not None and tsne_labels is not None:
        generators.append(
            ("tsne",
             lambda: plot_tsne(tsne_features, tsne_labels, save_dir)))

    if confusion_true is not None and confusion_pred is not None and confusion_classes:
        generators.append(
            ("confusion_matrix",
             lambda: plot_confusion_matrix(
                 confusion_true, confusion_pred, confusion_classes, save_dir)))

    if valence_arousal_data:
        for i, va in enumerate(valence_arousal_data):
            roi = va.get("roi_name", f"roi_{i}")
            generators.append(
                (f"valence_arousal_{roi}",
                 lambda v=va: plot_valence_arousal(
                     v["valence"], v["arousal"], v["activation"], v["roi_name"],
                     save_dir)))

    for name, gen in generators:
        try:
            p = gen()
            if p:
                paths.append(p)
                logger.info("plot_generated | name=%s", name)
        except Exception as e:
            logger.warning("plot_failed | name=%s error=%s", name, e)

    logger.info("all_plots_done | total=%d", len(paths))
    return paths
