"""
Beta Export
===========

Exports model betas to CSV in MATLAB-compatible column-vector format.

Design Principles:
    - Mean betas (averaged across voxels) as single-column CSV
    - Excludes intercept row (``b(2:end,:)`` in MATLAB → ``betas[1:, :]``)
    - Filename: ``meanbeta_sub-{s}_{roi}_fc7_invert_imageFeatures.csv``
    - Also exports full voxelwise betas for advanced downstream analysis

MATLAB Correspondence:
    - make_random_subregions_betas_to_csv.m → ``export_mean_betas_csv()``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from amod_encoder.utils.logging import get_logger, log_matlab_note

logger = get_logger(__name__)


def export_mean_betas_csv(
    betas: np.ndarray,
    subject_id: str,
    roi_name: str,
    output_dir: Path,
    feature_name: str = "fc7",
) -> Path:
    """Export mean betas (averaged across voxels) as a single-column CSV.

    Parameters
    ----------
    betas : np.ndarray, shape (D+1, V)
        Full beta matrix with intercept in first row.
    subject_id : str
        Subject ID.
    roi_name : str
        ROI name.
    output_dir : Path
        Output directory.
    feature_name : str
        Feature name for filename.

    Returns
    -------
    Path
        Path to saved CSV file.

    Notes
    -----
    MATLAB:
        csvwrite([...], mean(b(2:end, :)')');
    This computes mean across voxels (columns) for each feature (row),
    excluding the intercept, then writes as a column vector.
    """
    log_matlab_note(
        logger,
        "make_random_subregions_betas_to_csv.m",
        f"Exporting mean betas for sub-{subject_id} / {roi_name}",
    )

    # Remove intercept (first row), then average across voxels
    # MATLAB: mean(b(2:end,:)') → mean across voxels (dim 2) → (D,)
    coef = betas[1:, :]  # (D, V) — feature coefficients, no intercept
    mean_betas = coef.mean(axis=1)  # (D,) — mean across voxels

    # Save as single-column CSV (matching MATLAB csvwrite output)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    filename = f"meanbeta_sub-{subject_id}_{roi_name}_{feature_name}_invert_imageFeatures.csv"
    csv_path = tables_dir / filename

    np.savetxt(str(csv_path), mean_betas, delimiter=",", fmt="%.10f")

    logger.info("Exported mean betas: %s (shape=%s)", csv_path.name, mean_betas.shape)
    return csv_path


def export_full_betas_csv(
    betas: np.ndarray,
    subject_id: str,
    roi_name: str,
    output_dir: Path,
    feature_name: str = "fc7",
) -> Path:
    """Export full beta matrix as CSV.

    Parameters
    ----------
    betas : np.ndarray, shape (D+1, V)
        Full beta matrix.
    subject_id : str
        Subject ID.
    roi_name : str
        ROI name.
    output_dir : Path
        Output directory.
    feature_name : str
        Feature name.

    Returns
    -------
    Path
        Path to saved CSV.
    """
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    filename = f"beta_sub-{subject_id}_{roi_name}_{feature_name}_invert_imageFeatures.csv"
    csv_path = tables_dir / filename

    np.savetxt(str(csv_path), betas, delimiter=",", fmt="%.10f")

    logger.info("Exported full betas: %s (shape=%s)", csv_path.name, betas.shape)
    return csv_path


def export_random_subregion_betas(
    betas: np.ndarray,
    subject_id: str,
    n_subregions: int = 4,
    iteration: int = 1,
    output_dir: Path = Path("output"),
    seed: int = 42,
) -> list[Path]:
    """Export betas for random subregion splits (matching MATLAB random split analysis).

    Parameters
    ----------
    betas : np.ndarray, shape (D+1, V)
        Full beta matrix.
    subject_id : str
        Subject ID.
    n_subregions : int
        Number of random subregions.
    iteration : int
        Iteration number (for random split analysis).
    output_dir : Path
        Output directory.
    seed : int
        Random seed for reproducible split.

    Returns
    -------
    list[Path]
        Paths to saved CSV files.

    Notes
    -----
    MATLAB:
        split = randi(4, size(b,2), 1);
        for r = 1:4
            csvwrite([...], mean(b(2:end, split==r)')');
        end
    """
    log_matlab_note(
        logger,
        "make_random_subregions_betas_to_csv.m",
        f"Random subregion split: {n_subregions} regions, iteration {iteration}",
    )

    rng = np.random.RandomState(seed + iteration)
    n_voxels = betas.shape[1]
    split = rng.randint(1, n_subregions + 1, size=n_voxels)

    tables_dir = output_dir / "tables" / "random_subregion_betas"
    tables_dir.mkdir(parents=True, exist_ok=True)

    coef = betas[1:, :]  # remove intercept
    paths = []

    for r in range(1, n_subregions + 1):
        voxel_mask = split == r
        if voxel_mask.sum() == 0:
            logger.warning("Region %d has 0 voxels in iteration %d", r, iteration)
            continue

        mean_b = coef[:, voxel_mask].mean(axis=1)
        filename = (
            f"meanbeta_region{r}_sub-{subject_id}"
            f"_amyFearful_fc7_invert_imageFeatures_iteration_{iteration}.csv"
        )
        csv_path = tables_dir / filename
        np.savetxt(str(csv_path), mean_b, delimiter=",", fmt="%.10f")
        paths.append(csv_path)

    logger.info("Exported %d random subregion beta files", len(paths))
    return paths
