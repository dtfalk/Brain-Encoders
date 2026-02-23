"""
Subregion Atanh Masking
======================

Extracts subregion-level Fisher’s Z from whole-amygdala voxelwise matrices.

Core Algorithm::

    1. Load atanh_matrix of shape (S, V_amy)  [subjects × amygdala voxels]
    2. For each subregion mask, determine which voxels belong to it
    3. Replace zeros with NaN (MATLAB behaviour)
    4. Average across subregion voxels → avg_atanh of shape (S, R)

Design Principles:
    - Input is ROI-agnostic: any parent mask + child subregion masks
    - Voxel correspondence determined via ``nibabel`` + ``nilearn`` resampling
    - Exposed as both a Python function and a ``compile-atanh`` CLI command

MATLAB Correspondence:
    - make_atanh_matrix_subregion.m → ``mask_atanh_by_subregions()``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

from amod_encoder.utils.logging import get_logger, log_matlab_note

logger = get_logger(__name__)


@dataclass
class SubregionAtanhResult:
    """Container for subregion atanh masking results.

    Attributes
    ----------
    avg_atanh : np.ndarray
        Array of shape (S, R) — mean atanh per subject per subregion.
    subregion_names : list[str]
        Names of each subregion column.
    subregion_voxel_counts : dict[str, int]
        Number of voxels within each subregion inside the parent mask.
    parent_n_voxels : int
        Total number of voxels in the parent (amygdala) mask.
    """

    avg_atanh: np.ndarray
    subregion_names: list[str]
    subregion_voxel_counts: dict[str, int]
    parent_n_voxels: int


def _load_binary_mask_data(mask_path: Path) -> nib.Nifti1Image:
    """Load and validate a binary NIfTI mask."""
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    return nib.load(str(mask_path))


def compute_subregion_membership(
    parent_mask_path: Path,
    subregion_mask_paths: dict[str, Path],
    reference_img: nib.Nifti1Image | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Determine which parent-mask voxels belong to each subregion.

    This mirrors the MATLAB logic:
        amy_bin = double(masked_dat.dat(:,1) ~= 0);
        for r = 1:length(rois)
            masked_dat = apply_mask(dat, select_atlas_subset(load_atlas('canlab2018'), rois{r}));
            rois_bin(r,:) = masked_dat.dat(amy_bin==1, 1) ~= 0;
        end

    Parameters
    ----------
    parent_mask_path : Path
        Path to the parent ROI mask (e.g., whole amygdala).
    subregion_mask_paths : dict[str, Path]
        Mapping of subregion name → mask path (e.g., {"CM": Path(...), "LB": Path(...)}).
    reference_img : nib.Nifti1Image | None
        Optional reference image for resampling. If None, the parent mask is used.

    Returns
    -------
    parent_indices : np.ndarray
        Linear indices of parent mask voxels in the 3D volume.
    subregion_membership : dict[str, np.ndarray]
        Mapping of subregion name → boolean array of length len(parent_indices).
        True where a parent voxel falls within that subregion.
    """
    parent_img = _load_binary_mask_data(parent_mask_path)
    ref = reference_img if reference_img is not None else parent_img

    # Resample parent mask to reference space
    parent_resampled = resample_to_img(parent_img, ref, interpolation="nearest")
    parent_data = np.asarray(parent_resampled.dataobj).astype(bool)
    if parent_data.ndim == 4:
        parent_data = parent_data[..., 0]

    parent_indices = np.where(parent_data.ravel())[0]
    n_parent = len(parent_indices)
    logger.info("Parent mask: %d voxels", n_parent)

    subregion_membership: dict[str, np.ndarray] = {}
    for name, sub_path in subregion_mask_paths.items():
        sub_img = _load_binary_mask_data(sub_path)
        sub_resampled = resample_to_img(sub_img, ref, interpolation="nearest")
        sub_data = np.asarray(sub_resampled.dataobj).astype(bool)
        if sub_data.ndim == 4:
            sub_data = sub_data[..., 0]

        sub_flat = sub_data.ravel()
        # Boolean membership within the parent mask
        membership = sub_flat[parent_indices]
        subregion_membership[name] = membership
        logger.info(
            "Subregion '%s': %d / %d parent voxels",
            name,
            int(membership.sum()),
            n_parent,
        )

    return parent_indices, subregion_membership


def mask_atanh_by_subregions(
    atanh_matrix: np.ndarray,
    subregion_membership: dict[str, np.ndarray],
    replace_zero_with_nan: bool = True,
) -> SubregionAtanhResult:
    """Mask a whole-ROI atanh matrix by subregion membership and average.

    This reproduces the core MATLAB logic:
        atanh_matrix(atanh_matrix==0) = NaN;
        for r = 1:length(rois)
            atanh_subregion = atanh_matrix(:, rois_bin(r,:)==1);
            avg_atanh_subregion(:,r) = mean(atanh_subregion, 2);
        end

    Parameters
    ----------
    atanh_matrix : np.ndarray, shape (S, V)
        Fisher's Z transformed correlation matrix.
        S = subjects (or repetitions), V = voxels in parent mask.
    subregion_membership : dict[str, np.ndarray]
        Boolean arrays from ``compute_subregion_membership``.
    replace_zero_with_nan : bool
        If True, replace exact zeros with NaN before averaging (MATLAB behaviour).

    Returns
    -------
    SubregionAtanhResult
        Result container with averaged atanh values and metadata.
    """
    S, V = atanh_matrix.shape

    log_matlab_note(
        logger,
        "make_atanh_matrix_subregion.m",
        f"Masking atanh matrix ({S} × {V}) by {len(subregion_membership)} subregions",
    )

    # Replace zeros with NaN — MATLAB: atanh_matrix(atanh_matrix==0) = NaN
    if replace_zero_with_nan:
        work = atanh_matrix.copy().astype(np.float64)
        n_zeros = int(np.sum(work == 0))
        if n_zeros > 0:
            logger.info("Replacing %d zero entries with NaN", n_zeros)
            work[work == 0] = np.nan
    else:
        work = atanh_matrix.astype(np.float64)

    subregion_names = list(subregion_membership.keys())
    n_regions = len(subregion_names)
    avg_atanh = np.full((S, n_regions), np.nan, dtype=np.float64)
    voxel_counts: dict[str, int] = {}

    for r_idx, name in enumerate(subregion_names):
        mask = subregion_membership[name]
        if len(mask) != V:
            raise ValueError(
                f"Subregion '{name}' membership length ({len(mask)}) "
                f"doesn't match atanh_matrix columns ({V})"
            )

        n_sub_voxels = int(mask.sum())
        voxel_counts[name] = n_sub_voxels

        if n_sub_voxels == 0:
            logger.warning("Subregion '%s': 0 voxels — avg_atanh will be NaN", name)
            continue

        # Extract subregion columns and average (nanmean to handle NaN)
        sub_atanh = work[:, mask]  # (S, n_sub_voxels)
        avg_atanh[:, r_idx] = np.nanmean(sub_atanh, axis=1)

        logger.info(
            "Subregion '%s': %d voxels, mean atanh = %.4f",
            name,
            n_sub_voxels,
            float(np.nanmean(avg_atanh[:, r_idx])),
        )

    return SubregionAtanhResult(
        avg_atanh=avg_atanh,
        subregion_names=subregion_names,
        subregion_voxel_counts=voxel_counts,
        parent_n_voxels=V,
    )


def save_subregion_atanh(
    result: SubregionAtanhResult,
    output_dir: Path,
    prefix: str = "amygdala_subregions_atanh",
) -> Path:
    """Save subregion-averaged atanh matrix to disk.

    Saves both .npy (for Python) and .csv (for R/MATLAB interop).
    MATLAB equivalent: save(['amygdala_subregions_atanh.mat'], 'avg_atanh_subregion')

    Parameters
    ----------
    result : SubregionAtanhResult
        Output from ``mask_atanh_by_subregions``.
    output_dir : Path
        Directory to save files into.
    prefix : str
        Filename prefix.

    Returns
    -------
    Path
        Path to the saved .npy file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_path = output_dir / f"{prefix}.npy"
    np.save(npy_path, result.avg_atanh)
    logger.info("Saved atanh matrix: %s", npy_path)

    # CSV for interoperability
    try:
        import pandas as pd

        csv_path = output_dir / f"{prefix}.csv"
        df = pd.DataFrame(result.avg_atanh, columns=result.subregion_names)
        df.index.name = "subject"
        df.to_csv(csv_path)
        logger.info("Saved atanh CSV: %s", csv_path)
    except ImportError:
        logger.debug("pandas not available; skipping CSV export")

    return npy_path
