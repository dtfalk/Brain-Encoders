"""
ROI (Region of Interest) definition and masking — arbitrary mask-based, ROI-agnostic.

This module corresponds to AMOD script(s):
  - develop_encoding_models_amygdala.m  (apply_mask with canlab2018 'Amy')
  - develop_encoding_models_subregions.m  (apply_mask with canlab2018 subregion masks)
  - compile_matrices.m  (excluded_voxels = masked_dat.removed_voxels)
Key matched choices:
  - Applies a binary NIfTI mask to 4D BOLD data
  - Tracks 'removed_voxels' (voxels in mask that are zero/NaN in BOLD)
  - Returns voxel × time matrix matching MATLAB's masked_dat.dat
Assumptions / deviations:
  - MATLAB uses CanlabCore atlas functions (load_atlas, select_atlas_subset);
    we require a pre-made NIfTI mask instead (ROI-agnostic)
  - User must provide mask in the same space as the BOLD data
  - We use nilearn for resampling if mask resolution differs from BOLD
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

from amod_encoder.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MaskedData:
    """Container for masked BOLD data, mirroring MATLAB's masked_dat structure.

    Attributes
    ----------
    data : np.ndarray
        Array of shape (V, T) where V = number of active voxels, T = timepoints.
        This matches MATLAB's masked_dat.dat orientation.
    voxel_indices : np.ndarray
        1D integer array of linear voxel indices into the 3D mask volume.
    removed_voxels : np.ndarray
        Boolean array over ALL mask voxels: True = removed (no signal).
    mask_img : nib.Nifti1Image
        The mask image used.
    roi_name : str
        Human-readable name for this ROI.
    n_active_voxels : int
        Number of voxels with valid signal.
    n_trs : int
        Number of timepoints.
    """

    data: np.ndarray
    voxel_indices: np.ndarray
    removed_voxels: np.ndarray
    mask_img: nib.Nifti1Image
    roi_name: str
    n_active_voxels: int = 0
    n_trs: int = 0

    def __post_init__(self):
        self.n_active_voxels = self.data.shape[0]
        self.n_trs = self.data.shape[1]


def load_mask(mask_path: Path) -> nib.Nifti1Image:
    """Load a NIfTI mask file.

    Parameters
    ----------
    mask_path : Path
        Path to binary NIfTI mask.

    Returns
    -------
    nib.Nifti1Image
        Loaded mask image.
    """
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"ROI mask not found: {mask_path}")
    mask_img = nib.load(str(mask_path))
    logger.info("Loaded mask: %s, shape=%s", mask_path.name, mask_img.shape)
    return mask_img


def apply_mask(
    bold_data: np.ndarray,
    bold_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    roi_name: str = "unknown",
) -> MaskedData:
    """Apply a binary mask to 4D BOLD data, returning a MaskedData object.

    This mirrors MATLAB's:
        masked_dat = apply_mask(dat, select_atlas_subset(load_atlas('canlab2018'), {'Amy'}))

    Parameters
    ----------
    bold_data : np.ndarray
        4D BOLD data of shape (X, Y, Z, T).
    bold_img : nib.Nifti1Image
        BOLD image (for affine/header).
    mask_img : nib.Nifti1Image
        Binary NIfTI mask (1 = include, 0 = exclude).
    roi_name : str
        Name for this ROI (for logging/provenance).

    Returns
    -------
    MaskedData
        Masked data with voxel × time array and metadata.

    Notes
    -----
    MATLAB's apply_mask:
    1. Resamples mask to BOLD space if needed
    2. Extracts voxels where mask > 0
    3. Sets removed_voxels for any all-zero timeseries
    """
    # Resample mask to BOLD space if affines/shapes differ
    mask_resampled = resample_to_img(
        mask_img, bold_img, interpolation="nearest"
    )
    mask_data = np.asarray(mask_resampled.dataobj).astype(bool)

    # Handle 4D mask (take first volume) or 3D
    if mask_data.ndim == 4:
        mask_data = mask_data[..., 0]

    # Ensure mask shape matches BOLD spatial dims
    if mask_data.shape != bold_data.shape[:3]:
        raise ValueError(
            f"Mask shape {mask_data.shape} doesn't match BOLD spatial dims "
            f"{bold_data.shape[:3]} after resampling"
        )

    # Get all voxel positions within the mask
    mask_indices = np.where(mask_data.ravel())[0]
    n_mask_voxels = len(mask_indices)
    logger.info("ROI '%s': %d voxels in mask", roi_name, n_mask_voxels)

    # Extract time series for each mask voxel: (V_mask, T)
    bold_flat = bold_data.reshape(-1, bold_data.shape[3])  # (n_voxels_total, T)
    masked_ts = bold_flat[mask_indices, :]  # (V_mask, T)

    # Identify removed voxels (all-zero or all-NaN timeseries)
    # This matches MATLAB's masked_dat.removed_voxels
    is_zero = np.all(masked_ts == 0, axis=1)
    is_nan = np.all(np.isnan(masked_ts), axis=1)
    removed = is_zero | is_nan

    # Keep only active voxels
    active_mask = ~removed
    active_data = masked_ts[active_mask, :]  # (V_active, T)
    active_indices = mask_indices[active_mask]

    logger.info(
        "ROI '%s': %d active voxels (%d removed)",
        roi_name,
        int(active_mask.sum()),
        int(removed.sum()),
    )

    return MaskedData(
        data=active_data.astype(np.float64),
        voxel_indices=active_indices,
        removed_voxels=removed,
        mask_img=mask_resampled,
        roi_name=roi_name,
    )


def mean_roi_timeseries(masked_data: MaskedData) -> np.ndarray:
    """Compute the mean time series across active voxels.

    Parameters
    ----------
    masked_data : MaskedData
        Masked BOLD data.

    Returns
    -------
    np.ndarray
        1D array of shape (T,) — mean across voxels for each TR.
    """
    return masked_data.data.mean(axis=0)
