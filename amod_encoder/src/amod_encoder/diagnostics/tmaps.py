"""
Voxelwise T-Map Writer
======================

Computes group-level t-maps from encoding-model correlations and writes
them as NIfTI images for visualisation in standard neuroimaging tools.

Design Principles:
    - Fisher’s Z transform → average across datasets → group t-test
    - Thresholding: uncorrected p < .05 or FDR q < .05
    - Output is a standard NIfTI volume, viewable in FSLeyes / MRIcroGL
    - ``nibabel`` replaces MATLAB CanlabCore ``statistic_image`` objects

MATLAB Correspondence:
    - write_voxelwise_tmaps_IAPS_OASIS.m → ``write_voxelwise_tmap()``
    - make_parametric_map_amygdala.m → ``write_parametric_map()``
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy import stats

from amod_encoder.eval.metrics import fishers_z
from amod_encoder.utils.logging import get_logger, log_matlab_note

logger = get_logger(__name__)


def write_tmap_nifti(
    data_per_subject: np.ndarray,
    mask_img: nib.Nifti1Image,
    voxel_indices: np.ndarray,
    output_path: Path,
    threshold_p: float = 0.05,
    apply_fdr: bool = False,
    fdr_q: float = 0.05,
) -> Path:
    """Compute voxelwise t-statistics and write as NIfTI t-map.

    Parameters
    ----------
    data_per_subject : np.ndarray, shape (N_subjects, N_voxels)
        Per-subject data (e.g., Fisher Z-transformed correlations).
    mask_img : nib.Nifti1Image
        Reference mask image (for affine and shape).
    voxel_indices : np.ndarray
        Linear indices into the 3D volume for each voxel column.
    output_path : Path
        Output NIfTI file path.
    threshold_p : float
        P-value threshold for uncorrected thresholding.
    apply_fdr : bool
        Whether to apply FDR correction instead of uncorrected.
    fdr_q : float
        FDR q threshold.

    Returns
    -------
    Path
        Path to the written NIfTI file.

    Notes
    -----
    MATLAB:
        stats_object = statistic_image;
        stats_object.volInfo = masked_dat.volInfo;
        stats_object.dat = stats.tstat';
        write(stats_object, 'overwrite');
    """
    log_matlab_note(
        logger,
        "write_voxelwise_tmaps_IAPS_OASIS.m",
        f"Writing t-map: {data_per_subject.shape[0]} subjects, "
        f"{data_per_subject.shape[1]} voxels → {output_path.name}",
    )

    # One-sample t-test at each voxel
    t_stats, p_values = stats.ttest_1samp(data_per_subject, 0, axis=0, nan_policy="omit")

    # Apply threshold
    if apply_fdr:
        from statsmodels.stats.multitest import fdrcorrection

        valid = ~np.isnan(p_values)
        reject = np.zeros_like(p_values, dtype=bool)
        if valid.any():
            reject[valid], _ = fdrcorrection(p_values[valid], alpha=fdr_q)
        t_thresholded = np.where(reject, t_stats, np.nan)
    else:
        t_thresholded = np.where(p_values < threshold_p, t_stats, np.nan)

    # Create 3D volume
    mask_data = np.asarray(mask_img.dataobj)
    vol_shape = mask_data.shape[:3]
    t_volume = np.full(np.prod(vol_shape), np.nan, dtype=np.float32)
    t_volume[voxel_indices] = t_stats.astype(np.float32)
    t_volume = t_volume.reshape(vol_shape)

    # Write NIfTI
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(t_volume, mask_img.affine, mask_img.header)
    nib.save(nii, str(output_path))
    logger.info("T-map written: %s", output_path)

    # Also write thresholded version
    t_vol_thresh = np.full(np.prod(vol_shape), np.nan, dtype=np.float32)
    t_vol_thresh[voxel_indices] = t_thresholded.astype(np.float32)
    t_vol_thresh = t_vol_thresh.reshape(vol_shape)

    thresh_path = output_path.with_name(
        output_path.stem + "_thresholded" + output_path.suffix
    )
    nii_thresh = nib.Nifti1Image(t_vol_thresh, mask_img.affine, mask_img.header)
    nib.save(nii_thresh, str(thresh_path))
    logger.info("Thresholded t-map written: %s", thresh_path)

    return output_path


def average_and_ttest_correlation_maps(
    iaps_data: np.ndarray,
    oasis_data: np.ndarray,
    mask_img: nib.Nifti1Image,
    voxel_indices: np.ndarray,
    output_prefix: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Average IAPS and OASIS Fisher Z maps and compute group t-test.

    Parameters
    ----------
    iaps_data : np.ndarray, shape (N_subjects, N_voxels)
        Fisher Z-transformed IAPS correlation data.
    oasis_data : np.ndarray, shape (N_subjects, N_voxels)
        Fisher Z-transformed OASIS correlation data.
    mask_img : nib.Nifti1Image
        Mask image.
    voxel_indices : np.ndarray
        Voxel indices.
    output_prefix : str
        Prefix for output files (e.g., 'valence', 'arousal', 'interaction').
    output_dir : Path
        Output directory.

    Returns
    -------
    dict
        Maps output name → file path.

    Notes
    -----
    MATLAB:
        dat.dat = (atanh(dat.dat) + tdat.dat) / 2;  % average Fisher Z
        t = ttest(dat);
    """
    log_matlab_note(
        logger,
        "write_voxelwise_tmaps_IAPS_OASIS.m",
        f"Averaging IAPS+OASIS for '{output_prefix}' and computing t-map",
    )

    # Average Fisher Z-transformed data
    avg_data = (fishers_z(iaps_data) + fishers_z(oasis_data)) / 2.0

    output_path = output_dir / f"voxelwise_correlations_{output_prefix}.nii.gz"

    write_tmap_nifti(
        avg_data,
        mask_img,
        voxel_indices,
        output_path,
        threshold_p=0.05,
    )

    return {output_prefix: output_path}
