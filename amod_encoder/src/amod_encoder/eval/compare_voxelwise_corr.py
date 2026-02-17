"""
Compare voxelwise correlations between predicted activations and
valence/arousal for IAPS and OASIS image sets.

This module corresponds to AMOD script(s):
  - compare_voxelwise_corr_IAPS.m
  - compare_voxelwise_corr_OASIS.m
Key matched choices:
  - For each subject, refit PLS on movie data to get betas
  - Get fc7 activations for IAPS/OASIS images
  - Predict voxelwise activations: enc_prediction = [ones acts] * b
  - Correlate predicted activations with [zscore(val'), zscore(arousal'),
    zscore(val').*zscore(arousal')] — valence, arousal, interaction
  - Store voxelwise Rmat across subjects
  - Fisher Z transform (atanh) before group-level ttest
Assumptions / deviations:
  - MATLAB uses CanlabCore fmri_data objects for t-test and NIfTI writing;
    we use scipy.stats.ttest_1samp and nibabel
  - We assume IAPS/OASIS fc7 activations are pre-computed or computed externally
  - The regression includes val, arousal, and their interaction (3 regressors)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

from amod_encoder.eval.metrics import fishers_z
from amod_encoder.utils.logging import get_logger, log_matlab_note

logger = get_logger(__name__)


def correlate_predictions_with_ratings(
    enc_prediction: np.ndarray,
    valence: np.ndarray,
    arousal: np.ndarray,
) -> np.ndarray:
    """Correlate predicted voxelwise activations with valence, arousal, interaction.

    Parameters
    ----------
    enc_prediction : np.ndarray, shape (N_images, V)
        Predicted voxelwise activations from encoding model.
    valence : np.ndarray, shape (N_images,)
        Valence ratings for each image.
    arousal : np.ndarray, shape (N_images,)
        Arousal ratings for each image.

    Returns
    -------
    np.ndarray, shape (V, 3)
        Voxelwise correlations with [valence, arousal, val*arousal].

    Notes
    -----
    MATLAB:
        [voxelwise_correlations, p] = corr(
            squeeze(enc_prediction),
            [zscore(val'), zscore(arousal'), zscore(val').*zscore(arousal')]
        );
    """
    log_matlab_note(
        logger,
        "compare_voxelwise_corr_IAPS.m / OASIS.m",
        f"Correlating {enc_prediction.shape[1]} voxels with val/arousal/interaction",
    )

    # Z-score valence and arousal (matching MATLAB zscore)
    val_z = stats.zscore(valence, ddof=0)
    arous_z = stats.zscore(arousal, ddof=0)
    interaction_z = val_z * arous_z

    # Build regressor matrix: (N_images, 3)
    regressors = np.column_stack([val_z, arous_z, interaction_z])

    # Correlate each voxel's predictions with the 3 regressors
    # MATLAB corr(A, B) where A is (N, V) and B is (N, 3) → (V, 3)
    voxelwise_corr = np.zeros((enc_prediction.shape[1], 3))
    for reg_idx in range(3):
        for v in range(enc_prediction.shape[1]):
            r, _ = stats.pearsonr(enc_prediction[:, v], regressors[:, reg_idx])
            voxelwise_corr[v, reg_idx] = r

    return voxelwise_corr


def group_level_ttest_voxelwise(
    rmat: np.ndarray,
    excluded_voxels: np.ndarray,
    alpha: float = 0.005,
) -> dict:
    """Perform group-level one-sample t-test on Fisher Z-transformed correlations.

    Parameters
    ----------
    rmat : np.ndarray, shape (N_subjects, N_voxels, 3)
        Voxelwise correlations per subject (val, arousal, interaction).
    excluded_voxels : np.ndarray, shape (N_subjects, N_voxels_mask)
        Boolean array indicating removed voxels per subject.
    alpha : float
        Significance threshold (uncorrected). MATLAB uses 0.005.

    Returns
    -------
    dict with keys per regressor:
        't_stats': np.ndarray — t-statistics
        'p_values': np.ndarray — p-values
        'significant': np.ndarray — boolean mask at alpha threshold
        'atanh_data': np.ndarray — Fisher Z data used for test

    Notes
    -----
    MATLAB:
        regression_object.dat = squeeze(atanh(Rmat(:, any(excluded_voxels==0), 1)))';
        table(threshold(ttest(regression_object), .005, 'UNC'));
    """
    log_matlab_note(
        logger,
        "compare_voxelwise_corr_IAPS.m",
        f"Group t-test on {rmat.shape[0]} subjects, {rmat.shape[1]} voxels",
    )

    # Identify voxels present in all subjects
    # MATLAB: any(excluded_voxels==0) means any subject has this voxel active
    valid_voxels = np.any(excluded_voxels == 0, axis=0) if excluded_voxels.ndim == 2 else ~excluded_voxels

    results = {}
    regressor_names = ["valence", "arousal", "interaction"]

    for reg_idx, reg_name in enumerate(regressor_names):
        # Extract data for valid voxels, Fisher Z transform
        data = rmat[:, valid_voxels, reg_idx] if rmat.ndim == 3 else rmat[:, valid_voxels]
        atanh_data = fishers_z(data)

        # One-sample t-test (H0: mean = 0)
        t_stats, p_values = stats.ttest_1samp(atanh_data, 0, axis=0, nan_policy="omit")

        results[reg_name] = {
            "t_stats": t_stats,
            "p_values": p_values,
            "significant": p_values < alpha,
            "atanh_data": atanh_data,
            "n_significant": int(np.nansum(p_values < alpha)),
        }

        logger.info(
            "%s: %d/%d voxels significant (p < %.3f, uncorrected)",
            reg_name,
            results[reg_name]["n_significant"],
            int(valid_voxels.sum()) if valid_voxels.ndim else len(valid_voxels),
            alpha,
        )

    return results
