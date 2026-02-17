"""
Temporal alignment of stimulus features to BOLD TR grid.

This module corresponds to AMOD script(s):
  - develop_encoding_models_amygdala.m:
      features = resample(double(video_imageFeatures), size(masked_dat.dat,2), lendelta);
  - develop_encoding_models_subregions.m:
      features = resample(double(X), size(masked_dat.dat,2), size(X,1));
      (here X is already HRF-convolved before resampling)
Key matched choices:
  - MATLAB resample() uses polyphase anti-aliasing filtering
  - scipy.signal.resample uses FFT-based resampling (Fourier method)
  - These are NOT identical but are the closest scipy equivalent
  - The resampling maps N_frames → N_TRs (ratio = n_trs / n_frames)
  - Applied independently to each feature column
Assumptions / deviations:
  - MATLAB resample(x, p, q) uses a polyphase FIR filter with Kaiser window
  - scipy.signal.resample uses FFT zero-padding/truncation (sinc interpolation)
  - For large resampling factors these produce similar but not identical results
  - scipy.signal.resample_poly is closer to MATLAB's resample (polyphase)
  - We use resample_poly as the best match
  - TODO: validate numerical closeness against MATLAB output for this dataset
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly
from math import gcd

from amod_encoder.utils.logging import get_logger, log_matlab_note

logger = get_logger(__name__)


def align_features_to_trs(
    features: np.ndarray,
    n_trs: int,
    method: str = "resample",
) -> np.ndarray:
    """Resample feature matrix from frame rate to TR rate.

    This reproduces MATLAB's:
        features = resample(double(video_imageFeatures), size(masked_dat.dat,2), lendelta);

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (N_frames, N_features).
    n_trs : int
        Target number of time points (TRs in the BOLD data).
    method : str
        Alignment method. Currently only 'resample' is supported.

    Returns
    -------
    np.ndarray
        Resampled feature matrix of shape (n_trs, N_features), dtype float64.

    Notes
    -----
    MATLAB's resample(x, p, q):
      - p = target length (n_trs)
      - q = source length (n_frames)
      - Uses a polyphase FIR anti-aliasing filter with Kaiser window
      - Applied column-wise

    We use scipy.signal.resample_poly(x, up, down) which is the closest
    Python equivalent to MATLAB's polyphase resample:
      - up = n_trs
      - down = n_frames
      - Simplify by GCD for efficiency
      - Uses a FIR anti-aliasing filter (polyphase implementation)
    """
    if method != "resample":
        raise ValueError(f"Unknown alignment method: {method}. Only 'resample' is supported.")

    n_frames, n_features = features.shape
    log_matlab_note(
        logger,
        "develop_encoding_models_amygdala.m",
        f"resample({n_frames} frames → {n_trs} TRs) for {n_features} features",
    )

    # Simplify the resampling ratio by GCD for efficiency
    # MATLAB: resample(x, n_trs, n_frames)
    common = gcd(n_trs, n_frames)
    up = n_trs // common
    down = n_frames // common

    logger.debug("Resample ratio: up=%d, down=%d (GCD=%d)", up, down, common)

    # Apply resample_poly column-wise (axis=0), matching MATLAB behavior
    resampled = resample_poly(features, up, down, axis=0)

    # resample_poly may produce slightly more or fewer samples due to
    # filter transients; truncate or pad to exact n_trs
    if resampled.shape[0] > n_trs:
        resampled = resampled[:n_trs, :]
    elif resampled.shape[0] < n_trs:
        # Pad with last value (edge case; should rarely happen)
        pad_width = n_trs - resampled.shape[0]
        resampled = np.pad(resampled, ((0, pad_width), (0, 0)), mode="edge")
        logger.warning(
            "resample_poly produced %d samples, expected %d; padded %d",
            resampled.shape[0] - pad_width,
            n_trs,
            pad_width,
        )

    assert resampled.shape == (n_trs, n_features), (
        f"Alignment produced shape {resampled.shape}, expected ({n_trs}, {n_features})"
    )

    logger.info("Aligned features: (%d, %d) → (%d, %d)", n_frames, n_features, n_trs, n_features)
    return resampled.astype(np.float64)
