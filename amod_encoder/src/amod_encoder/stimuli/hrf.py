"""
HRF (Hemodynamic Response Function) convolution matching MATLAB spm_hrf.

This module corresponds to AMOD script(s):
  - develop_encoding_models_amygdala.m:
      for i = 1:size(features, 2)
          tmp = conv(double(features(:,i)), spm_hrf(1));
          conv_features(:,i) = tmp(:);
      end
      timematched_features = conv_features(1:size(masked_dat.dat,2), :);
  - develop_encoding_models_subregions.m:
      (same convolution but applied to raw features BEFORE resampling;
       then result is resampled to BOLD length)
Key matched choices:
  - SPM canonical HRF: double-gamma function
  - spm_hrf(1) evaluates at dt=1 second resolution
  - Convolution is linear (numpy.convolve with mode='full')
  - After convolution, truncate to first n_trs samples
  - No temporal or dispersion derivatives used
Assumptions / deviations:
  - Our SPM HRF implementation follows the standard double-gamma parameterization
  - Parameters: peak delay=6, undershoot delay=16, peak dispersion=1,
    undershoot dispersion=1, peak-to-undershoot ratio=6, onset=0, length=32s
  - These are the SPM12 defaults used by spm_hrf(dt)
  - Verified against SPM12 source code
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma as gamma_dist

from amod_encoder.utils.logging import get_logger, log_matlab_note

logger = get_logger(__name__)

# SPM canonical HRF parameters (from SPM12 spm_hrf.m)
_SPM_HRF_DEFAULTS = {
    "peak_delay": 6.0,        # delay of response (relative to onset) [seconds]
    "undershoot_delay": 16.0,  # delay of undershoot [seconds]
    "peak_disp": 1.0,         # dispersion of response
    "undershoot_disp": 1.0,   # dispersion of undershoot
    "ratio": 6.0,             # ratio of response to undershoot
    "onset": 0.0,             # onset [seconds]
    "length": 32.0,           # length of kernel [seconds]
}


def spm_hrf(dt: float = 1.0, **kwargs) -> np.ndarray:
    """Generate SPM canonical HRF kernel, matching MATLAB spm_hrf(dt).

    Parameters
    ----------
    dt : float
        Time resolution in seconds. MATLAB uses spm_hrf(1) → dt=1s.
    **kwargs
        Override default HRF parameters (peak_delay, undershoot_delay, etc.).

    Returns
    -------
    np.ndarray
        HRF kernel sampled at ``dt`` resolution.
        Shape: (n_samples,) where n_samples = floor(length / dt) + 1.

    Notes
    -----
    SPM12 spm_hrf.m implementation:
        u   = [0:dt:p(7)]' - p(6);
        hrf = gam_pdf(u, p(1)/p(3), dt/p(3)) -
              gam_pdf(u, p(2)/p(4), dt/p(4)) / p(5);
        hrf = hrf / sum(hrf);

    Where gam_pdf is the gamma PDF.
    """
    params = {**_SPM_HRF_DEFAULTS, **kwargs}

    p_delay = params["peak_delay"]
    u_delay = params["undershoot_delay"]
    p_disp = params["peak_disp"]
    u_disp = params["undershoot_disp"]
    ratio = params["ratio"]
    onset = params["onset"]
    length = params["length"]

    # Time vector
    u = np.arange(0, length + dt, dt) - onset

    # Gamma PDFs (matching SPM's parameterization)
    # SPM uses: gamma_pdf(u, shape=delay/dispersion, scale=dt/dispersion)
    # scipy.stats.gamma.pdf(x, a, scale=scale) where a = shape
    peak_shape = p_delay / p_disp
    peak_scale = dt / p_disp
    under_shape = u_delay / u_disp
    under_scale = dt / u_disp

    hrf = (
        gamma_dist.pdf(u, peak_shape, scale=peak_scale)
        - gamma_dist.pdf(u, under_shape, scale=under_scale) / ratio
    )

    # Normalize to unit sum (matching SPM)
    hrf = hrf / hrf.sum()

    log_matlab_note(
        logger,
        "spm_hrf(1)",
        f"Generated SPM canonical HRF: dt={dt}s, {len(hrf)} samples, "
        f"peak_delay={p_delay}s, length={length}s",
    )

    return hrf


def convolve_features_with_hrf(
    features: np.ndarray,
    n_trs: int,
    hrf_model: str = "spm_canonical",
    dt: float = 1.0,
) -> np.ndarray:
    """Convolve each feature column with the HRF and truncate to n_trs.

    This reproduces MATLAB's:
        for i = 1:size(features, 2)
            tmp = conv(double(features(:,i)), spm_hrf(1));
            conv_features(:,i) = tmp(:);
        end
        timematched_features = conv_features(1:size(masked_dat.dat,2), :);

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (T, N_features) — already resampled to BOLD rate.
    n_trs : int
        Number of TRs to keep after convolution (truncation length).
    hrf_model : str
        HRF model name. Only 'spm_canonical' is supported.
    dt : float
        HRF sampling rate in seconds.

    Returns
    -------
    np.ndarray
        Convolved and truncated features, shape (n_trs, N_features).

    Notes
    -----
    MATLAB conv() is linear convolution with 'full' mode by default.
    Output length = len(signal) + len(kernel) - 1.
    We then truncate to first n_trs samples.
    """
    if hrf_model != "spm_canonical":
        raise ValueError(f"Unknown HRF model: {hrf_model}")

    hrf = spm_hrf(dt)

    n_timepoints, n_features = features.shape

    log_matlab_note(
        logger,
        "develop_encoding_models_amygdala.m",
        f"Convolving {n_features} features with HRF (len={len(hrf)}), "
        f"truncating to {n_trs} TRs",
    )

    # Convolve each column independently (matching MATLAB's column-wise conv)
    conv_features = np.zeros((n_timepoints + len(hrf) - 1, n_features), dtype=np.float64)
    for i in range(n_features):
        conv_features[:, i] = np.convolve(features[:, i].astype(np.float64), hrf, mode="full")

    # Truncate to match BOLD length: timematched_features = conv_features(1:n_trs, :)
    timematched = conv_features[:n_trs, :]

    logger.info(
        "HRF convolution: (%d, %d) → (%d, %d) [truncated from %d]",
        n_timepoints,
        n_features,
        timematched.shape[0],
        timematched.shape[1],
        conv_features.shape[0],
    )

    return timematched
