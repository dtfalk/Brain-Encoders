"""
fc7 feature loader from OSF .mat file.

This module corresponds to AMOD script(s):
  - extract_features.m  (saves video_imageFeatures to .mat)
  - develop_encoding_models_amygdala.m  (loads 500_days_of_summer_fc7_features.mat)
Key matched choices:
  - Variable name in .mat is 'video_imageFeatures'
  - Shape is (N_frames, 4096) where N_frames = number of sampled frames
  - Features are extracted from every 5th frame using EmoNet fc7 layer
  - lendelta = size(video_imageFeatures, 1) is the original frame count
  - Data is loaded as float64 (MATLAB default)
Assumptions / deviations:
  - .mat file may be v7.3 (HDF5) or earlier; we handle both via h5py/scipy
  - We do NOT re-extract features from video; we use the pre-computed .mat
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from amod_encoder.utils.logging import get_logger

logger = get_logger(__name__)


def load_fc7_features(mat_path: Path) -> np.ndarray:
    """Load fc7 image features from the OSF .mat file.

    Parameters
    ----------
    mat_path : Path
        Path to 500_days_of_summer_fc7_features.mat.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (N_frames, N_features).
        - N_frames: number of sampled movie frames (every 5th frame)
        - N_features: 4096 (fc7 layer dimensionality in EmoNet/VGG)

    Raises
    ------
    FileNotFoundError
        If the .mat file does not exist.
    KeyError
        If 'video_imageFeatures' variable is not found in the file.

    Notes
    -----
    MATLAB loads this as:
        load('500_days_of_summer_fc7_features.mat')
        lendelta = size(video_imageFeatures, 1);  % number of frames

    The file was saved with '-v7.3' format (HDF5-based).
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"fc7 features file not found: {mat_path}")

    features = _try_load_mat(mat_path)
    features = np.asarray(features, dtype=np.float64)

    # Ensure shape is (frames, features) — MATLAB may save transposed
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {features.shape}")

    # MATLAB convention: if features has more rows than columns, it's (frames, features)
    # fc7 has 4096 features; frame count should be larger
    if features.shape[1] > features.shape[0]:
        logger.warning(
            "Feature matrix appears transposed (%s); transposing to (frames, features)",
            features.shape,
        )
        features = features.T

    logger.info(
        "Loaded fc7 features: shape=%s (N_frames=%d, N_features=%d)",
        features.shape,
        features.shape[0],
        features.shape[1],
    )
    return features


def _try_load_mat(mat_path: Path) -> np.ndarray:
    """Try loading .mat file using h5py first (v7.3), then scipy (v5/v7).

    Parameters
    ----------
    mat_path : Path
        Path to .mat file.

    Returns
    -------
    np.ndarray
        The video_imageFeatures variable.
    """
    var_name = "video_imageFeatures"

    # Try HDF5 format first (MATLAB v7.3, saved with '-v7.3')
    try:
        import h5py

        with h5py.File(str(mat_path), "r") as f:
            if var_name in f:
                # h5py loads HDF5 datasets; need to read into memory
                data = f[var_name][:]
                # HDF5/MATLAB stores in column-major; h5py reads row-major
                # MATLAB (N_features, N_frames) → Python (N_frames, N_features)
                if data.ndim == 2:
                    data = data.T
                logger.info("Loaded .mat via h5py (v7.3 format)")
                return data
            else:
                available = list(f.keys())
                raise KeyError(
                    f"Variable '{var_name}' not found. Available: {available}"
                )
    except (OSError, Exception) as e:
        logger.debug("h5py load failed (%s), trying scipy.io.loadmat", e)

    # Fall back to scipy for older .mat formats
    from scipy.io import loadmat

    mat_data = loadmat(str(mat_path))
    if var_name in mat_data:
        logger.info("Loaded .mat via scipy.io.loadmat (v5/v7 format)")
        return mat_data[var_name]
    else:
        available = [k for k in mat_data.keys() if not k.startswith("__")]
        raise KeyError(
            f"Variable '{var_name}' not found. Available: {available}"
        )
