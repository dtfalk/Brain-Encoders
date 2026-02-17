"""Tests for fc7 feature loading (fc7_mat_loader.py).

Runs WITHOUT the actual dataset — uses synthetic .mat files.

MATLAB reference: extract_features.m
  Variable: video_imageFeatures  (N_frames, 4096)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from amod_encoder.stimuli.fc7_mat_loader import load_fc7_features


@pytest.fixture()
def synthetic_fc7_v5(tmp_path: Path) -> Path:
    """Create a synthetic v5 .mat file with fc7 features."""
    n_frames, n_feat = 500, 4096
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_frames, n_feat)).astype(np.float32)
    mat_path = tmp_path / "fc7_features.mat"
    sio.savemat(str(mat_path), {"video_imageFeatures": data})
    return mat_path


def test_load_shape(synthetic_fc7_v5: Path):
    """Loaded array should keep (N_frames, 4096) shape."""
    features = load_fc7_features(synthetic_fc7_v5)
    assert features.shape == (500, 4096)


def test_load_dtype(synthetic_fc7_v5: Path):
    """Loaded features should be float64 (or at least float)."""
    features = load_fc7_features(synthetic_fc7_v5)
    assert np.issubdtype(features.dtype, np.floating)


def test_values_match(synthetic_fc7_v5: Path):
    """Loaded values should match what was written."""
    rng = np.random.default_rng(42)
    expected = rng.standard_normal((500, 4096)).astype(np.float32)
    features = load_fc7_features(synthetic_fc7_v5)
    np.testing.assert_allclose(features, expected.astype(np.float64), atol=1e-5)


def test_missing_variable(tmp_path: Path):
    """Should raise KeyError if 'video_imageFeatures' is missing."""
    mat_path = tmp_path / "wrong_var.mat"
    sio.savemat(str(mat_path), {"wrong_key": np.zeros((10, 10))})
    with pytest.raises(KeyError):
        load_fc7_features(mat_path)


def test_missing_file():
    """Should raise FileNotFoundError for a nonexistent path."""
    with pytest.raises(FileNotFoundError):
        load_fc7_features(Path("/nonexistent/path/fc7.mat"))
