"""Tests for temporal alignment (align.py).

Validates that resample_poly alignment correctly maps stimulus frames → TRs.

MATLAB reference: develop_encoding_models_amygdala.m
  timecourse = resample(double(imageFeatures'), n_TRs, n_images);
  Resamples from n_images to n_TRs using polyphase FIR.

Key properties to verify:
  - Output has correct shape (n_trs, n_features)
  - Mean is roughly preserved (no wild scaling)
  - Edge cases handled (n_frames == n_trs, n_frames >> n_trs)
"""

from __future__ import annotations

import numpy as np
import pytest

from amod_encoder.stimuli.align import align_features_to_trs


class TestAlignFeatures:
    """Tests for align_features_to_trs()."""

    def test_output_shape_downsample(self):
        """Downsampling: n_frames > n_trs → output is (n_trs, n_features)."""
        rng = np.random.default_rng(42)
        features = rng.standard_normal((1000, 50))
        result = align_features_to_trs(features, n_trs=200)
        assert result.shape == (200, 50)

    def test_output_shape_upsample(self):
        """Upsampling: n_frames < n_trs → still works."""
        rng = np.random.default_rng(42)
        features = rng.standard_normal((50, 10))
        result = align_features_to_trs(features, n_trs=200)
        assert result.shape == (200, 10)

    def test_identity_when_equal(self):
        """When n_frames == n_trs, output ≈ input (no resampling needed)."""
        rng = np.random.default_rng(42)
        features = rng.standard_normal((100, 5))
        result = align_features_to_trs(features, n_trs=100)
        # With resample_poly, p/q = 1/1 → should be identity
        np.testing.assert_allclose(result, features, atol=1e-10)

    def test_mean_approximately_preserved(self):
        """Resampling should roughly preserve the mean signal level."""
        rng = np.random.default_rng(42)
        features = rng.standard_normal((1000, 20)) + 5.0  # non-zero mean
        result = align_features_to_trs(features, n_trs=200)
        # Means should be within ~20% (polyphase filter can cause small shifts)
        original_mean = np.mean(features, axis=0)
        resampled_mean = np.mean(result, axis=0)
        np.testing.assert_allclose(resampled_mean, original_mean, rtol=0.3)

    def test_all_finite(self):
        """Output should have no NaN or Inf values."""
        rng = np.random.default_rng(42)
        features = rng.standard_normal((500, 10))
        result = align_features_to_trs(features, n_trs=150)
        assert np.all(np.isfinite(result))

    def test_large_downsample_ratio(self):
        """Should handle large downsampling ratios (e.g., 10:1)."""
        rng = np.random.default_rng(42)
        features = rng.standard_normal((5000, 8))
        result = align_features_to_trs(features, n_trs=500)
        assert result.shape == (500, 8)

    def test_single_feature(self):
        """Should work with a single feature column."""
        rng = np.random.default_rng(42)
        features = rng.standard_normal((300, 1))
        result = align_features_to_trs(features, n_trs=100)
        assert result.shape == (100, 1)
