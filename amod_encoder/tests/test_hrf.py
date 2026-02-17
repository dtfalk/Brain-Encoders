"""Tests for HRF generation and convolution (hrf.py).

Validates against known properties of the SPM canonical double-gamma HRF.

MATLAB reference: spm_hrf(1) in develop_encoding_models_amygdala.m
  - Peak at ~5-6 s
  - Undershoot at ~15-16 s
  - Length: 32 samples at dt=1s → 33 points
  - Integral close to 1 (but not exactly)
"""

from __future__ import annotations

import numpy as np
import pytest

from amod_encoder.stimuli.hrf import spm_hrf, convolve_features_with_hrf


class TestSpmHrf:
    """Tests for the SPM canonical HRF."""

    def test_length_default(self):
        """Default HRF at dt=1 should be 33 points (0..32)."""
        hrf = spm_hrf(dt=1.0)
        assert hrf.shape == (33,)

    def test_length_fine_resolution(self):
        """At dt=0.1, length should be 321 points."""
        hrf = spm_hrf(dt=0.1)
        assert hrf.shape == (321,)

    def test_peak_location(self):
        """Peak (positive max) should be at ~5-6 seconds."""
        hrf = spm_hrf(dt=1.0)
        peak_idx = np.argmax(hrf)
        # Peak is at index 5 (5 seconds)
        assert 4 <= peak_idx <= 6, f"Peak at index {peak_idx}, expected 4-6"

    def test_undershoot(self):
        """There should be a negative undershoot after the peak."""
        hrf = spm_hrf(dt=1.0)
        # The undershoot is around 15-16 seconds
        assert np.any(hrf[10:25] < 0), "HRF should have a negative undershoot"

    def test_starts_near_zero(self):
        """HRF at t=0 should be close to zero."""
        hrf = spm_hrf(dt=1.0)
        assert abs(hrf[0]) < 1e-10

    def test_ends_near_zero(self):
        """HRF at t=32 should be close to zero."""
        hrf = spm_hrf(dt=1.0)
        assert abs(hrf[-1]) < 0.01

    def test_positive_peak_dominates(self):
        """The absolute max should be the positive peak, not the undershoot."""
        hrf = spm_hrf(dt=1.0)
        assert np.max(hrf) > abs(np.min(hrf))

    def test_unit_vector_direction(self):
        """Integral (sum) should be positive."""
        hrf = spm_hrf(dt=1.0)
        assert np.sum(hrf) > 0


class TestConvolveFeatures:
    """Tests for HRF convolution."""

    def test_output_shape(self):
        """Output should be (n_trs, n_features) after truncation."""
        rng = np.random.default_rng(42)
        features = rng.standard_normal((200, 50))
        n_trs = 200
        result = convolve_features_with_hrf(features, n_trs)
        assert result.shape == (n_trs, 50)

    def test_output_shape_truncation(self):
        """If n_trs < n_timepoints, output should be truncated to n_trs."""
        rng = np.random.default_rng(42)
        features = rng.standard_normal((200, 10))
        result = convolve_features_with_hrf(features, n_trs=150)
        assert result.shape == (150, 10)

    def test_convolution_shifts_signal(self):
        """An impulse at t=0 should produce the HRF shape, shifted forward."""
        n, d = 100, 1
        impulse = np.zeros((n, d))
        impulse[0, 0] = 1.0
        result = convolve_features_with_hrf(impulse, n_trs=n)
        hrf = spm_hrf(dt=1.0)
        # First n values of result[:, 0] should match the HRF
        expected_len = min(n, len(hrf))
        np.testing.assert_allclose(
            result[:expected_len, 0], hrf[:expected_len], atol=1e-10
        )

    def test_zero_input(self):
        """Zero input should produce zero output."""
        features = np.zeros((100, 5))
        result = convolve_features_with_hrf(features, n_trs=100)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)
