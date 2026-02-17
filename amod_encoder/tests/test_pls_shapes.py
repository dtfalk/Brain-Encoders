"""Tests for PLS encoding model shapes and behavior (pls.py).

Validates that PLSEncodingModel produces correctly-shaped outputs
and matches expected conventions from the MATLAB pipeline.

MATLAB reference: develop_encoding_models_amygdala.m
  [~, ~, ~, ~, BETA] = plsregress(X, Y, 20);
  betahat = BETA;  % (D+1, V) including intercept row
  testpredicted = [ones(N_test, 1) X_test] * betahat;

Key properties:
  - betas shape: (n_features + 1, n_voxels)
  - predictions shape: (n_samples, n_voxels)
  - intercept shape: (n_voxels,)
  - n_components capped at min(T-1, D, V)
"""

from __future__ import annotations

import numpy as np
import pytest

from amod_encoder.models.pls import PLSEncodingModel


class TestPLSShapes:
    """Tests for PLS model shape conventions."""

    @pytest.fixture()
    def synthetic_data(self):
        """Create synthetic X, Y data mimicking the pipeline."""
        rng = np.random.default_rng(42)
        T, D, V = 200, 50, 30  # small for speed
        X = rng.standard_normal((T, D))
        # Make Y somewhat correlated with X
        W = rng.standard_normal((D, V)) * 0.1
        Y = X @ W + rng.standard_normal((T, V)) * 0.5
        return X, Y, T, D, V

    def test_betas_shape(self, synthetic_data):
        """betas should be (D+1, V) — intercept + features."""
        X, Y, T, D, V = synthetic_data
        model = PLSEncodingModel(n_components=10)
        model.fit(X, Y)
        assert model.betas.shape == (D + 1, V)

    def test_intercept_shape(self, synthetic_data):
        """intercept should be (V,)."""
        X, Y, T, D, V = synthetic_data
        model = PLSEncodingModel(n_components=10)
        model.fit(X, Y)
        assert model.intercept.shape == (V,)

    def test_predictions_shape(self, synthetic_data):
        """predict() should return (T_test, V)."""
        X, Y, T, D, V = synthetic_data
        model = PLSEncodingModel(n_components=10)
        model.fit(X, Y)
        preds = model.predict(X[:50])
        assert preds.shape == (50, V)

    def test_predict_with_intercept(self, synthetic_data):
        """predict_with_intercept should match [ones X] @ betas."""
        X, Y, T, D, V = synthetic_data
        model = PLSEncodingModel(n_components=10)
        model.fit(X, Y)
        X_test = X[:20]
        preds = model.predict_with_intercept(X_test)
        manual = np.column_stack([np.ones(20), X_test]) @ model.betas
        np.testing.assert_allclose(preds, manual, atol=1e-10)

    def test_n_components_capped(self):
        """n_components should be capped at min(T-1, D, V)."""
        rng = np.random.default_rng(42)
        T, D, V = 15, 50, 30  # T-1=14 is the bottleneck
        X = rng.standard_normal((T, D))
        Y = rng.standard_normal((T, V))
        model = PLSEncodingModel(n_components=20)  # request 20 > T-1=14
        model.fit(X, Y)
        # Should succeed without error; components internally capped
        assert model.betas.shape == (D + 1, V)

    def test_single_voxel(self):
        """Should work with Y as a single column."""
        rng = np.random.default_rng(42)
        T, D = 100, 20
        X = rng.standard_normal((T, D))
        Y = rng.standard_normal((T, 1))
        model = PLSEncodingModel(n_components=5)
        model.fit(X, Y)
        assert model.betas.shape == (D + 1, 1)
        preds = model.predict(X[:10])
        assert preds.shape == (10, 1)

    def test_deterministic(self):
        """Two fits with same data should produce identical betas."""
        rng = np.random.default_rng(42)
        T, D, V = 100, 30, 10
        X = rng.standard_normal((T, D))
        Y = rng.standard_normal((T, V))

        model1 = PLSEncodingModel(n_components=5)
        model1.fit(X, Y)

        model2 = PLSEncodingModel(n_components=5)
        model2.fit(X, Y)

        np.testing.assert_allclose(model1.betas, model2.betas, atol=1e-12)

    def test_nonzero_predictions(self, synthetic_data):
        """Predictions should not all be zero (model should learn something)."""
        X, Y, T, D, V = synthetic_data
        model = PLSEncodingModel(n_components=10)
        model.fit(X, Y)
        preds = model.predict(X)
        assert np.any(np.abs(preds) > 1e-6)
