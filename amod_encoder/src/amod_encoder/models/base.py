"""
Encoding Model Base Class
=========================

Abstract interface for voxelwise encoding models.

Design Principles:
    - ``fit(X, Y) → betas``: matches MATLAB ``plsregress(X, Y, n_comp)``
    - ``predict(X) → Yhat``: matches ``[ones(N,1) X] * betas``
    - Betas include intercept as row 0: shape ``(D+1, V)``
    - Concrete implementations: PLS (``pls.py``) and Ridge (``ridge.py``)

MATLAB Correspondence:
    - ``[~,~,~,~,b] = plsregress(X, Y, n_comp)``
    - ``b`` is ``(D+1, V)`` where ``b(1,:) = intercept``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class EncodingModel(ABC):
    """Abstract base class for voxelwise encoding models.

    All encoding models take features X of shape (T, D) and brain data Y of
    shape (T, V), and produce betas of shape (D+1, V) where the first row
    is the intercept term.

    This matches MATLAB's plsregress output:
        [~,~,~,~,b] = plsregress(X, Y, n_comp)
        % b is (D+1, V) where b(1,:) = intercept, b(2:end,:) = coefficients

    Prediction:
        yhat = [ones(N,1) X] * b
    """

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit the encoding model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (T, D). T = timepoints, D = feature dims.
        Y : np.ndarray
            Brain data, shape (T, V). V = voxels (or 1 for ROI mean).
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict brain activity from features.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (N, D).

        Returns
        -------
        np.ndarray
            Predicted brain data, shape (N, V).
        """
        ...

    @property
    @abstractmethod
    def betas(self) -> np.ndarray:
        """Model coefficients, shape (D+1, V). First row = intercept.

        This matches MATLAB plsregress output `b` where:
            b(1, :)     = intercept
            b(2:end, :) = feature coefficients

        Prediction: yhat = [ones(N,1) X] * betas
        """
        ...

    @property
    @abstractmethod
    def intercept(self) -> np.ndarray:
        """Intercept vector, shape (V,)."""
        ...

    def predict_with_intercept(self, X: np.ndarray) -> np.ndarray:
        """Predict using the MATLAB convention: [ones(N,1) X] * betas.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (N, D).

        Returns
        -------
        np.ndarray
            Predicted brain data, shape (N, V).
        """
        N = X.shape[0]
        X_aug = np.column_stack([np.ones(N), X])
        return X_aug @ self.betas
