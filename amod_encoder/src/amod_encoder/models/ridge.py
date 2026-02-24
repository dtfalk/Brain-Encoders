"""
Ridge Encoding Model
====================

Regularised least-squares baseline and GPU-accelerable alternative to PLS.

Design Principles:
    - Betas stored in same ``(D+1, V)`` format as PLS for interchangeability
    - ``alpha = 1.0`` default gives mild L2 regularisation
    - When ``ComputeBackend`` is ``torch``, Ridge runs on GPU
    - Useful for sanity checks, fast debugging, and ``roi_mean`` mode

MATLAB Correspondence:
    - MATLAB pipeline uses PLS exclusively; Ridge is an extension
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from amod_encoder.models.base import EncodingModel
from amod_encoder.utils.compute_backend import ComputeBackend
from amod_encoder.utils.logging import get_logger

logger = get_logger(__name__)


class RidgeEncodingModel(EncodingModel):
    """Voxelwise ridge regression encoding model.

    Parameters
    ----------
    alpha : float
        Regularization strength. Default 1.0.
    standardize_X : bool
        Whether to z-score features before fitting.
    standardize_Y : bool
        Whether to z-score targets before fitting.
    compute : ComputeBackend or None
        Compute backend. If None, uses CPU.

    Attributes
    ----------
    _betas : np.ndarray or None
        Coefficient matrix (D+1, V) after fitting.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        standardize_X: bool = False,
        standardize_Y: bool = False,
        compute: Optional[ComputeBackend] = None,
    ):
        self.alpha = alpha
        self.standardize_X = standardize_X
        self.standardize_Y = standardize_Y
        self.compute = compute or ComputeBackend()
        self._betas: Optional[np.ndarray] = None
        self._intercept: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit ridge regression model.

        Parameters
        ----------
        X : np.ndarray, shape (T, D)
            Feature matrix.
        Y : np.ndarray, shape (T, V)
            Brain data matrix.

        Notes
        -----
        Ridge solution: B = (X'X + alpha*I)^{-1} X'Y
        With intercept: center X and Y, then B0 = Y_mean - X_mean @ B
        """
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        T, D = X.shape
        V = Y.shape[1]

        logger.info("Fitting ridge: X(%d,%d) → Y(%d,%d), alpha=%.4f", T, D, T, V, self.alpha)

        if self.compute.is_torch:
            self._fit_torch(X, Y)
        else:
            self._fit_numpy(X, Y)

    def _fit_numpy(self, X: np.ndarray, Y: np.ndarray) -> None:
        """CPU ridge via numpy."""
        T, D = X.shape
        V = Y.shape[1]

        # Center
        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        Xc = X - X_mean
        Yc = Y - Y_mean

        # Optional X standardization
        if self.standardize_X:
            X_std = Xc.std(axis=0, ddof=1)
            X_std[X_std == 0] = 1.0
            Xc = Xc / X_std
        else:
            X_std = np.ones(D)

        # Optional Y standardization
        if self.standardize_Y:
            Y_std = Yc.std(axis=0, ddof=1)
            Y_std[Y_std == 0] = 1.0
            Yc = Yc / Y_std
        else:
            Y_std = np.ones(V)

        # Ridge solution
        XtX = Xc.T @ Xc
        XtY = Xc.T @ Yc
        I = np.eye(D) * self.alpha
        coef = np.linalg.solve(XtX + I, XtY)  # (D, V)

        # Undo standardization
        if self.standardize_X:
            coef = coef / X_std.reshape(-1, 1)
        if self.standardize_Y:
            coef = coef * Y_std.reshape(1, -1)

        # Intercept (computed from original-scale means)
        intercept = Y_mean - X_mean @ coef

        self._intercept = intercept
        self._betas = np.vstack([intercept.reshape(1, -1), coef])
        logger.info("Ridge fit (numpy): betas shape = %s", self._betas.shape)

    def _fit_torch(self, X: np.ndarray, Y: np.ndarray) -> None:
        """GPU-accelerated ridge via PyTorch."""
        from amod_encoder.utils.compute_backend import get_torch
        torch = get_torch()
        self.compute.warn_numerical_difference("ridge regression")

        T, D = X.shape
        V = Y.shape[1]

        X_t = self.compute.to_tensor(X)
        Y_t = self.compute.to_tensor(Y)

        # Center
        X_mean = X_t.mean(dim=0)
        Y_mean = Y_t.mean(dim=0)
        Xc = X_t - X_mean
        Yc = Y_t - Y_mean

        # Optional Y standardization
        if self.standardize_Y:
            Y_std = Yc.std(dim=0, correction=1)
            Y_std[Y_std == 0] = 1.0
            Yc = Yc / Y_std
        else:
            Y_std = torch.ones(V, device=Y_t.device, dtype=Y_t.dtype)

        # Ridge
        XtX = Xc.T @ Xc
        XtY = Xc.T @ Yc
        I = torch.eye(D, device=X_t.device, dtype=X_t.dtype) * self.alpha
        coef = torch.linalg.solve(XtX + I, XtY)

        # Undo Y standardization
        if self.standardize_Y:
            coef = coef * Y_std.unsqueeze(0)

        # Intercept
        intercept = Y_mean - X_mean @ coef

        # Convert back to numpy
        self._intercept = self.compute.to_numpy(intercept)
        coef_np = self.compute.to_numpy(coef)
        self._betas = np.vstack([self._intercept.reshape(1, -1), coef_np])
        logger.info("Ridge fit (torch): betas shape = %s", self._betas.shape)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._betas is None:
            raise RuntimeError("Model not fitted.")
        return self.predict_with_intercept(X)

    @property
    def betas(self) -> np.ndarray:
        if self._betas is None:
            raise RuntimeError("Model not fitted.")
        return self._betas

    @property
    def intercept(self) -> np.ndarray:
        if self._intercept is None:
            raise RuntimeError("Model not fitted.")
        return self._intercept
