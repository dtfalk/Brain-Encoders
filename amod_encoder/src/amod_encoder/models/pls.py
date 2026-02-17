"""
PLS (Partial Least Squares) encoding model matching MATLAB plsregress.

This module corresponds to AMOD script(s):
  - develop_encoding_models_amygdala.m:
      [~,~,~,~,b] = plsregress(timematched_features, masked_dat.dat', 20);
  - develop_encoding_models_subregions.m:
      [~,~,~,~,b] = plsregress(features, masked_dat.dat', 20);
  - decode_activation_targets_artificial_stim.m:
      [~,~,~,~,b] = plsregress(double(enc_prediction(train,:)), Y(train,:), 7);
Key matched choices:
  - MATLAB plsregress uses SIMPLS algorithm
  - sklearn PLSRegression uses NIPALS by default (closest available)
  - n_components = 20 (amygdala) or min(20, n_voxels) capped per MATLAB
  - Y is multivariate: all voxels at once (not per-voxel fitting)
  - betas include intercept in first row
  - MATLAB plsregress centers X and Y internally; sklearn does the same
  - MATLAB does NOT standardize (scale) X or Y; sklearn PLSRegression
    has scale=False available but defaults to True — we set scale=False
Assumptions / deviations:
  - MATLAB SIMPLS ≠ sklearn NIPALS exactly, but they optimize the same
    objective and produce very similar (not identical) betas
  - For exact SIMPLS reproduction, a custom implementation would be needed
  - We provide a SIMPLS implementation below as primary, with sklearn fallback
  - TODO: quantify numerical difference on this specific dataset
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from amod_encoder.models.base import EncodingModel
from amod_encoder.utils.logging import get_logger, log_matlab_note

logger = get_logger(__name__)


class PLSEncodingModel(EncodingModel):
    """Voxelwise PLS encoding model matching MATLAB plsregress.

    Parameters
    ----------
    n_components : int
        Number of PLS components. MATLAB uses 20 for amygdala.
    standardize_X : bool
        Whether to standardize features. MATLAB does NOT standardize.
    standardize_Y : bool
        Whether to standardize targets. MATLAB does NOT standardize.

    Attributes
    ----------
    _betas : np.ndarray or None
        Coefficient matrix (D+1, V) after fitting. First row = intercept.
    _model : PLSRegression or None
        Fitted sklearn model (used only for sklearn path).
    """

    def __init__(
        self,
        n_components: int = 20,
        standardize_X: bool = False,
        standardize_Y: bool = False,
    ):
        self.n_components = n_components
        self.standardize_X = standardize_X
        self.standardize_Y = standardize_Y
        self._betas: Optional[np.ndarray] = None
        self._intercept: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit PLS model to features X and brain data Y.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (T, D).
        Y : np.ndarray
            Brain data, shape (T, V).

        Notes
        -----
        MATLAB:
            [~,~,~,~,b] = plsregress(X, Y, 20)
        where X is (T, D), Y is (T, V), b is (D+1, V).

        MATLAB plsregress:
        1. Centers X and Y (subtracts column means)
        2. Computes PLS via SIMPLS algorithm
        3. Returns b where b(1,:) = intercept, b(2:end,:) = coefficients
        4. Prediction: yhat = [ones(N,1) X] * b
        """
        T, D = X.shape
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        T_y, V = Y.shape
        assert T == T_y, f"X has {T} samples but Y has {T_y}"

        # Cap n_components at min(T-1, D, V) to avoid numerical issues
        # MATLAB: min(20, size(masked_dat.dat, 1))
        max_comp = min(T - 1, D, V)
        n_comp = min(self.n_components, max_comp)
        if n_comp < self.n_components:
            logger.warning(
                "Capping PLS components from %d to %d (T=%d, D=%d, V=%d)",
                self.n_components,
                n_comp,
                T,
                D,
                V,
            )

        log_matlab_note(
            logger,
            "plsregress(X, Y, 20)",
            f"Fitting PLS: X({T},{D}) → Y({T},{V}), n_components={n_comp}",
        )

        # Use SIMPLS implementation to match MATLAB more closely
        try:
            betas_no_intercept, intercept = _simpls(X, Y, n_comp)
            self._intercept = intercept
            self._betas = np.vstack([intercept.reshape(1, -1), betas_no_intercept])
            logger.info("PLS fit via SIMPLS: betas shape = %s", self._betas.shape)
        except Exception as e:
            logger.warning("SIMPLS failed (%s), falling back to sklearn PLSRegression", e)
            self._fit_sklearn(X, Y, n_comp)

    def _fit_sklearn(self, X: np.ndarray, Y: np.ndarray, n_comp: int) -> None:
        """Fallback: fit using sklearn PLSRegression.

        sklearn PLSRegression uses NIPALS, not SIMPLS. Results will be similar
        but not identical to MATLAB.
        """
        model = PLSRegression(
            n_components=n_comp,
            scale=self.standardize_X or self.standardize_Y,
            max_iter=500,
            tol=1e-06,
        )
        model.fit(X, Y)

        # Reconstruct betas in MATLAB format: (D+1, V)
        # sklearn stores coef_ as (V, D) or (n_targets, n_features)
        # and intercept_ as (V,)
        coef = model.coef_  # (V, D) in sklearn
        intercept = model.intercept_  # (V,)

        if coef.shape[0] == Y.shape[1]:
            # coef is (V, D) → transpose to (D, V)
            coef = coef.T

        self._intercept = intercept
        # Stack intercept as first row: (D+1, V)
        self._betas = np.vstack([intercept.reshape(1, -1), coef])
        logger.info(
            "PLS fit via sklearn (NIPALS): betas shape = %s", self._betas.shape
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict brain activity from features.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (N, D).

        Returns
        -------
        np.ndarray
            Predicted activity, shape (N, V).

        Notes
        -----
        MATLAB: yhat = [ones(N,1) X] * b
        """
        if self._betas is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
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


def _simpls(X: np.ndarray, Y: np.ndarray, n_components: int):
    """SIMPLS algorithm matching MATLAB plsregress internals.

    Parameters
    ----------
    X : np.ndarray, shape (T, D)
        Predictor matrix.
    Y : np.ndarray, shape (T, V)
        Response matrix.
    n_components : int
        Number of PLS components.

    Returns
    -------
    betas : np.ndarray, shape (D, V)
        Regression coefficients (without intercept).
    intercept : np.ndarray, shape (V,)
        Intercept vector.

    Notes
    -----
    SIMPLS (Statistically Inspired Modification of PLS) by de Jong (1993).
    This is the algorithm used internally by MATLAB's plsregress.

    The algorithm:
    1. Center X and Y
    2. Iteratively find weight vectors that maximize covariance
    3. Deflate the cross-product matrix (not X or Y directly)
    4. Compute regression coefficients from the PLS decomposition

    Reference: de Jong, S. (1993). SIMPLS: an alternative approach to partial
    least squares regression. Chemometrics and Intelligent Laboratory Systems.
    """
    T, D = X.shape
    V = Y.shape[1]

    # Center X and Y (MATLAB plsregress does this internally)
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X0 = X - X_mean
    Y0 = Y - Y_mean

    # Cross-product matrix
    S = X0.T @ Y0  # (D, V)

    # Preallocate
    W = np.zeros((D, n_components))   # X weights
    P = np.zeros((D, n_components))   # X loadings
    Q = np.zeros((V, n_components))   # Y loadings
    T_scores = np.zeros((T, n_components))  # X scores
    U = np.zeros((T, n_components))   # Y scores
    V_orth = np.zeros((D, n_components))  # orthogonal projectors

    for a in range(n_components):
        # SVD of cross-product to get dominant direction
        if V == 1:
            # Univariate Y: weight is just S normalized
            q = np.array([[1.0]])
            w = S.ravel()
            w = w / np.linalg.norm(w)
        else:
            # Dominant singular vector of S
            U_svd, s_svd, Vt_svd = np.linalg.svd(S, full_matrices=False)
            w = U_svd[:, 0]
            q_vec = Vt_svd[0, :]

        # X scores
        t = X0 @ w
        t_norm_sq = t @ t
        if t_norm_sq < 1e-14:
            logger.warning("PLS component %d has near-zero variance; stopping early", a + 1)
            W = W[:, :a]
            P = P[:, :a]
            Q = Q[:, :a]
            T_scores = T_scores[:, :a]
            break

        # X and Y loadings
        p = X0.T @ t / t_norm_sq
        q_vec = Y0.T @ t / t_norm_sq

        # Store
        W[:, a] = w
        P[:, a] = p
        Q[:, a] = q_vec
        T_scores[:, a] = t

        # Deflate S (SIMPLS deflation of cross-product, not X/Y)
        v = p.copy()
        # Orthogonalize v against previous basis
        if a > 0:
            v = v - V_orth[:, :a] @ (V_orth[:, :a].T @ p)
        v = v / np.linalg.norm(v)
        V_orth[:, a] = v

        S = S - v.reshape(-1, 1) @ (v.reshape(1, -1) @ S)

    # Compute regression coefficients
    # B = W * (P' * W)^{-1} * Q'
    n_used = W.shape[1]
    PW = P[:, :n_used].T @ W[:, :n_used]  # (n_comp, n_comp)
    try:
        B = W[:, :n_used] @ np.linalg.solve(PW, Q[:, :n_used].T)  # (D, V)
    except np.linalg.LinAlgError:
        B = W[:, :n_used] @ np.linalg.lstsq(PW, Q[:, :n_used].T, rcond=None)[0]

    # Intercept: Y_mean - X_mean @ B
    intercept = Y_mean - X_mean @ B

    return B, intercept
