"""
PLS Encoding Model
==================

Partial Least Squares regression matching MATLAB ``plsregress``.

Core Algorithm (SIMPLS)::

    1. Centre X and Y
    2. For each component k = 1 .. n_components:
       a. r = dominant left singular vector of XᵀY
       b. t = X r           (scores)
       c. p = Xᵀt / (tᵀt)  (X loadings)
       d. q = Yᵀt / (tᵀt)  (Y loadings)
       e. Deflate XᵀY by removing component
    3. Betas = R (TᵀT)⁻¹ Tᵀ Y  (with intercept prepended)

Design Principles:
    - Custom SIMPLS is the primary solver (matches MATLAB ``plsregress``)
    - sklearn ``PLSRegression`` (NIPALS) is the fallback if SIMPLS fails
    - ``n_components = 20`` matches the paper; capped at min(D, V, T)
    - Y is multivariate: all voxels fitted simultaneously
    - Betas include intercept in row 0: shape ``(D+1, V)``

MATLAB Correspondence:
    - develop_encoding_models_amygdala.m → ``plsregress(X, Y', 20)``
    - develop_encoding_models_subregions.m → same call, subregion masks
    - decode_activation_targets_artificial_stim.m → ``plsregress(..., 7)``
"""

from __future__ import annotations

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

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        device: Optional[str] = None,
    ) -> None:
        """Fit PLS model to features X and brain data Y.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (T, D).
        Y : np.ndarray
            Brain data, shape (T, V).
        device : str or None
            If set (e.g. 'cuda:0'), the SVD inside SIMPLS runs on that GPU.
            None → pure numpy (default, MATLAB-faithful).

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
            betas_no_intercept, intercept = _simpls(
                X, Y, n_comp,
                standardize_X=self.standardize_X,
                standardize_Y=self.standardize_Y,
                device=device,
            )
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


def _simpls(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    standardize_X: bool = False,
    standardize_Y: bool = False,
    device: Optional[str] = None,
):
    """SIMPLS algorithm matching MATLAB plsregress internals.

    Parameters
    ----------
    X : np.ndarray, shape (T, D)
        Predictor matrix.
    Y : np.ndarray, shape (T, V)
        Response matrix.
    n_components : int
        Number of PLS components.
    standardize_X : bool
        If True, z-score X columns after centering. MATLAB does NOT do this.
    standardize_Y : bool
        If True, z-score Y columns after centering. MATLAB does NOT do this.

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

    # Optional standardization (MATLAB does NOT do this — off by default)
    if standardize_X:
        X_std = X0.std(axis=0, ddof=1)
        X_std[X_std == 0] = 1.0
        X0 = X0 / X_std
    else:
        X_std = np.ones(D)

    if standardize_Y:
        Y_std = Y0.std(axis=0, ddof=1)
        Y_std[Y_std == 0] = 1.0
        Y0 = Y0 / Y_std
    else:
        Y_std = np.ones(V)

    # Cross-product matrix
    S = X0.T @ Y0  # (D, V)

    # Preallocate
    W = np.zeros((D, n_components))   # X weights
    P = np.zeros((D, n_components))   # X loadings
    Q = np.zeros((V, n_components))   # Y loadings
    T_scores = np.zeros((T, n_components))  # X scores
    V_orth = np.zeros((D, n_components))  # orthogonal projectors

    # -- Optional GPU SVD setup -------------------------------------------------
    # When device is set (e.g. 'cuda:0'), we move S to the GPU and use
    # torch.linalg.svd for the per-component SVD, which is ~3-8× faster than
    # numpy on the L40S for matrices of shape (D=4096, V=1000-5000).
    _use_gpu_svd = False
    if device is not None and device != "cpu":
        try:
            import torch as _torch_pls
            if _torch_pls.cuda.is_available():
                _S_gpu = _torch_pls.from_numpy(S.astype(np.float32)).to(device)
                _use_gpu_svd = True
                logger.debug("SIMPLS SVD running on %s", device)
        except Exception as _e:
            logger.debug("GPU SVD unavailable (%s), falling back to numpy", _e)
    # ---------------------------------------------------------------------------

    for a in range(n_components):
        # SVD of cross-product to get dominant direction
        if V == 1:
            # Univariate Y: weight is just S normalized
            w = S.ravel()
            w = w / np.linalg.norm(w)
        elif _use_gpu_svd:
            # GPU path: torch.linalg.svd is significantly faster for large (D,V)
            U_gpu, _, Vt_gpu = _torch_pls.linalg.svd(_S_gpu, full_matrices=False)
            w = U_gpu[:, 0].cpu().numpy().astype(np.float64)
            q_vec = Vt_gpu[0, :].cpu().numpy().astype(np.float64)
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
        if _use_gpu_svd:
            _S_gpu = _torch_pls.from_numpy(S.astype(np.float32)).to(device)

    # Compute regression coefficients
    # B = W * (P' * W)^{-1} * Q'
    n_used = W.shape[1]
    PW = P[:, :n_used].T @ W[:, :n_used]  # (n_comp, n_comp)
    try:
        B = W[:, :n_used] @ np.linalg.solve(PW, Q[:, :n_used].T)  # (D, V)
    except np.linalg.LinAlgError:
        B = W[:, :n_used] @ np.linalg.lstsq(PW, Q[:, :n_used].T, rcond=None)[0]

    # Undo standardization to get coefficients in original scale
    if standardize_X:
        B = B / X_std.reshape(-1, 1)
    if standardize_Y:
        B = B * Y_std.reshape(1, -1)

    # Intercept: Y_mean - X_mean @ B
    intercept = Y_mean - X_mean @ B

    return B, intercept
