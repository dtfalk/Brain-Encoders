"""
Evaluation Metrics
==================

Voxelwise correlation and Fisher’s Z for encoding model assessment.

Core Algorithm::

    For each CV fold k:
        1. Predict: Ŷ = [1, X_test] @ betas
        2. Correlate: diag_corr[k, v] = corr(Ŷ[:, v], Y_test[:, v])
    3. mean_diag_corr = mean(diag_corr, axis=0)   # per-voxel mean
    4. atanh_matrix = arctanh(mean_diag_corr)      # Fisher’s Z

Design Principles:
    - Primary metric is the *diagonal* of the full correlation matrix
    - Fisher’s Z (arctanh) stabilises variance before group-level t-tests
    - Clips extreme r values to [-0.9999, 0.9999] to avoid ±Inf

MATLAB Correspondence:
    - develop_encoding_models_amygdala.m → ``cross_validated_correlation()``
    - compile_matrices.m → ``fishers_z()``
"""

from __future__ import annotations


import numpy as np

from amod_encoder.utils.logging import get_logger, log_matlab_note

logger = get_logger(__name__)


def voxelwise_correlation(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """Compute voxelwise Pearson correlation between predicted and observed.

    Parameters
    ----------
    y_pred : np.ndarray, shape (T, V)
        Predicted brain data.
    y_true : np.ndarray, shape (T, V)
        Observed brain data.

    Returns
    -------
    np.ndarray, shape (V,)
        Pearson correlation for each voxel.

    Notes
    -----
    MATLAB equivalent:
        pred_obs_corr = corr(yhat, y_actual);
        diag_corr = diag(pred_obs_corr);

    We compute the diagonal directly for efficiency:
        r[v] = corr(y_pred[:, v], y_true[:, v])
    """
    assert y_pred.shape == y_true.shape, (
        f"Shape mismatch: pred={y_pred.shape}, true={y_true.shape}"
    )
    T, V = y_pred.shape

    # Center
    pred_c = y_pred - y_pred.mean(axis=0, keepdims=True)
    true_c = y_true - y_true.mean(axis=0, keepdims=True)

    # Compute correlation column-wise (vectorized)
    num = (pred_c * true_c).sum(axis=0)
    denom = np.sqrt((pred_c**2).sum(axis=0) * (true_c**2).sum(axis=0))

    # Avoid division by zero
    denom[denom == 0] = np.nan

    r = num / denom

    log_matlab_note(
        logger,
        "diag(corr(yhat, y_actual))",
        f"Voxelwise correlation: V={V}, mean_r={np.nanmean(r):.4f}, "
        f"median_r={np.nanmedian(r):.4f}",
    )

    return r


def fishers_z(r: np.ndarray) -> np.ndarray:
    """Apply Fisher's Z (arctanh) transformation to correlation values.

    Parameters
    ----------
    r : np.ndarray
        Correlation values, typically in [-1, 1].

    Returns
    -------
    np.ndarray
        Fisher's Z transformed values.

    Notes
    -----
    MATLAB: atanh_matrix = atanh(new_matrix)

    Handles edge cases:
    - r = 1.0 → Inf; we clip to 0.9999 before transform
    - r = NaN → NaN (preserved)
    """
    r_clipped = np.clip(r, -0.9999, 0.9999)
    z = np.arctanh(r_clipped)

    n_clipped = int(np.sum(np.abs(r) >= 0.9999))
    if n_clipped > 0:
        logger.warning(
            "Fisher's Z: %d values clipped to ±0.9999 before arctanh", n_clipped
        )

    return z


def cross_validated_correlation(
    X: np.ndarray,
    Y: np.ndarray,
    model_class,
    model_kwargs: dict,
    cv_splits,
) -> dict:
    """Run cross-validated encoding and compute voxelwise correlations.

    This reproduces the full MATLAB CV loop from develop_encoding_models_amygdala.m:
        kinds = crossvalind('k', N, 5);
        for k=1:5
            [~,~,~,~,beta_cv] = plsregress(X(kinds~=k,:), Y(kinds~=k,:), 20);
            yhat(kinds==k,:) = [ones(N_test,1) X(kinds==k,:)] * beta_cv;
            pred_obs_corr(:,:,k) = corr(yhat(kinds==k,:), Y(kinds==k,:));
            diag_corr(k,:) = diag(pred_obs_corr(:,:,k));
        end
        mean_diag_corr = mean(diag_corr);

    Parameters
    ----------
    X : np.ndarray, shape (T, D)
        Feature matrix.
    Y : np.ndarray, shape (T, V)
        Brain data matrix.
    model_class : type
        EncodingModel subclass (e.g., PLSEncodingModel).
    model_kwargs : dict
        Keyword arguments for model constructor.
    cv_splits : generator
        Generator yielding (train_idx, test_idx) tuples.

    Returns
    -------
    dict with keys:
        'mean_diag_corr': np.ndarray (V,) — mean correlation per voxel across folds
        'diag_corr': np.ndarray (K, V) — per-fold correlation per voxel
        'yhat': np.ndarray (T, V) — full predicted matrix (combined across folds)
        'mean_corr_scalar': float — grand mean of mean_diag_corr
    """
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    T, V = Y.shape
    yhat = np.zeros_like(Y)
    fold_corrs = []

    # Materialize the generator to a list so we can count and iterate
    cv_splits_list = list(cv_splits) if not isinstance(cv_splits, list) else cv_splits

    log_matlab_note(
        logger,
        "develop_encoding_models_amygdala.m",
        f"Running {len(cv_splits_list)}-fold CV "
        f"on X({T},{X.shape[1]}) → Y({T},{V})",
    )

    for fold_num, (train_idx, test_idx) in enumerate(cv_splits_list):
        logger.info("CV fold %d: train=%d, test=%d", fold_num + 1, len(train_idx), len(test_idx))

        # Fit on training data
        model = model_class(**model_kwargs)
        model.fit(X[train_idx], Y[train_idx])

        # Predict on test data
        yhat[test_idx] = model.predict(X[test_idx])

        # Per-fold voxelwise correlation
        fold_r = voxelwise_correlation(yhat[test_idx], Y[test_idx])
        fold_corrs.append(fold_r)

    diag_corr = np.array(fold_corrs)  # (K, V)
    mean_diag_corr = np.nanmean(diag_corr, axis=0)  # (V,)

    result = {
        "mean_diag_corr": mean_diag_corr,
        "diag_corr": diag_corr,
        "yhat": yhat,
        "mean_corr_scalar": float(np.nanmean(mean_diag_corr)),
    }

    logger.info(
        "CV complete: mean voxelwise r = %.4f ± %.4f",
        result["mean_corr_scalar"],
        float(np.nanstd(mean_diag_corr)),
    )

    return result
