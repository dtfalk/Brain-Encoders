"""
Cross-Validation Splits
=======================

Generates train/test index partitions for encoding model evaluation.

Design Principles:
    - ``kfold`` with shuffle matches MATLAB ``crossvalind('k', N, 5)``
    - Deterministic seed for reproducibility (MATLAB has no explicit seed)
    - ``block`` CV preserves temporal contiguity (useful for fMRI data)

MATLAB Correspondence:
    - develop_encoding_models_amygdala.m → ``generate_cv_splits(scheme='kfold')``
"""

from __future__ import annotations

from typing import Generator, Literal

import numpy as np
from sklearn.model_selection import KFold

from amod_encoder.utils.logging import get_logger, log_matlab_note

logger = get_logger(__name__)


def generate_cv_splits(
    n_samples: int,
    scheme: Literal["kfold", "block"] = "kfold",
    n_folds: int = 5,
    seed: int = 42,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Generate train/test index arrays for cross-validation.

    Parameters
    ----------
    n_samples : int
        Total number of samples (timepoints).
    scheme : str
        'kfold': random k-fold (matches MATLAB crossvalind('k', N, 5))
        'block': contiguous temporal blocks
    n_folds : int
        Number of folds. MATLAB uses 5.
    seed : int
        Random seed. Set to -1 for unseeded (MATLAB-like behavior).

    Yields
    ------
    train_idx : np.ndarray
        Training sample indices.
    test_idx : np.ndarray
        Testing sample indices.

    Notes
    -----
    MATLAB:
        kinds = crossvalind('k', length(masked_dat.dat), 5);
        for k=1:5
            train = kinds ~= k;
            test  = kinds == k;
        end
    """
    log_matlab_note(
        logger,
        "crossvalind('k', N, 5)",
        f"Generating {n_folds}-fold {scheme} CV splits for {n_samples} samples",
    )

    if scheme == "kfold":
        rng = None if seed == -1 else seed
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=rng)
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_samples))):
            logger.debug(
                "Fold %d/%d: train=%d, test=%d",
                fold_idx + 1,
                n_folds,
                len(train_idx),
                len(test_idx),
            )
            yield train_idx, test_idx

    elif scheme == "block":
        # Contiguous temporal blocks
        indices = np.arange(n_samples)
        fold_sizes = np.full(n_folds, n_samples // n_folds)
        fold_sizes[: n_samples % n_folds] += 1
        current = 0
        folds = []
        for size in fold_sizes:
            folds.append(indices[current : current + size])
            current += size

        for k in range(n_folds):
            test_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            logger.debug(
                "Block fold %d/%d: train=%d, test=%d",
                k + 1,
                n_folds,
                len(train_idx),
                len(test_idx),
            )
            yield train_idx, test_idx

    else:
        raise ValueError(f"Unknown CV scheme: {scheme}")


def get_fold_assignments(
    n_samples: int,
    scheme: Literal["kfold", "block"] = "kfold",
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Return a fold-assignment vector like MATLAB's crossvalind.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    scheme : str
        CV scheme.
    n_folds : int
        Number of folds.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        1D integer array of shape (n_samples,) with values 1..n_folds.
        Matches MATLAB's `kinds` variable.
    """
    kinds = np.zeros(n_samples, dtype=int)
    for fold_num, (_, test_idx) in enumerate(
        generate_cv_splits(n_samples, scheme, n_folds, seed)
    ):
        kinds[test_idx] = fold_num + 1  # 1-indexed like MATLAB
    return kinds
