"""
IAPS / OASIS Prediction
=======================

Predicts amygdala activations for standardised affective images using
fitted encoding model betas.

Core Algorithm::

    For each image i, subject s:
        enc_pred[i] = [1, fc7[i, :]] @ betas[s]

    Leave-one-subject-out average prediction::

        enc_pred_avg[i] = [1, fc7[i, :]] @ mean(betas[~s])

    Correlate predictions with valence / arousal ratings.

Design Principles:
    - Betas loaded from saved ``(D+1, V)`` arrays per subject per ROI
    - Ratings come from CSV (``IAPS_data_amygdala_z.csv``, etc.)
    - Prediction formula matches MATLAB: ``[ones(N,1), acts] * betas``
    - Supports binning by arousal / valence deciles for Friedman test

MATLAB Correspondence:
    - predict_activations_IAPS_OASIS.m → ``predict_iaps_oasis()``
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from amod_encoder.utils.logging import get_logger, log_matlab_note

logger = get_logger(__name__)


def predict_activations(
    acts: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """Predict activations for a set of images given fc7 features and model betas.

    Parameters
    ----------
    acts : np.ndarray, shape (N_images, D)
        fc7 activations for each image.
    betas : np.ndarray, shape (D+1, V)
        Model betas with intercept in first row.

    Returns
    -------
    np.ndarray, shape (N_images, V)
        Predicted activations.

    Notes
    -----
    MATLAB: enc_prediction = [ones(N,1) acts] * betas
    """
    N = acts.shape[0]
    X_aug = np.column_stack([np.ones(N), acts])  # (N, D+1)
    return X_aug @ betas  # (N, V)


def predict_iaps_oasis(
    image_features: np.ndarray,
    betas_per_subject: dict[str, np.ndarray],
    valence: np.ndarray,
    arousal: np.ndarray,
    subject_ids: list[str],
) -> dict:
    """Predict IAPS/OASIS activations for all subjects.

    Parameters
    ----------
    image_features : np.ndarray, shape (N_images, D)
        fc7 activations for IAPS or OASIS images.
    betas_per_subject : dict
        Maps subject_id → betas array of shape (D+1, V) or (D+1,) for mean-beta.
    valence : np.ndarray, shape (N_images,)
        Valence ratings.
    arousal : np.ndarray, shape (N_images,)
        Arousal ratings.
    subject_ids : list[str]
        Subject IDs to process.

    Returns
    -------
    dict with keys:
        'enc_prediction': np.ndarray — (N_images * N_subjects, V)
        'enc_prediction_avg': np.ndarray — leave-one-out average predictions
        'valence_corr': np.ndarray — (N_subjects, V) correlations with valence
        'arousal_corr': np.ndarray — (N_subjects, V) correlations with arousal
        'subject_labels': list[str] — subject label for each row
        'valence_full': np.ndarray — expanded valence
        'arousal_full': np.ndarray — expanded arousal

    Notes
    -----
    MATLAB loop structure:
        for f = num_run (each image)
            for s = 1:length(subject)
                enc_prediction(c,:) = squeeze(acts(f,:)) * squeeze(betas(s,:,:));
                enc_prediction_avg(c,:) = squeeze(acts(f,:)) * squeeze(mean(betas(subj_inds~=s,:,:)));
            end
        end
    """
    log_matlab_note(
        logger,
        "predict_activations_IAPS_OASIS.m",
        f"Predicting for {len(subject_ids)} subjects × "
        f"{image_features.shape[0]} images",
    )

    N_images, D = image_features.shape
    N_subjects = len(subject_ids)

    # Determine V (number of voxels/ROIs)
    first_betas = next(iter(betas_per_subject.values()))
    if first_betas.ndim == 1:
        V = 1
    else:
        V = first_betas.shape[1]

    all_predictions = []
    all_predictions_avg = []
    subj_labels = []
    val_full = []
    arous_full = []

    # Stack all betas for leave-one-out averaging
    betas_stack = np.stack(
        [betas_per_subject[s] for s in subject_ids], axis=0
    )  # (N_subjects, D+1, V) or (N_subjects, D+1)

    for f in range(N_images):
        acts_f = image_features[f : f + 1, :]  # (1, D)
        acts_aug = np.column_stack([np.ones(1), acts_f])  # (1, D+1)

        for s_idx, s_id in enumerate(subject_ids):
            b = betas_per_subject[s_id]
            pred = (acts_aug @ b).ravel()  # (V,)
            all_predictions.append(pred)

            # Leave-one-out average
            mask = np.ones(N_subjects, dtype=bool)
            mask[s_idx] = False
            avg_betas = betas_stack[mask].mean(axis=0)
            pred_avg = (acts_aug @ avg_betas).ravel()
            all_predictions_avg.append(pred_avg)

            subj_labels.append(s_id)
            val_full.append(valence[f])
            arous_full.append(arousal[f])

    enc_prediction = np.array(all_predictions)  # (N_images * N_subjects, V)
    enc_prediction_avg = np.array(all_predictions_avg)
    val_full = np.array(val_full)
    arous_full = np.array(arous_full)

    # Per-subject correlations with valence and arousal
    valence_corr = np.zeros((N_subjects, V))
    arousal_corr = np.zeros((N_subjects, V))

    for s_idx, s_id in enumerate(subject_ids):
        s_mask = np.array(subj_labels) == s_id
        pred_s = enc_prediction[s_mask]
        val_s = val_full[s_mask]
        arous_s = arous_full[s_mask]

        for v in range(V):
            if np.std(pred_s[:, v]) > 0:
                valence_corr[s_idx, v] = stats.pearsonr(pred_s[:, v], val_s)[0]
                arousal_corr[s_idx, v] = stats.pearsonr(pred_s[:, v], arous_s)[0]

    logger.info(
        "Predictions complete: %d total rows, %d ROIs/voxels",
        enc_prediction.shape[0],
        V,
    )

    return {
        "enc_prediction": enc_prediction,
        "enc_prediction_avg": enc_prediction_avg,
        "valence_corr": valence_corr,
        "arousal_corr": arousal_corr,
        "subject_labels": subj_labels,
        "valence_full": val_full,
        "arousal_full": arous_full,
    }


def load_ratings_csv(csv_path: Path) -> pd.DataFrame:
    """Load IAPS/OASIS ratings CSV (e.g., IAPS_data_amygdala_z.csv).

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with columns like subject, val_full_z, arous_full_z,
        amy_enc_pred, vc_enc_pred, median_red_z, etc.

    Notes
    -----
    These CSVs are the pre-compiled data tables used in MATLAB for regression.
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded ratings CSV: %s, shape=%s, columns=%s", csv_path.name, df.shape, list(df.columns))
    return df
