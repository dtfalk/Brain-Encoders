"""
Artificial Stimulus Prediction
==============================

Predicts and decodes activations for DNN-generated artificial stimuli.

Core Algorithm::

    1. Load fc7 activations for each generated image
    2. Predict: enc_pred = [1, fc7] @ betas
    3. Decode: PLS classifier with 7 components, leave-one-subject-out CV
    4. Accuracy = mean(pred_category == true_category)
    5. t-SNE visualisation of predicted activation space

Design Principles:
    - Same prediction formula as IAPS/OASIS: ``[ones(N,1), acts] * betas``
    - Decoding PLS uses 7 components (different from encoding’s 20)
    - Random subregion decoding: 1000 iterations with random voxel splits
    - Assumes fc7 features are pre-computed (requires emonet-pytorch)

MATLAB Correspondence:
    - predict_activations_artificial_stim_amygdala.m
    - predict_activations_artificial_stim_subregion.m
    - decode_activation_targets_artificial_stim.m
    - decode_activation_targets_random_subregions.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

from amod_encoder.models.pls import PLSEncodingModel
from amod_encoder.utils.logging import get_logger, log_matlab_note

logger = get_logger(__name__)


def predict_artificial_stim(
    image_features: np.ndarray,
    betas_per_subject: dict[str, np.ndarray],
    subject_ids: list[str],
    target_subjects: Optional[list[str]] = None,
    target_rois: Optional[list[str]] = None,
) -> dict:
    """Predict activations for artificial stimuli.

    Parameters
    ----------
    image_features : np.ndarray, shape (N_images, D)
        fc7 activations for artificial stimuli.
    betas_per_subject : dict
        subject_id → betas (D+1, V).
    subject_ids : list[str]
        Encoding model subject IDs.
    target_subjects : list[str] or None
        Target subject for each image (from filename parsing).
    target_rois : list[str] or None
        Target ROI for each image.

    Returns
    -------
    dict with:
        'enc_prediction': np.ndarray (N_images * N_subjects, V)
        'enc_prediction_avg': np.ndarray (leave-one-out)
        'subject_labels': list
        'target_subjects': list or None
        'target_rois': list or None
    """
    log_matlab_note(
        logger,
        "predict_activations_artificial_stim_amygdala.m",
        f"Predicting for {len(subject_ids)} subjects × "
        f"{image_features.shape[0]} artificial stimuli",
    )

    N_images, D = image_features.shape
    N_subjects = len(subject_ids)

    first_betas = next(iter(betas_per_subject.values()))
    V = first_betas.shape[1] if first_betas.ndim > 1 else 1

    betas_stack = np.stack([betas_per_subject[s] for s in subject_ids], axis=0)

    all_preds = []
    all_preds_avg = []
    subj_labels = []

    for f in range(N_images):
        acts_aug = np.column_stack([np.ones(1), image_features[f:f+1, :]])

        for s_idx, s_id in enumerate(subject_ids):
            pred = (acts_aug @ betas_per_subject[s_id]).ravel()
            all_preds.append(pred)

            mask = np.ones(N_subjects, dtype=bool)
            mask[s_idx] = False
            avg_b = betas_stack[mask].mean(axis=0)
            all_preds_avg.append((acts_aug @ avg_b).ravel())

            subj_labels.append(s_id)

    return {
        "enc_prediction": np.array(all_preds),
        "enc_prediction_avg": np.array(all_preds_avg),
        "subject_labels": subj_labels,
        "target_subjects": target_subjects,
        "target_rois": target_rois,
    }


def decode_activation_targets(
    enc_prediction: np.ndarray,
    roi_labels: np.ndarray,
    subject_labels: np.ndarray,
    n_pls_components: int = 7,
) -> dict:
    """Decode ROI targets from predicted activations using PLS + LOO-subject CV.

    Parameters
    ----------
    enc_prediction : np.ndarray, shape (N, V)
        Predicted activations for all images × subjects.
    roi_labels : np.ndarray, shape (N,)
        True ROI target labels (categorical).
    subject_labels : np.ndarray, shape (N,)
        Subject labels for leave-one-subject-out CV.
    n_pls_components : int
        PLS components for decoder. MATLAB uses 7.

    Returns
    -------
    dict with:
        'accuracy': float
        'pred_cat': np.ndarray — predicted category indices
        'true_cat': np.ndarray — true category indices
        'yhat': np.ndarray — raw predictions

    Notes
    -----
    MATLAB:
        Y = zscore(condf2indic(categorical(roi_full)));
        kinds = double(categorical(subj_full));
        for k=1:max(kinds)
            [~,~,~,~,b] = plsregress(enc_prediction(train,:), Y(train,:), 7);
            yhat(kinds==k,:) = [ones(N_test,1) enc_prediction(kinds==k,:)] * b;
        end
        [~, pred_cat] = max(yhat, [], 2);
    """
    log_matlab_note(
        logger,
        "decode_activation_targets_artificial_stim.m",
        f"Decoding {len(np.unique(roi_labels))} ROI targets, "
        f"LOO-subject CV, PLS n_comp={n_pls_components}",
    )

    # Create indicator matrix Y (one-hot, z-scored)
    unique_rois = np.unique(roi_labels)
    n_classes = len(unique_rois)
    roi_to_idx = {r: i for i, r in enumerate(unique_rois)}

    Y_indicator = np.zeros((len(roi_labels), n_classes))
    for i, r in enumerate(roi_labels):
        Y_indicator[i, roi_to_idx[r]] = 1.0

    # Z-score indicator columns
    Y_z = stats.zscore(Y_indicator, axis=0, ddof=0)

    # Leave-one-subject-out CV
    unique_subjects = np.unique(subject_labels)
    yhat = np.zeros_like(Y_z)

    for s in unique_subjects:
        train_mask = subject_labels != s
        test_mask = ~train_mask

        model = PLSEncodingModel(n_components=min(n_pls_components, n_classes))
        model.fit(enc_prediction[train_mask].astype(np.float64), Y_z[train_mask])

        yhat[test_mask] = model.predict(enc_prediction[test_mask].astype(np.float64))

    pred_cat = np.argmax(yhat, axis=1)
    true_cat = np.argmax(Y_indicator, axis=1)
    accuracy = float(np.mean(pred_cat == true_cat))

    logger.info("Decoding accuracy: %.2f%% (chance=%.2f%%)", accuracy * 100, 100 / n_classes)

    return {
        "accuracy": accuracy,
        "pred_cat": pred_cat,
        "true_cat": true_cat,
        "yhat": yhat,
        "roi_names": list(unique_rois),
    }
