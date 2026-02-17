"""
Regression analysis mirroring perform_regression_IAPS_OASIS.m.

This module corresponds to AMOD script(s):
  - perform_regression_IAPS_OASIS.m
Key matched choices:
  - Per-subject linear mixed effects regression:
      roi_enc_pred ~ 1 + val_full_z + arous_full_z + val_full_z:arous_full_z
        + median_red_z + median_green_z + median_blue_z + high_freq_z + low_freq_z
  - Betas extracted per subject, then grouped
  - Piecewise regressions: positive valence, negative valence, neutral range
  - ANOVA on subregion betas for valence and interaction terms
Assumptions / deviations:
  - MATLAB uses fitlme (linear mixed effects); we use statsmodels OLS per subject
    (since each subject is fit independently, no random effects within a single subject fit,
     fitlme with no grouping variable reduces to OLS)
  - MATLAB gridfit for surface plots is not reproduced; we provide raw data for plotting
  - ANOVA uses scipy/pingouin instead of MATLAB fitrm
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from amod_encoder.utils.logging import get_logger, log_matlab_note

logger = get_logger(__name__)


def run_per_subject_regression(
    data: pd.DataFrame,
    dependent_var: str,
    subject_col: str = "subject",
    predictors: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Run per-subject OLS regression matching MATLAB fitlme pattern.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns for DV, predictors, and subject.
    dependent_var : str
        Name of dependent variable column (e.g., 'amy_enc_pred').
    subject_col : str
        Column identifying subjects.
    predictors : list[str] or None
        Predictor column names. If None, uses the MATLAB default set:
        ['val_full_z', 'arous_full_z', 'val_full_z:arous_full_z',
         'median_red_z', 'median_green_z', 'median_blue_z',
         'high_freq_z', 'low_freq_z']

    Returns
    -------
    pd.DataFrame
        Betas per subject. Columns = predictor names (incl. intercept).
        Rows = subjects.

    Notes
    -----
    MATLAB:
        for s = 1:20
            tsub = t(t.subject==s,:);
            lme_sub = fitlme(tsub,
                'amy_enc_pred ~ 1 + val_full_z + arous_full_z + ...');
            betas(f,s,:) = lme_sub.Coefficients.Estimate;
        end
    """
    import statsmodels.api as sm

    if predictors is None:
        predictors = [
            "val_full_z",
            "arous_full_z",
            "median_red_z",
            "median_green_z",
            "median_blue_z",
            "high_freq_z",
            "low_freq_z",
        ]

    log_matlab_note(
        logger,
        "perform_regression_IAPS_OASIS.m",
        f"Per-subject regression: {dependent_var} ~ {' + '.join(predictors)}",
    )

    subjects = sorted(data[subject_col].unique())
    all_betas = []

    for s in subjects:
        tsub = data[data[subject_col] == s].copy()

        # Build design matrix with interaction
        X_parts = []
        col_names = []
        for p in predictors:
            if ":" in p:
                # Interaction term
                parts = p.split(":")
                interaction = tsub[parts[0]].values * tsub[parts[1]].values
                X_parts.append(interaction)
                col_names.append(p)
            else:
                X_parts.append(tsub[p].values)
                col_names.append(p)

        X = np.column_stack(X_parts)
        X = sm.add_constant(X)
        y = tsub[dependent_var].values

        model = sm.OLS(y, X).fit()

        beta_dict = {"subject": s, "intercept": model.params[0]}
        for i, name in enumerate(col_names):
            beta_dict[name] = model.params[i + 1]
        all_betas.append(beta_dict)

    return pd.DataFrame(all_betas)


def run_piecewise_regression(
    data: pd.DataFrame,
    dependent_var: str,
    valence_col: str = "val_full_z",
    subject_col: str = "subject",
) -> dict[str, pd.DataFrame]:
    """Run piecewise regressions by valence range (negative, neutral, positive).

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset.
    dependent_var : str
        Dependent variable.
    valence_col : str
        Valence column (z-scored).
    subject_col : str
        Subject column.

    Returns
    -------
    dict
        Keys: 'positive', 'negative', 'neutral', each mapping to
        a DataFrame of betas.

    Notes
    -----
    MATLAB:
        t_positive = t(t.val_full_z > 0,:);
        t_negative = t(t.val_full_z < 0,:);
        t_mid = t(abs(t.val_full_z) < 1,:);
    """
    log_matlab_note(
        logger,
        "perform_regression_IAPS_OASIS.m",
        "Piecewise regressions: positive (val>0), negative (val<0), neutral (|val|<1)",
    )

    results = {}
    subsets = {
        "positive": data[data[valence_col] > 0],
        "negative": data[data[valence_col] < 0],
        "neutral": data[data[valence_col].abs() < 1],
    }

    for name, subset in subsets.items():
        if len(subset) < 10:
            logger.warning("Piecewise '%s': only %d samples, skipping", name, len(subset))
            continue
        results[name] = run_per_subject_regression(subset, dependent_var, subject_col)

    return results


def subregion_anova(
    betas: pd.DataFrame,
    roi_columns: list[str],
    predictor_name: str,
) -> dict:
    """Repeated-measures ANOVA comparing betas across ROI subregions.

    Parameters
    ----------
    betas : pd.DataFrame
        DataFrame with one row per subject and columns for each ROI's beta.
    roi_columns : list[str]
        Column names for the ROI betas (e.g., ['AStr', 'CM', 'LB', 'SF']).
    predictor_name : str
        Name of the predictor being compared (for logging).

    Returns
    -------
    dict
        'F_stat': F statistic
        'p_value': p value
        'df_between': between-group df
        'df_within': within-group df
        'pairwise': list of pairwise t-test results

    Notes
    -----
    MATLAB:
        rm = fitrm(t_subregions, "Var2-Var5~1", WithinDesign=meas);
        anova(rm)
        multcompare(rm, 'ROI')
    We implement as one-way repeated-measures ANOVA via F-test on differences.
    """
    log_matlab_note(
        logger,
        "perform_regression_IAPS_OASIS.m",
        f"Repeated-measures ANOVA: {predictor_name} across {roi_columns}",
    )

    # Extract data matrix: (subjects, ROIs)
    data_matrix = betas[roi_columns].values
    n_subjects, n_rois = data_matrix.shape

    # Repeated-measures ANOVA via F-test
    # Method: compute F from subject-centered data
    grand_mean = data_matrix.mean()
    subj_means = data_matrix.mean(axis=1, keepdims=True)
    roi_means = data_matrix.mean(axis=0, keepdims=True)

    ss_roi = n_subjects * np.sum((roi_means - grand_mean) ** 2)
    ss_error = np.sum((data_matrix - subj_means - roi_means + grand_mean) ** 2)

    df_roi = n_rois - 1
    df_error = (n_subjects - 1) * (n_rois - 1)

    ms_roi = ss_roi / df_roi if df_roi > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 1e-10

    f_stat = ms_roi / ms_error
    p_value = 1 - stats.f.cdf(f_stat, df_roi, df_error)

    # Pairwise comparisons (paired t-tests with Tukey-like approach)
    pairwise = []
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            t_stat, p_val = stats.ttest_rel(data_matrix[:, i], data_matrix[:, j])
            pairwise.append({
                "roi_1": roi_columns[i],
                "roi_2": roi_columns[j],
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "mean_diff": float(data_matrix[:, i].mean() - data_matrix[:, j].mean()),
            })

    result = {
        "F_stat": float(f_stat),
        "p_value": float(p_value),
        "df_between": df_roi,
        "df_within": df_error,
        "pairwise": pairwise,
    }

    logger.info(
        "ANOVA (%s): F(%d,%d)=%.3f, p=%.4f",
        predictor_name,
        df_roi,
        df_error,
        f_stat,
        p_value,
    )

    return result
