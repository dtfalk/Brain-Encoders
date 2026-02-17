"""
Statistical tests (one-sample t-test, paired t-test) for encoding model analyses.

This module corresponds to AMOD script(s):
  - perform_ttest_IAPS_OASIS_amygdala.m:
      [h, p, ci, st] = ttest(squeeze(avg_betas))
  - perform_pairwise_ttest_IAPS_OASIS_amygdala.m:
      [h, p, ci, st] = ttest(squeeze(mean(betas)) - squeeze(mean(betas_vc)))
  - perform_ttest_artificial_stim_subregion.m:
      [h, p, ci, st] = ttest(betas)
  - make_parametric_map_amygdala.m:
      [h, p, ci, stats] = ttest(atanh_matrix);
Key matched choices:
  - MATLAB ttest() performs one-sample t-test (H0: mean = 0)
  - We use scipy.stats.ttest_1samp for exact match
  - MATLAB ttest(A - B) for paired comparison
  - We use scipy.stats.ttest_rel for paired t-test
  - FDR correction: MATLAB's FDR(p, .05); we use statsmodels fdrcorrection
Assumptions / deviations:
  - MATLAB ttest returns [h, p, ci, stats]; we return a dict with all fields
  - FDR implementation may differ slightly between MATLAB and statsmodels
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats

from amod_encoder.utils.logging import get_logger, log_matlab_note

logger = get_logger(__name__)


def one_sample_ttest(
    data: np.ndarray,
    alpha: float = 0.05,
    fdr_correct: bool = False,
) -> dict:
    """One-sample t-test (H0: population mean = 0) matching MATLAB ttest.

    Parameters
    ----------
    data : np.ndarray
        If 1D (N,): single test over N subjects.
        If 2D (N, P): N subjects, P variables — test each column.
    alpha : float
        Significance level.
    fdr_correct : bool
        Apply FDR correction. Matches MATLAB FDR(p, .05).

    Returns
    -------
    dict with keys:
        'h': np.ndarray — reject null (boolean)
        'p': np.ndarray — p-values
        'ci': tuple of np.ndarray — confidence interval (lower, upper)
        't_stat': np.ndarray — t-statistics
        'df': int — degrees of freedom
        'mean': np.ndarray — sample means
        'se': np.ndarray — standard errors

    Notes
    -----
    MATLAB: [h, p, ci, st] = ttest(squeeze(avg_betas))
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    N, P = data.shape

    log_matlab_note(
        logger,
        "perform_ttest_IAPS_OASIS_amygdala.m",
        f"One-sample t-test: N={N}, P={P} variables",
    )

    t_stat, p_values = stats.ttest_1samp(data, 0, axis=0, nan_policy="omit")

    # Confidence intervals
    df = N - 1
    se = stats.sem(data, axis=0, nan_policy="omit")
    mean_vals = np.nanmean(data, axis=0)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_low = mean_vals - t_crit * se
    ci_high = mean_vals + t_crit * se

    h = p_values < alpha

    if fdr_correct:
        from statsmodels.stats.multitest import fdrcorrection

        reject, p_corrected = fdrcorrection(p_values, alpha=alpha)
        h = reject
        p_values = p_corrected
        logger.info("FDR correction applied: %d/%d significant", int(h.sum()), P)

    result = {
        "h": h,
        "p": p_values,
        "ci": (ci_low, ci_high),
        "t_stat": t_stat,
        "df": df,
        "mean": mean_vals,
        "se": se,
    }

    # Log summary
    if P <= 20:
        for i in range(P):
            sig_str = "*" if h[i] else ""
            logger.info(
                "  var %d: t(%d)=%.3f, p=%.4f, mean=%.4f %s",
                i + 1,
                df,
                t_stat[i],
                p_values[i],
                mean_vals[i],
                sig_str,
            )

    return result


def paired_ttest(
    data_a: np.ndarray,
    data_b: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Paired t-test (H0: mean(A - B) = 0) matching MATLAB paired ttest.

    Parameters
    ----------
    data_a : np.ndarray, shape (N,) or (N, P)
        First condition data.
    data_b : np.ndarray, shape (N,) or (N, P)
        Second condition data.
    alpha : float
        Significance level.

    Returns
    -------
    dict
        Same keys as one_sample_ttest.

    Notes
    -----
    MATLAB: [h, p, ci, st] = ttest(squeeze(mean(betas)) - squeeze(mean(betas_vc)))
    This is equivalent to a paired t-test.
    """
    log_matlab_note(
        logger,
        "perform_pairwise_ttest_IAPS_OASIS_amygdala.m",
        "Paired t-test: A vs B",
    )

    diff = data_a - data_b
    return one_sample_ttest(diff, alpha=alpha)


def voxelwise_ttest_with_fdr(
    atanh_matrix: np.ndarray,
    q: float = 0.05,
) -> dict:
    """Voxelwise one-sample t-test with FDR correction.

    Parameters
    ----------
    atanh_matrix : np.ndarray, shape (N_subjects, N_voxels)
        Fisher Z-transformed correlation matrix.
    q : float
        FDR threshold. MATLAB: FDR(p, .05).

    Returns
    -------
    dict with keys:
        't_stats': np.ndarray — t-statistics per voxel
        'p_values': np.ndarray — uncorrected p-values
        'fdr_threshold': float — FDR-corrected p-value threshold
        'significant_fdr': np.ndarray — boolean mask
        'significant_unc': np.ndarray — uncorrected significance at q

    Notes
    -----
    MATLAB:
        [h, p, ci, stats] = ttest(atanh_matrix);
        th_stats_object.dat(~(p < FDR(p, .05))) = NaN;
    """
    log_matlab_note(
        logger,
        "make_parametric_map_amygdala.m",
        f"Voxelwise t-test with FDR q={q}",
    )

    # Handle NaN columns (voxels with no data)
    valid = ~np.all(np.isnan(atanh_matrix), axis=0)

    t_stats = np.full(atanh_matrix.shape[1], np.nan)
    p_values = np.full(atanh_matrix.shape[1], np.nan)

    if valid.any():
        t_stats[valid], p_values[valid] = stats.ttest_1samp(
            atanh_matrix[:, valid], 0, axis=0, nan_policy="omit"
        )

    # FDR correction on valid p-values
    fdr_threshold = np.nan
    significant_fdr = np.zeros(atanh_matrix.shape[1], dtype=bool)

    valid_p = p_values[valid & ~np.isnan(p_values)]
    if len(valid_p) > 0:
        from statsmodels.stats.multitest import fdrcorrection

        reject, _ = fdrcorrection(valid_p, alpha=q)
        # Find the FDR threshold (largest p that is still rejected)
        if reject.any():
            fdr_threshold = float(valid_p[reject].max())
            significant_fdr[valid & ~np.isnan(p_values)] = p_values[
                valid & ~np.isnan(p_values)
            ] <= fdr_threshold

    result = {
        "t_stats": t_stats,
        "p_values": p_values,
        "fdr_threshold": fdr_threshold,
        "significant_fdr": significant_fdr,
        "significant_unc": p_values < q,
        "n_significant_fdr": int(significant_fdr.sum()),
        "n_significant_unc": int(np.nansum(p_values < q)),
    }

    logger.info(
        "Voxelwise t-test: %d voxels significant (FDR q=%.2f), %d (uncorrected)",
        result["n_significant_fdr"],
        q,
        result["n_significant_unc"],
    )

    return result
