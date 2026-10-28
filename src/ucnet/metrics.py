"""
Calibration, sharpness, proper scoring rules, and statistical evaluation metrics for UCNet:
- z-score validation (standardized residuals)
- Empirical interval coverage (q in [0.5, 0.8, 0.9, 0.95])
- Probability Integral Transform (PIT) & Kolmogorov-Smirnov test
- Regression Calibration Error (Kuleshov et al., 2018)
- Continuous Ranked Probability Score (CRPS, exact Gaussian & mixture)
- Negative Log-Likelihood (NLL)
- Sharpness (mean predictive standard deviation)
- Non-parametric bootstrap confidence intervals and paired difference testing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats
from scipy.special import ndtr, ndtri


@dataclass
class CalibrationReport:
    """Summary of uncertainty calibration metrics."""

    nll: float
    crps: float
    sharpness: float
    z_mean: float
    z_var: float
    coverage_50: float
    coverage_80: float
    coverage_90: float
    coverage_95: float
    calibration_error: float
    ks_statistic: float
    ks_pvalue: float


def compute_z_score(y_true: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Computes standardized residuals: z_i = (y_i - mu_i) / std_i."""
    y = np.asarray(y_true).ravel()
    m = np.asarray(mu).ravel()
    s = np.asarray(std).ravel()
    return (y - m) / np.maximum(s, 1e-12)


def compute_coverage(
    y_true: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    q: float = 0.90,
) -> float:
    """
    Computes empirical coverage for central prediction interval of nominal coverage q:
    Interval: [mu - z_{(1+q)/2} * std, mu + z_{(1+q)/2} * std]
    """
    y = np.asarray(y_true).ravel()
    m = np.asarray(mu).ravel()
    s = np.asarray(std).ravel()

    # z-critical value: ndtri((1 + q) / 2)
    z_crit = float(ndtri((1.0 + q) / 2.0))
    inside = np.abs(y - m) <= z_crit * s
    return float(np.mean(inside))


def compute_pit_values(
    y_true: np.ndarray,
    cdf_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Computes Probability Integral Transform values: u_i = F_i(y_i).
    Under ideal calibration, u_i ~ Uniform(0, 1).
    """
    y = np.asarray(y_true).ravel()
    u = np.asarray(cdf_fn(y)).ravel()
    if u.shape != y.shape:
        raise ValueError(
            f"cdf_fn must return one CDF value per target (expected {y.size}, got {u.size}); "
            "a mismatch usually means the targets were broadcast against the predictions."
        )
    return np.clip(u, 1e-9, 1.0 - 1e-9)


def make_original_scale_cdf(
    cdf_standardized: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X_norm: np.ndarray,
    y_mean: float,
    y_scale: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Adapt a predictor's CDF defined on standardized targets, `cdf(y_std, X)`, into a function of
    targets in original units, as expected by `compute_pit_values` / `evaluate_calibration`.
    """

    def cdf_fn(y_orig: np.ndarray) -> np.ndarray:
        y_std = (np.asarray(y_orig, dtype=np.float64) - y_mean) / y_scale
        return np.asarray(cdf_standardized(y_std, X_norm)).ravel()

    return cdf_fn


def compute_gaussian_pit(y_true: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Fast PIT for single Gaussian predictions: u_i = Phi((y_i - mu_i) / std_i)."""
    z = compute_z_score(y_true, mu, std)
    return np.clip(ndtr(z), 1e-9, 1.0 - 1e-9)


def compute_ks_test(pit_values: np.ndarray) -> tuple[float, float]:
    """
    Kolmogorov-Smirnov test comparing empirical PIT values against Uniform(0, 1).
    Returns (ks_statistic, p_value).
    """
    res = stats.kstest(pit_values, "uniform")
    return float(res.statistic), float(res.pvalue)


def compute_calibration_error(
    pit_values: np.ndarray,
    num_bins: int = 100,
) -> float:
    """
    Computes Regression Calibration Error (Kuleshov et al., 2018):
    CalErr = (1 / K) * sum_{j=1}^K (p_j - hat{p}_j)^2
    where hat{p}_j is the empirical CDF of PIT values at nominal level p_j.
    """
    u = np.sort(np.asarray(pit_values).ravel())
    N = len(u)
    p_levels = np.linspace(0.01, 0.99, num_bins)

    # Empirical cumulative proportions: fraction of u <= p
    p_hat = np.searchsorted(u, p_levels, side="right") / N
    return float(np.mean((p_levels - p_hat) ** 2))


def compute_crps_gaussian(y_true: np.ndarray, mu: np.ndarray, std: np.ndarray) -> float:
    """
    Exact closed-form Continuous Ranked Probability Score (CRPS) for Gaussian distributions:
    CRPS(N(mu, sigma^2), y) = sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    where z = (y - mu) / sigma.
    """
    y = np.asarray(y_true).ravel()
    m = np.asarray(mu).ravel()
    s = np.asarray(std).ravel()

    z = (y - m) / np.maximum(s, 1e-12)
    phi_z = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (z ** 2))
    Phi_z = ndtr(z)

    crps_values = s * (z * (2.0 * Phi_z - 1.0) + 2.0 * phi_z - (1.0 / np.sqrt(np.pi)))
    return float(np.mean(crps_values))


def compute_nll(y_true: np.ndarray, mu: np.ndarray, std: np.ndarray) -> float:
    """Mean Gaussian Negative Log-Likelihood."""
    y = np.asarray(y_true).ravel()
    m = np.asarray(mu).ravel()
    s = np.maximum(np.asarray(std).ravel(), 1e-12)

    var = s ** 2
    nll_values = 0.5 * np.log(var) + 0.5 * ((y - m) ** 2) / var + 0.5 * np.log(2.0 * np.pi)
    return float(np.mean(nll_values))


def compute_sharpness(std: np.ndarray) -> float:
    """Sharpness metric: mean predictive standard deviation."""
    return float(np.mean(std))


def evaluate_calibration(
    y_true: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    cdf_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> CalibrationReport:
    """Computes comprehensive calibration evaluation metrics."""
    y = np.asarray(y_true).ravel()
    m = np.asarray(mu).ravel()
    s = np.asarray(std).ravel()

    z = compute_z_score(y, m, s)
    z_mean = float(np.mean(z))
    z_var = float(np.var(z))

    cov_50 = compute_coverage(y, m, s, q=0.50)
    cov_80 = compute_coverage(y, m, s, q=0.80)
    cov_90 = compute_coverage(y, m, s, q=0.90)
    cov_95 = compute_coverage(y, m, s, q=0.95)

    if cdf_fn is not None:
        pit = compute_pit_values(y, cdf_fn)
    else:
        pit = compute_gaussian_pit(y, m, s)

    ks_stat, ks_pval = compute_ks_test(pit)
    cal_err = compute_calibration_error(pit)

    nll = compute_nll(y, m, s)
    crps = compute_crps_gaussian(y, m, s)
    sharpness = compute_sharpness(s)

    return CalibrationReport(
        nll=nll,
        crps=crps,
        sharpness=sharpness,
        z_mean=z_mean,
        z_var=z_var,
        coverage_50=cov_50,
        coverage_80=cov_80,
        coverage_90=cov_90,
        coverage_95=cov_95,
        calibration_error=cal_err,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
    )


def bootstrap_metric_ci(
    y_true: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval:
    Returns (point_estimate, lower_percentile, upper_percentile).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).ravel()
    m = np.asarray(mu).ravel()
    s = np.asarray(std).ravel()
    N = len(y)

    point_est = metric_fn(y, m, s)

    boot_estimates = np.empty(num_bootstrap, dtype=np.float64)
    for b in range(num_bootstrap):
        idx = rng.integers(0, N, size=N)
        boot_estimates[b] = metric_fn(y[idx], m[idx], s[idx])

    alpha = 1.0 - confidence_level
    low = float(np.percentile(boot_estimates, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_estimates, 100.0 * (1.0 - alpha / 2.0)))
    return point_est, low, high


def bootstrap_paired_difference_ci(
    metric_values_a: Sequence[float],
    metric_values_b: Sequence[float],
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float, bool]:
    """
    Bootstrap confidence interval on paired model difference: Delta = Metric_A - Metric_B.
    Returns (mean_diff, lower_ci, upper_ci, is_statistically_significant).
    Significant if CI does not contain 0.
    """
    diffs = np.asarray(metric_values_a) - np.asarray(metric_values_b)
    N = len(diffs)
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(num_bootstrap, dtype=np.float64)
    for b in range(num_bootstrap):
        sample = rng.choice(diffs, size=N, replace=True)
        boot_diffs[b] = np.mean(sample)

    alpha = 1.0 - confidence_level
    low = float(np.percentile(boot_diffs, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_diffs, 100.0 * (1.0 - alpha / 2.0)))

    # Significant if 0 is outside [low, high]
    is_significant = (low > 0.0) or (high < 0.0)
    return mean_diff, low, high, is_significant


def hypothesis_verdict(
    advantage_per_seed: Sequence[float],
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float, bool]:
    """
    Pre-registered decision rule (docs/hypotheses.md section 3).
    `advantage_per_seed[k]` = Metric(M1)_k - Metric(M2)_k for a lower-is-better metric, so a positive
    value means M2 wins. Returns (mean, ci_low, ci_high, confirmed) where `confirmed` requires
    (a) M2 ahead in every seed AND (b) the bootstrap CI of the mean strictly above zero.
    A CI strictly below zero (M2 reliably worse) is NOT a confirmation.
    """
    diffs = np.asarray(advantage_per_seed, dtype=np.float64)
    mean_diff, low, high, _ = bootstrap_paired_difference_ci(
        diffs, np.zeros_like(diffs), num_bootstrap=num_bootstrap,
        confidence_level=confidence_level, seed=seed,
    )
    confirmed = bool(np.all(diffs > 0.0) and low > 0.0)
    return mean_diff, low, high, confirmed
