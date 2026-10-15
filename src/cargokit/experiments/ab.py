"""A/B testing toolkit: power analysis, CUPED, Welch test, and a sequential test.

Built for pricing experiments where the unit is the shipper and the metric is
revenue per shipper, a heavy-tailed metric where CUPED pays off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def sample_size_per_arm(
    baseline_sd: float, mde: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Units per arm to detect an absolute lift `mde` with a two-sided test."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return int(np.ceil(2 * ((z_a + z_b) * baseline_sd / mde) ** 2))


def cuped(y: np.ndarray, x_pre: np.ndarray) -> np.ndarray:
    """Variance-reduced metric y - theta * (x - mean(x)), theta = cov(y, x) / var(x)."""
    theta = np.cov(y, x_pre)[0, 1] / np.var(x_pre, ddof=1)
    return y - theta * (x_pre - x_pre.mean())


@dataclass
class TestResult:
    lift: float
    ci_low: float
    ci_high: float
    p_value: float
    relative_lift: float


def welch_test(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05) -> TestResult:
    diff = treatment.mean() - control.mean()
    se = np.sqrt(treatment.var(ddof=1) / len(treatment) + control.var(ddof=1) / len(control))
    df = se**4 / (
        (treatment.var(ddof=1) / len(treatment)) ** 2 / (len(treatment) - 1)
        + (control.var(ddof=1) / len(control)) ** 2 / (len(control) - 1)
    )
    t = stats.t.ppf(1 - alpha / 2, df)
    p = 2 * stats.t.sf(abs(diff / se), df)
    return TestResult(diff, diff - t * se, diff + t * se, float(p), float(diff / control.mean()))


def msprt_p_values(
    control: np.ndarray, treatment: np.ndarray, tau: float | None = None
) -> np.ndarray:
    """Always-valid p-values (mixture SPRT, normal approximation) after each pair of units.

    Lets the team peek at results daily without inflating the false-positive rate.
    """
    n = min(len(control), len(treatment))
    d = treatment[:n] - control[:n]
    sigma2 = np.var(d, ddof=1)
    tau2 = tau**2 if tau is not None else sigma2  # mixing variance
    k = np.arange(1, n + 1)
    mean = np.cumsum(d) / k
    lam = np.sqrt(sigma2 / (sigma2 + k * tau2)) * np.exp(
        (k**2 * tau2 * mean**2) / (2 * sigma2 * (sigma2 + k * tau2))
    )
    return np.minimum.accumulate(np.minimum(1.0, 1 / lam))
