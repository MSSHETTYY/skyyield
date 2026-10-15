"""Baseline forecasters. Every fancier model has to beat these in the backtest."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def seasonal_naive(history: pd.Series, horizon: int, season: int = 12) -> np.ndarray:
    """Repeat the last observed season."""
    last = history.to_numpy()[-season:]
    return np.resize(last, horizon)


def sarima(
    history: pd.Series, horizon: int, alpha: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SARIMA(1,1,1)(0,1,1)12 on log tonnes. Returns mean, lower, upper (1 - alpha interval)."""
    y = np.log(history.astype(float).clip(lower=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(
            y.to_numpy(),
            order=(1, 1, 1),
            seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        fc = fit.get_forecast(horizon)
    mean = np.exp(fc.predicted_mean)
    ci = np.exp(fc.conf_int(alpha=alpha))
    return mean, ci[:, 0], ci[:, 1]
