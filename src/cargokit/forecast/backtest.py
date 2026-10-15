"""Rolling-origin backtest comparing forecasters on the same folds.

Metrics: MASE (scaled by the seasonal-naive in-sample error, so 1.0 = as good as
repeating last year), sMAPE, and empirical coverage of the prediction interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cargokit.forecast.baselines import sarima, seasonal_naive
from cargokit.forecast.gbm import GlobalGBMForecaster


def mase(actual: np.ndarray, forecast: np.ndarray, insample: np.ndarray, season: int = 12) -> float:
    scale = np.mean(np.abs(insample[season:] - insample[:-season]))
    return float(np.mean(np.abs(actual - forecast)) / scale)


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denom = (np.abs(actual) + np.abs(forecast)) / 2
    return float(np.mean(np.abs(actual - forecast) / denom) * 100)


def rolling_backtest(
    df: pd.DataFrame,
    horizon: int = 6,
    folds: int = 4,
    step: int = 6,
    models: tuple[str, ...] = ("seasonal_naive", "sarima", "gbm", "ensemble"),
) -> pd.DataFrame:
    """Return one row per (fold, lane, model) with MASE, sMAPE and interval coverage.

    "ensemble" is the geometric mean of the SARIMA and GBM forecasts (and the average of
    their interval bounds); it needs both of them in `models`.
    """
    months = np.sort(df["month"].unique())
    results = []
    for k in range(folds):
        cutoff = months[-(horizon + k * step) - 1]
        train = df[df["month"] <= cutoff]
        test = df[(df["month"] > cutoff) & (df["month"] <= cutoff + pd.DateOffset(months=horizon))]
        gbm_fc = None
        if "gbm" in models:
            gbm_fc = GlobalGBMForecaster().fit(train).predict(train, horizon)
        for lane, hist in train.groupby("lane"):
            hist = hist.sort_values("month")
            actual = test[test["lane"] == lane].sort_values("month")["tonnes"].to_numpy()
            if len(actual) < horizon:
                continue
            insample = hist["tonnes"].to_numpy()
            preds: dict[str, tuple] = {}
            for name in models:
                if name == "seasonal_naive":
                    preds[name] = (seasonal_naive(hist["tonnes"], horizon), None, None)
                elif name == "sarima":
                    preds[name] = sarima(hist["tonnes"], horizon)
                elif name == "gbm":
                    sub = gbm_fc[gbm_fc["lane"] == lane].sort_values("month")
                    preds[name] = tuple(sub[c].to_numpy() for c in ("forecast", "lower", "upper"))
            if "ensemble" in models:
                s, g = preds["sarima"], preds["gbm"]
                preds["ensemble"] = (np.sqrt(s[0] * g[0]), (s[1] + g[1]) / 2, (s[2] + g[2]) / 2)
            for name in models:
                fc, lower, upper = preds[name]
                coverage = (
                    float(np.mean((actual >= lower) & (actual <= upper)))
                    if lower is not None
                    else np.nan
                )
                results.append(
                    {
                        "fold": k,
                        "cutoff": pd.Timestamp(cutoff),
                        "lane": lane,
                        "model": name,
                        "mase": mase(actual, fc, insample),
                        "smape": smape(actual, fc),
                        "coverage_80": coverage,
                    }
                )
    return pd.DataFrame(results)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("model")[["mase", "smape", "coverage_80"]]
        .mean()
        .sort_values("mase")
        .round(3)
    )
