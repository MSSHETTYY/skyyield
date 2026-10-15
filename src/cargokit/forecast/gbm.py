"""Global LightGBM model across all lanes, with quantile models for intervals.

One model learns shared seasonality and shock behaviour from every lane, which beats
per-lane models when each lane has only ~100 months of history. Forecasts are made
recursively, feeding each prediction back in as the next step's lag.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 6, 12)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling and calendar features to a (month, lane, tonnes) frame."""
    out = df.sort_values(["lane", "month"]).copy()
    out["log_t"] = np.log(out["tonnes"].clip(lower=1.0))
    g = out.groupby("lane")["log_t"]
    for lag in LAGS:
        out[f"lag_{lag}"] = g.shift(lag)
    out["roll_mean_3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    out["roll_mean_12"] = g.transform(lambda s: s.shift(1).rolling(12).mean())
    out["month_of_year"] = out["month"].dt.month
    out["lane_cat"] = out["lane"].astype("category")
    return out


FEATURES = [f"lag_{lag}" for lag in LAGS] + [
    "roll_mean_3",
    "roll_mean_12",
    "month_of_year",
    "lane_cat",
]


class GlobalGBMForecaster:
    def __init__(self, quantiles: tuple[float, float] = (0.1, 0.9), seed: int = 0):
        self.quantiles = quantiles
        self.seed = seed
        self.models: dict[str, lgb.LGBMRegressor] = {}

    def _model(self, objective: str, alpha: float | None = None) -> lgb.LGBMRegressor:
        kw = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10)
        if alpha is not None:
            kw["alpha"] = alpha
        return lgb.LGBMRegressor(objective=objective, random_state=self.seed, verbose=-1, **kw)

    def fit(self, df: pd.DataFrame) -> GlobalGBMForecaster:
        feats = make_features(df).dropna(subset=FEATURES)
        # Predict the change versus last month so the model extrapolates trends.
        target = feats["log_t"] - feats["lag_1"]
        self.lane_categories = feats["lane_cat"].cat.categories
        self.models["mean"] = self._model("regression").fit(feats[FEATURES], target)
        for q in self.quantiles:
            self.models[f"q{q}"] = self._model("quantile", q).fit(feats[FEATURES], target)
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Recursive multi-step forecast for every lane in `history`."""
        hist = history[["month", "lane", "tonnes"]].copy()
        rows = []
        # One-step quantile half-widths compound with horizon: errors add in log space,
        # so the interval at step h uses the root of the summed squared half-widths.
        cum_lo: dict[str, float] = {}
        cum_hi: dict[str, float] = {}
        for _ in range(horizon):
            next_month = hist.groupby("lane")["month"].max() + pd.offsets.MonthBegin(1)
            stub = next_month.rename("month").reset_index().assign(tonnes=np.nan)
            feats = make_features(pd.concat([hist, stub], ignore_index=True))
            feats["lane_cat"] = pd.Categorical(feats["lane"], categories=self.lane_categories)
            step = feats[feats["tonnes"].isna()].copy()
            X = step[FEATURES]
            base = step["lag_1"].to_numpy()
            mean_delta = self.models["mean"].predict(X)
            lo, hi = self.quantiles
            lo_w = np.maximum(mean_delta - self.models[f"q{lo}"].predict(X), 0.0)
            hi_w = np.maximum(self.models[f"q{hi}"].predict(X) - mean_delta, 0.0)
            lanes = step["lane"].to_numpy()
            for lane, lw, hw in zip(lanes, lo_w, hi_w):
                cum_lo[lane] = cum_lo.get(lane, 0.0) + lw**2
                cum_hi[lane] = cum_hi.get(lane, 0.0) + hw**2
            log_fc = base + mean_delta
            step["forecast"] = np.exp(log_fc)
            step["lower"] = np.exp(log_fc - np.sqrt([cum_lo[x] for x in lanes]))
            step["upper"] = np.exp(log_fc + np.sqrt([cum_hi[x] for x in lanes]))
            rows.append(step[["month", "lane", "forecast", "lower", "upper"]])
            hist = pd.concat(
                [hist, step[["month", "lane"]].assign(tonnes=step["forecast"].to_numpy())],
                ignore_index=True,
            )
        return pd.concat(rows, ignore_index=True)
