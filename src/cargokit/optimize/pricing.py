"""Price response and price setting.

1. Fit a logistic price-response curve per segment: P(book) vs price relative to the
   lane reference.
2. Pick the price multiplier that maximizes expected revenue per quote (or margin
   when a cost per kg is given).
3. Size overbooking with the newsvendor critical ratio on show-up uncertainty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression


def fit_price_response(bookings: pd.DataFrame) -> pd.DataFrame:
    """Per-segment logistic fit of booked ~ relative price. Returns intercept, slope."""
    rows = []
    for seg, g in bookings.groupby("segment"):
        x = (g["quoted_usd_kg"] / g["ref_usd_kg"] - 1).to_numpy().reshape(-1, 1)
        model = LogisticRegression(C=1e6).fit(x, g["booked"].to_numpy())
        rows.append(
            {
                "segment": seg,
                "intercept": model.intercept_[0],
                "slope": model.coef_[0, 0],
                "n": len(g),
            }
        )
    return pd.DataFrame(rows).set_index("segment")


def book_prob(rel_price_change: np.ndarray | float, intercept: float, slope: float):
    return 1 / (1 + np.exp(-(intercept + slope * np.asarray(rel_price_change))))


def optimal_multiplier(
    intercept: float,
    slope: float,
    cost_ratio: float = 0.0,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Best price multiplier m on the reference price.

    Maximizes (m - cost_ratio) * P(book | m), where cost_ratio is cost per kg divided
    by the reference price. Returns (m, expected value per reference-priced kg).
    """
    grid = np.linspace(0.7, 1.6, 181) if grid is None else grid
    value = (grid - cost_ratio) * book_prob(grid - 1, intercept, slope)
    i = int(np.argmax(value))
    return float(grid[i]), float(value[i])


def overbooking_limit(
    capacity_kg: float, show_rate: float, show_sd: float, spoilage_cost: float, offload_cost: float
) -> float:
    """Kg to accept so that expected spoilage and offload costs balance (newsvendor).

    Show-ups ~ Normal(show_rate * B, show_sd * B). Critical ratio
    = spoilage / (spoilage + offload). Solve show_rate*B + z*show_sd*B = capacity.
    """
    z = stats.norm.ppf(spoilage_cost / (spoilage_cost + offload_cost))
    return float(capacity_kg / (show_rate + z * show_sd))
