"""Revenue and contribution risk: scenarios, Monte Carlo, and a fuel hedge what-if.

The Monte Carlo draws correlated shocks to demand, yield, fuel and FX for the next
12 months and reports contribution-at-risk (the 5th percentile shortfall versus plan).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cargokit.config import NON_FUEL_COST_USD_KG
from cargokit.finance.pnl import fuel_cost_per_kg


@dataclass(frozen=True)
class RiskAssumptions:
    demand_vol: float = 0.08  # annualised log-volatility of tonnes
    yield_vol: float = 0.10
    fuel_vol: float = 0.30
    fx_vol: float = 0.06  # share of revenue billed in non-USD currencies moves with this
    non_usd_revenue_share: float = 0.35
    # correlation order: demand, yield, fuel, fx
    corr: tuple[tuple[float, ...], ...] = (
        (1.0, 0.4, 0.2, 0.1),
        (0.4, 1.0, 0.5, 0.0),
        (0.2, 0.5, 1.0, 0.0),
        (0.1, 0.0, 0.0, 1.0),
    )


def scenario(
    plan: pd.DataFrame,
    fuel_shift: float = 0.0,
    demand_shift: float = 0.0,
    yield_shift: float = 0.0,
    load_factor: float = 0.7,
) -> pd.DataFrame:
    """Deterministic what-if on a lane plan (lane, tonnes, yield_usd_kg, fuel_usd_gal).

    Shifts are relative, e.g. fuel_shift=0.3 means fuel +30%.
    """
    df = plan.copy()
    df["tonnes"] *= 1 + demand_shift
    df["yield_usd_kg"] *= 1 + yield_shift
    fuel = df["fuel_usd_gal"] * (1 + fuel_shift)
    kg = df["tonnes"] * 1000
    fuel_kg = np.array(
        [float(fuel_cost_per_kg(lane, f, load_factor)) for lane, f in zip(df["lane"], fuel)]
    )
    df["revenue_usd"] = kg * df["yield_usd_kg"]
    df["contribution_usd"] = df["revenue_usd"] - kg * (fuel_kg + NON_FUEL_COST_USD_KG)
    return df


def monte_carlo_contribution(
    plan: pd.DataFrame,
    n_sims: int = 10_000,
    assumptions: RiskAssumptions = RiskAssumptions(),
    hedge_ratio: float = 0.0,
    surcharge_pass_through: float = 0.0,
    load_factor: float = 0.7,
    seed: int = 0,
) -> dict[str, float | np.ndarray]:
    """Simulate total annual contribution for a 12-month lane plan.

    `hedge_ratio` locks that share of fuel at the plan price (a swap), which narrows
    the distribution: the classic fuel-hedging trade-off in one number.
    `surcharge_pass_through` recovers that share of any market fuel move from shippers
    through the fuel surcharge, a natural hedge that overlaps with a financial one.
    """
    rng = np.random.default_rng(seed)
    a = assumptions
    vols = np.array([a.demand_vol, a.yield_vol, a.fuel_vol, a.fx_vol])
    cov = np.outer(vols, vols) * np.array(a.corr)
    shocks = rng.multivariate_normal(np.zeros(4), cov, size=n_sims)
    shocks -= 0.5 * vols**2  # keep each lognormal factor's mean at 1
    d, y, f, fx = (np.exp(shocks[:, i])[:, None] for i in range(4))

    kg = plan["tonnes"].to_numpy()[None, :] * 1000 * d
    fx_factor = 1 + a.non_usd_revenue_share * (fx - 1)
    revenue = kg * plan["yield_usd_kg"].to_numpy()[None, :] * y * fx_factor
    plan_fuel = plan["fuel_usd_gal"].to_numpy()[None, :]
    eff_fuel = hedge_ratio * plan_fuel + (1 - hedge_ratio) * plan_fuel * f
    lanes = list(plan["lane"])

    def per_kg(fuel_price: np.ndarray) -> np.ndarray:
        return np.stack(
            [fuel_cost_per_kg(lane, fuel_price[:, j], load_factor) for j, lane in enumerate(lanes)],
            axis=1,
        )

    fuel_kg = per_kg(eff_fuel)
    surcharge_kg = surcharge_pass_through * (per_kg(plan_fuel * f) - per_kg(plan_fuel))
    revenue = revenue + kg * surcharge_kg
    contribution = (revenue - kg * (fuel_kg + NON_FUEL_COST_USD_KG)).sum(axis=1)

    base = scenario(plan, load_factor=load_factor)["contribution_usd"].sum()
    p5 = float(np.percentile(contribution, 5))
    return {
        "plan_contribution_usd": float(base),
        "mean_usd": float(contribution.mean()),
        "p5_usd": p5,
        "p95_usd": float(np.percentile(contribution, 95)),
        "contribution_at_risk_usd": float(base - p5),
        "samples": contribution,
    }
