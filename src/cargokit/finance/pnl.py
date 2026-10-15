"""Lane-level profit and loss.

Revenue = chargeable kg x yield. Cost = fuel (block hours x burn x price, spread over
the payload actually flown) + non-fuel cost per kg. A fuel surcharge passes part of
any fuel move above a reference price through to the yield, with a lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cargokit.config import (
    FREIGHTER_PAYLOAD_TONNES,
    GALLONS_PER_BLOCK_HOUR,
    LANE_BY_CODE,
    NON_FUEL_COST_USD_KG,
)


@dataclass(frozen=True)
class SurchargePolicy:
    reference_usd_gal: float = 2.30
    pass_through: float = 0.7  # share of the fuel cost change recovered from shippers
    lag_months: int = 1


def fuel_cost_per_kg(lane: str, fuel_usd_gal: float | np.ndarray, load_factor: float = 0.7):
    """Fuel cost per chargeable kg on a lane, given load factor of the payload."""
    hours = LANE_BY_CODE[lane].block_hours
    kg_flown = FREIGHTER_PAYLOAD_TONNES * 1000 * load_factor
    return hours * GALLONS_PER_BLOCK_HOUR * np.asarray(fuel_usd_gal) / kg_flown


def lane_pnl(
    gold: pd.DataFrame,
    load_factor: float = 0.7,
    surcharge: SurchargePolicy | None = SurchargePolicy(),
) -> pd.DataFrame:
    """Add revenue, cost and contribution columns to a gold lane-month table.

    Needs columns month, lane, tonnes, yield_usd_kg, fuel_usd_gal.
    """
    df = gold.sort_values(["lane", "month"]).copy()
    df["fuel_cost_usd_kg"] = [
        float(fuel_cost_per_kg(lane, f, load_factor))
        for lane, f in zip(df["lane"], df["fuel_usd_gal"])
    ]
    df["surcharge_usd_kg"] = 0.0
    if surcharge is not None:
        ref_cost = np.array(
            [
                float(fuel_cost_per_kg(lane, surcharge.reference_usd_gal, load_factor))
                for lane in df["lane"]
            ]
        )
        delta = (df["fuel_cost_usd_kg"].to_numpy() - ref_cost) * surcharge.pass_through
        df["surcharge_usd_kg"] = (
            pd.Series(delta, index=df.index)
            .groupby(df["lane"])
            .shift(surcharge.lag_months)
            .fillna(0.0)
        )
    df["all_in_yield_usd_kg"] = df["yield_usd_kg"] + df["surcharge_usd_kg"]
    kg = df["tonnes"] * 1000
    df["revenue_usd"] = kg * df["all_in_yield_usd_kg"]
    df["cost_usd"] = kg * (df["fuel_cost_usd_kg"] + NON_FUEL_COST_USD_KG)
    df["contribution_usd"] = df["revenue_usd"] - df["cost_usd"]
    df["margin_pct"] = df["contribution_usd"] / df["revenue_usd"] * 100
    return df


def kpi_summary(pnl: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """Annual KPIs per lane: tonnes, revenue, contribution, margin, revenue per tonne."""
    df = pnl if year is None else pnl[pnl["month"].dt.year == year]
    out = df.groupby("lane").agg(
        tonnes=("tonnes", "sum"),
        revenue_usd=("revenue_usd", "sum"),
        contribution_usd=("contribution_usd", "sum"),
    )
    out["margin_pct"] = out["contribution_usd"] / out["revenue_usd"] * 100
    out["revenue_per_tonne_usd"] = out["revenue_usd"] / out["tonnes"]
    return out.sort_values("contribution_usd", ascending=False).round(1)
