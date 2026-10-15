"""Synthetic data generators.

Real airline bookings and contracted cargo rates are private, so SkyYield simulates
them. The simulators are seeded and calibrated so their totals behave like public
series (seasonality, a 2020 shock, fuel pass-through). See docs/adr/0001.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cargokit.config import LANES

# Month-of-year demand index: soft Q1 (Lunar New Year), Q4 peak season.
SEASONALITY = np.array([0.92, 0.85, 1.00, 0.98, 0.99, 0.97, 0.96, 0.97, 1.02, 1.08, 1.14, 1.12])

SEGMENTS = {
    # name: (share of shippers, mean monthly quotes, price sensitivity, show-up rate)
    "global_forwarder": (0.15, 60, 1.6, 0.93),
    "regional_forwarder": (0.35, 18, 2.4, 0.88),
    "ecommerce": (0.20, 30, 3.1, 0.90),
    "pharma": (0.10, 8, 0.9, 0.97),
    "perishables": (0.20, 14, 2.0, 0.85),
}


def simulate_fuel_prices(
    start: str = "2015-01-01", end: str = "2025-12-31", seed: int = 7, s0: float = 1.6
) -> pd.DataFrame:
    """Daily jet fuel price (USD/gallon) as a mean-reverting log process.

    Used only when the real FRED series (DJFUELUSGULF) is not available offline.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    log_p = np.empty(len(dates))
    log_p[0] = np.log(s0)
    mu, kappa, sigma = np.log(2.3), 0.004, 0.022
    for t in range(1, len(dates)):
        log_p[t] = log_p[t - 1] + kappa * (mu - log_p[t - 1]) + sigma * rng.standard_normal()
    return pd.DataFrame({"date": dates, "usd_per_gallon": np.exp(log_p)})


def simulate_lane_demand(
    start: str = "2015-01", end: str = "2025-12", seed: int = 11
) -> pd.DataFrame:
    """Monthly chargeable tonnes and yield per lane.

    Columns: month, lane, tonnes, yield_usd_kg.
    """
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, end, freq="M")
    t = np.arange(len(months))
    month_idx = months.month.to_numpy() - 1
    # 2020 shock: belly capacity disappears (Apr-Jun) then yields spike while volume recovers.
    shock = np.ones(len(months))
    yield_shock = np.ones(len(months))
    for i, p in enumerate(months):
        if p.year == 2020 and p.month in (4, 5, 6):
            shock[i] = 0.72
            yield_shock[i] = 1.65
        elif p.year == 2020 and p.month >= 7 or p.year == 2021:
            yield_shock[i] = 1.35
        elif p.year == 2022 and p.month <= 6:
            yield_shock[i] = 1.25

    rows = []
    for lane in LANES:
        growth = 0.0025 + 0.002 * rng.standard_normal()  # monthly trend
        level = lane.base_tonnes * np.exp(growth * t) * SEASONALITY[month_idx] * shock
        ar = np.zeros(len(months))
        for i in range(1, len(months)):
            ar[i] = 0.55 * ar[i - 1] + 0.05 * rng.standard_normal()
        tonnes = level * np.exp(ar)
        yld = lane.base_yield_usd_kg * yield_shock * np.exp(0.04 * rng.standard_normal(len(months)))
        rows.append(
            pd.DataFrame(
                {
                    "month": months.to_timestamp(),
                    "lane": lane.code,
                    "tonnes": tonnes.round(1),
                    "yield_usd_kg": yld.round(3),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def simulate_bookings(
    n_shippers: int = 400,
    months: int = 12,
    seed: int = 23,
    price_multiplier: float = 1.0,
    segment_seed: int = 0,
) -> pd.DataFrame:
    """Quote-level booking funnel: quote -> booked -> tendered -> flown.

    `price_multiplier` scales quoted prices, which lets the experimentation module
    simulate a pricing A/B test. Shipper segments come from `segment_seed`, so runs with
    different `seed`s describe the same shippers in different periods.

    Columns: quote_id, shipper_id, segment, lane, month, kg, quoted_usd_kg, ref_usd_kg,
    booked, tendered, flown, revenue_usd.
    """
    rng = np.random.default_rng(seed)
    names = list(SEGMENTS)
    shares = np.array([SEGMENTS[s][0] for s in names])
    seg_rng = np.random.default_rng(segment_seed)
    shipper_segment = seg_rng.choice(names, size=n_shippers, p=shares / shares.sum())
    lane_codes = [lane.code for lane in LANES]
    ref_yield = {lane.code: lane.base_yield_usd_kg for lane in LANES}

    frames = []
    for sid, seg in enumerate(shipper_segment):
        _, mean_quotes, sensitivity, show_rate = SEGMENTS[seg]
        n = rng.poisson(mean_quotes * months)
        if n == 0:
            continue
        home_lanes = rng.choice(lane_codes, size=2, replace=False)
        lane = rng.choice(home_lanes, size=n)
        ref = np.array([ref_yield[c] for c in lane])
        quoted = ref * np.exp(0.08 * rng.standard_normal(n)) * price_multiplier
        kg = np.round(rng.lognormal(mean=np.log(1800), sigma=0.9, size=n))
        # Booking probability falls with price relative to the lane reference.
        logit = 1.2 - sensitivity * (quoted / ref - 1.0) * 4.0
        booked = rng.random(n) < 1 / (1 + np.exp(-logit))
        tendered = booked & (rng.random(n) < show_rate)
        flown = tendered & (rng.random(n) < 0.96)  # offloads for capacity or documents
        frames.append(
            pd.DataFrame(
                {
                    "shipper_id": sid,
                    "segment": seg,
                    "lane": lane,
                    "month": rng.integers(1, months + 1, size=n),
                    "kg": kg,
                    "quoted_usd_kg": quoted.round(3),
                    "ref_usd_kg": ref,
                    "booked": booked,
                    "tendered": tendered,
                    "flown": flown,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "quote_id", np.arange(len(df)))
    df["revenue_usd"] = np.where(df["flown"], df["kg"] * df["quoted_usd_kg"], 0.0).round(2)
    return df
