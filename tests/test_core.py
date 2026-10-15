import numpy as np
import pandas as pd
import pytest

from cargokit.analytics.funnel import funnel
from cargokit.analytics.segmentation import segment_shippers, shipper_features
from cargokit.data.quality import check_lane_demand
from cargokit.data.simulate import simulate_bookings, simulate_fuel_prices, simulate_lane_demand
from cargokit.experiments.ab import cuped, msprt_p_values, sample_size_per_arm, welch_test
from cargokit.finance.pnl import fuel_cost_per_kg, kpi_summary, lane_pnl
from cargokit.finance.risk import monte_carlo_contribution, scenario
from cargokit.forecast.backtest import rolling_backtest, summarize
from cargokit.forecast.baselines import seasonal_naive
from cargokit.optimize.capacity import FleetSpec, allocate_freighters
from cargokit.optimize.pricing import fit_price_response, optimal_multiplier, overbooking_limit


@pytest.fixture(scope="module")
def demand():
    return simulate_lane_demand()


@pytest.fixture(scope="module")
def gold(demand):
    fuel = simulate_fuel_prices()
    fuel["month"] = fuel["date"].dt.to_period("M").dt.to_timestamp()
    monthly = fuel.groupby("month", as_index=False)["usd_per_gallon"].mean()
    return demand.merge(monthly.rename(columns={"usd_per_gallon": "fuel_usd_gal"}), on="month")


@pytest.fixture(scope="module")
def bookings():
    return simulate_bookings(n_shippers=300)


def test_simulated_demand_passes_quality(demand):
    report = check_lane_demand(demand)
    assert report.ok, report.failures
    assert demand["lane"].nunique() == 8


def test_quality_catches_duplicates_and_negatives(demand):
    bad = pd.concat([demand.head(3), demand.head(1)]).assign(tonnes=[-1.0, 1, 1, 1])
    report = check_lane_demand(bad)
    assert not report.ok
    assert any("negative" in f for f in report.failures)
    assert any("duplicate" in f for f in report.failures)


def test_seasonal_naive_repeats_last_season():
    s = pd.Series(np.arange(24, dtype=float))
    assert list(seasonal_naive(s, 3)) == [12.0, 13.0, 14.0]


def test_models_beat_seasonal_naive_in_backtest(demand):
    summary = summarize(rolling_backtest(demand, horizon=6, folds=2))
    assert summary.loc["gbm", "mase"] < summary.loc["seasonal_naive", "mase"]
    assert summary.loc["ensemble", "mase"] < summary.loc["seasonal_naive", "mase"]
    # 80% intervals should cover roughly 80% of actuals
    assert 0.6 <= summary.loc["gbm", "coverage_80"] <= 0.95


def test_fuel_cost_scales_with_block_hours():
    assert fuel_cost_per_kg("DXB-JFK", 2.5) > fuel_cost_per_kg("DXB-BOM", 2.5) * 4


def test_lane_pnl_identity(gold):
    pnl = lane_pnl(gold)
    np.testing.assert_allclose(pnl["contribution_usd"], pnl["revenue_usd"] - pnl["cost_usd"])
    kpis = kpi_summary(pnl, 2025)
    assert set(kpis.index) == set(gold["lane"])


def test_fuel_shock_reduces_contribution(gold):
    plan = gold.groupby("lane", as_index=False).agg(
        tonnes=("tonnes", "mean"),
        yield_usd_kg=("yield_usd_kg", "mean"),
        fuel_usd_gal=("fuel_usd_gal", "mean"),
    )
    base = scenario(plan)["contribution_usd"].sum()
    assert scenario(plan, fuel_shift=0.3)["contribution_usd"].sum() < base


def test_hedging_narrows_contribution_distribution(gold):
    plan = (
        gold[gold["month"].dt.year == 2025]
        .groupby("lane", as_index=False)
        .agg(
            tonnes=("tonnes", "sum"),
            yield_usd_kg=("yield_usd_kg", "mean"),
            fuel_usd_gal=("fuel_usd_gal", "mean"),
        )
    )
    unhedged = monte_carlo_contribution(plan, n_sims=4000, hedge_ratio=0.0)
    hedged = monte_carlo_contribution(plan, n_sims=4000, hedge_ratio=0.8)
    assert np.std(hedged["samples"]) < np.std(unhedged["samples"])
    assert unhedged["p5_usd"] < unhedged["mean_usd"] < unhedged["p95_usd"]


def test_allocation_respects_fleet_hours_and_demand():
    lanes = pd.DataFrame(
        {
            "lane": ["DXB-LHR", "DXB-JFK", "DXB-BOM"],
            "demand_tonnes": [1500.0, 1200.0, 900.0],
            "belly_tonnes": [600.0, 400.0, 500.0],
            "yield_usd_kg": [2.1, 3.4, 1.35],
        }
    )
    fleet = FleetSpec(freighters=3)
    plan, value = allocate_freighters(lanes, fleet)
    hours = {"DXB-LHR": 7.5, "DXB-JFK": 14.0, "DXB-BOM": 3.0}
    used = sum(2 * hours[r.lane] * r.freighter_trips for r in plan.itertuples())
    assert used <= fleet.freighters * fleet.block_hours_per_aircraft_week + 1e-6
    assert (plan["sold_tonnes"] <= plan["demand_tonnes"] + 1e-6).all()
    assert value > 0


def test_price_response_recovers_sensitivity_order(bookings):
    resp = fit_price_response(bookings)
    # pharma is the least price-sensitive segment in the simulator, ecommerce the most
    assert resp.loc["pharma", "slope"] > resp.loc["ecommerce", "slope"]
    m_pharma, _ = optimal_multiplier(resp.loc["pharma", "intercept"], resp.loc["pharma", "slope"])
    m_ecom, _ = optimal_multiplier(
        resp.loc["ecommerce", "intercept"], resp.loc["ecommerce", "slope"]
    )
    assert m_pharma > m_ecom


def test_overbooking_above_capacity_when_offload_is_cheap():
    assert overbooking_limit(100_000, 0.9, 0.05, spoilage_cost=3, offload_cost=1) > 100_000


def test_funnel_rates_are_monotonic(bookings):
    f = funnel(bookings)
    row = f.iloc[0]
    assert row["quoted"] >= row["booked"] >= row["tendered"] >= row["flown"]
    by_seg = funnel(bookings, by="segment")
    assert set(by_seg.index) == set(bookings["segment"])


def test_segmentation_finds_clusters(bookings):
    feats = shipper_features(bookings)
    labels, k, score, profiles = segment_shippers(feats, k_range=range(3, 6))
    assert 3 <= k <= 5 and score > 0
    assert profiles["shippers"].sum() == len(feats)


def test_ab_toolkit_detects_real_lift_and_cuped_reduces_variance():
    rng = np.random.default_rng(1)
    pre = rng.lognormal(8, 0.6, 4000)
    y = pre * rng.lognormal(0, 0.2, 4000)
    treat = rng.random(4000) < 0.5
    y = np.where(treat, y * 1.05, y)
    adj = cuped(y, pre)
    assert np.var(adj) < np.var(y) * 0.5
    res = welch_test(adj[~treat], adj[treat])
    assert res.p_value < 0.05 and res.lift > 0
    assert sample_size_per_arm(baseline_sd=1.0, mde=0.1) == 1570


def test_msprt_controls_false_positives_under_null():
    rng = np.random.default_rng(3)
    rejections = 0
    for _ in range(200):
        p = msprt_p_values(rng.normal(size=500), rng.normal(size=500))
        rejections += p[-1] < 0.05
    assert rejections / 200 <= 0.08


def test_hedging_on_top_of_surcharge_over_hedges(gold):
    plan = (
        gold[gold["month"].dt.year == 2025]
        .groupby("lane", as_index=False)
        .agg(
            tonnes=("tonnes", "sum"),
            yield_usd_kg=("yield_usd_kg", "mean"),
            fuel_usd_gal=("fuel_usd_gal", "mean"),
        )
    )
    kw = dict(n_sims=4000, surcharge_pass_through=0.7)
    unhedged = monte_carlo_contribution(plan, hedge_ratio=0.0, **kw)
    hedged = monte_carlo_contribution(plan, hedge_ratio=0.5, **kw)
    # the surcharge already offsets most fuel risk, so a 50% swap leaves a net long position
    assert hedged["contribution_at_risk_usd"] > unhedged["contribution_at_risk_usd"]
