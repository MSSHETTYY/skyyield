"""FastAPI service: forecasts, lane P&L scenarios, freighter allocation and pricing.

Models are trained at startup on the gold table (or simulated data when none is
configured), so the container is self-contained for a demo.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cargokit import __version__
from cargokit.config import LANE_BY_CODE
from cargokit.data.simulate import simulate_bookings, simulate_fuel_prices, simulate_lane_demand
from cargokit.finance.risk import scenario
from cargokit.forecast.gbm import GlobalGBMForecaster
from cargokit.optimize.capacity import FleetSpec, allocate_freighters
from cargokit.optimize.pricing import fit_price_response, optimal_multiplier

app = FastAPI(title="SkyYield", version=__version__)


@lru_cache(maxsize=1)
def _state():
    gold_path = os.environ.get("SKYYIELD_GOLD")
    if gold_path:
        gold = pd.read_parquet(gold_path)
    else:
        gold = simulate_lane_demand()
        fuel = simulate_fuel_prices()
        fuel["month"] = fuel["date"].dt.to_period("M").dt.to_timestamp()
        gold = gold.merge(
            fuel.groupby("month", as_index=False)["usd_per_gallon"]
            .mean()
            .rename(columns={"usd_per_gallon": "fuel_usd_gal"}),
            on="month",
        )
    model = GlobalGBMForecaster().fit(gold)
    response = fit_price_response(simulate_bookings())
    return gold, model, response


class ScenarioRequest(BaseModel):
    fuel_shift: float = Field(0.0, ge=-0.8, le=2.0, description="relative change, 0.3 = +30%")
    demand_shift: float = Field(0.0, ge=-0.8, le=2.0)
    yield_shift: float = Field(0.0, ge=-0.8, le=2.0)


class AllocationRequest(BaseModel):
    freighters: int = Field(10, ge=0, le=60)
    fuel_usd_gal: float = Field(2.3, gt=0, le=10)
    belly_share: float = Field(
        0.45, ge=0, le=1, description="share of demand belly capacity can carry"
    )


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/forecast/{lane}")
def forecast(lane: str, horizon: int = 6):
    if lane not in LANE_BY_CODE:
        raise HTTPException(404, f"unknown lane {lane}")
    if not 1 <= horizon <= 18:
        raise HTTPException(422, "horizon must be 1 to 18 months")
    gold, model, _ = _state()
    fc = model.predict(gold[["month", "lane", "tonnes"]], horizon)
    fc = fc[fc["lane"] == lane].assign(month=lambda d: d["month"].dt.strftime("%Y-%m"))
    return {"lane": lane, "unit": "tonnes", "forecast": fc.round(1).to_dict(orient="records")}


def _last_year_plan(gold: pd.DataFrame) -> pd.DataFrame:
    last = gold[gold["month"] > gold["month"].max() - pd.DateOffset(months=12)]
    return last.groupby("lane", as_index=False).agg(
        tonnes=("tonnes", "sum"),
        yield_usd_kg=("yield_usd_kg", "mean"),
        fuel_usd_gal=("fuel_usd_gal", "mean"),
    )


@app.post("/scenario")
def run_scenario(req: ScenarioRequest):
    gold, _, _ = _state()
    plan = _last_year_plan(gold)
    base = scenario(plan)
    shocked = scenario(plan, req.fuel_shift, req.demand_shift, req.yield_shift)
    out = base[["lane", "contribution_usd"]].merge(
        shocked[["lane", "contribution_usd"]], on="lane", suffixes=("_base", "_scenario")
    )
    out["delta_usd"] = out["contribution_usd_scenario"] - out["contribution_usd_base"]
    return {
        "total_delta_usd": round(float(out["delta_usd"].sum())),
        "lanes": out.round(0).to_dict(orient="records"),
    }


@app.post("/allocate")
def allocate(req: AllocationRequest):
    gold, model, _ = _state()
    nxt = model.predict(gold[["month", "lane", "tonnes"]], 1)
    yields = _last_year_plan(gold).set_index("lane")["yield_usd_kg"]
    weekly = nxt.assign(demand_tonnes=lambda d: d["forecast"] / 4.33)
    lanes = pd.DataFrame(
        {
            "lane": weekly["lane"],
            "demand_tonnes": weekly["demand_tonnes"].round(1),
            "belly_tonnes": (weekly["demand_tonnes"] * req.belly_share).round(1),
            "yield_usd_kg": weekly["lane"].map(yields).to_numpy(),
        }
    )
    plan, value = allocate_freighters(lanes, FleetSpec(freighters=req.freighters), req.fuel_usd_gal)
    return {"weekly_contribution_usd": round(value), "plan": plan.to_dict(orient="records")}


@app.get("/pricing/{segment}")
def pricing(segment: str, cost_ratio: float = 0.0):
    _, _, response = _state()
    if segment not in response.index:
        raise HTTPException(404, f"unknown segment {segment}")
    row = response.loc[segment]
    m, value = optimal_multiplier(row["intercept"], row["slope"], cost_ratio)
    return {
        "segment": segment,
        "price_multiplier": round(m, 3),
        "expected_value_per_ref_kg": round(value, 4),
    }
