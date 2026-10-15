"""Weekly freighter allocation across lanes as a mixed-integer program (OR-Tools).

Decide how many freighter round trips to fly on each lane. Each trip adds payload on
top of the lane's belly capacity; sold tonnes are capped by forecast demand and by
capacity. Maximize contribution subject to the fleet's weekly block-hour budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from ortools.linear_solver import pywraplp

from cargokit.config import FREIGHTER_PAYLOAD_TONNES, LANE_BY_CODE, NON_FUEL_COST_USD_KG
from cargokit.finance.pnl import fuel_cost_per_kg


@dataclass(frozen=True)
class FleetSpec:
    freighters: int = 10
    block_hours_per_aircraft_week: float = 95.0
    max_trips_per_lane: int = 14


def allocate_freighters(
    lanes: pd.DataFrame,
    fleet: FleetSpec = FleetSpec(),
    fuel_usd_gal: float = 2.3,
) -> tuple[pd.DataFrame, float]:
    """`lanes` needs lane, demand_tonnes, belly_tonnes, yield_usd_kg (all weekly).

    Returns the plan per lane and total weekly contribution in USD.
    """
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:  # pragma: no cover - SCIP ships with ortools wheels
        raise RuntimeError("SCIP solver unavailable")

    trips, sold, rows = {}, {}, []
    objective = solver.Objective()
    hours = solver.Constraint(
        0, fleet.freighters * fleet.block_hours_per_aircraft_week, "fleet_hours"
    )
    for r in lanes.itertuples(index=False):
        lane = LANE_BY_CODE[r.lane]
        trips[r.lane] = solver.IntVar(0, fleet.max_trips_per_lane, f"trips_{r.lane}")
        sold[r.lane] = solver.NumVar(0, r.demand_tonnes, f"sold_{r.lane}")
        # sold <= belly + trips * payload
        cap = solver.Constraint(-solver.infinity(), r.belly_tonnes, f"cap_{r.lane}")
        cap.SetCoefficient(sold[r.lane], 1)
        cap.SetCoefficient(trips[r.lane], -FREIGHTER_PAYLOAD_TONNES)
        # a round trip burns fuel both ways; cost per trip at full payload
        trip_fuel = (
            2
            * fuel_cost_per_kg(r.lane, fuel_usd_gal, load_factor=1.0)
            * FREIGHTER_PAYLOAD_TONNES
            * 1000
        )
        hours.SetCoefficient(trips[r.lane], 2 * lane.block_hours)
        objective.SetCoefficient(sold[r.lane], 1000 * (r.yield_usd_kg - NON_FUEL_COST_USD_KG))
        objective.SetCoefficient(trips[r.lane], -float(trip_fuel))
    objective.SetMaximization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"solver status {status}")
    for r in lanes.itertuples(index=False):
        rows.append(
            {
                "lane": r.lane,
                "freighter_trips": int(round(trips[r.lane].solution_value())),
                "sold_tonnes": round(sold[r.lane].solution_value(), 1),
                "demand_tonnes": r.demand_tonnes,
                "spill_tonnes": round(r.demand_tonnes - sold[r.lane].solution_value(), 1),
            }
        )
    return pd.DataFrame(rows), objective.Value()
