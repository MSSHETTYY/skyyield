"""Lane definitions and economic assumptions shared across modules.

Every number here is an assumption for a portfolio model, not airline data. They are
kept in one place so a reviewer can see and change them, and so tests can pin them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Lane:
    code: str
    origin: str
    destination: str
    region: str
    base_tonnes: float  # mean monthly tonnes at the start of the simulated history
    base_yield_usd_kg: float  # average revenue per chargeable kg
    block_hours: float  # one-way flight time, drives fuel burn


LANES: tuple[Lane, ...] = (
    Lane("DXB-LHR", "DXB", "LHR", "Europe", 5200, 2.10, 7.5),
    Lane("DXB-FRA", "DXB", "FRA", "Europe", 4800, 2.05, 6.8),
    Lane("DXB-JFK", "DXB", "JFK", "Americas", 4100, 3.40, 14.0),
    Lane("DXB-ORD", "DXB", "ORD", "Americas", 2600, 3.55, 14.5),
    Lane("DXB-HKG", "DXB", "HKG", "Asia", 6100, 2.60, 8.0),
    Lane("DXB-PVG", "DXB", "PVG", "Asia", 5600, 2.55, 8.5),
    Lane("DXB-BOM", "DXB", "BOM", "Asia", 3900, 1.35, 3.0),
    Lane("DXB-NBO", "DXB", "NBO", "Africa", 2200, 2.25, 5.0),
)

LANE_BY_CODE = {lane.code: lane for lane in LANES}

# Fuel burn per block hour for a wide-body freighter equivalent, in US gallons.
GALLONS_PER_BLOCK_HOUR = 3_600
# Cargo payload of that freighter equivalent, in tonnes, used to convert fuel to cost per kg.
FREIGHTER_PAYLOAD_TONNES = 100
# Non-fuel cost per kg (handling, crew, ownership, overflight), in USD.
NON_FUEL_COST_USD_KG = 0.85
