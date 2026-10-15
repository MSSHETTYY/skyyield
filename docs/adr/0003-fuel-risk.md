# ADR 0003: Model the fuel surcharge and a fuel hedge together

Status: accepted

## Context
Fuel is the largest variable cost. Carriers reduce fuel risk in two ways: a fuel
surcharge that passes part of price moves to shippers, and financial hedges (swaps).

## Decision
The Monte Carlo in `cargokit.finance.risk` draws correlated shocks to demand, yield,
fuel and FX and applies both the surcharge pass-through and the hedge ratio.

## Consequences
The model shows the two hedges overlap: with 70% surcharge pass-through, adding a 50%
swap increases contribution-at-risk instead of reducing it. Hedge ratios must be set
net of the surcharge.
