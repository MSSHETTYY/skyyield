# ADR 0001: Simulate bookings and rates, use public data for market volumes

Status: accepted

## Context
Airline booking records, contract rates and shipper data are commercially sensitive and
not public. Paid rate indices exist but cost money, and this project must cost nothing.

## Decision
- Market-level history (freight tonnes by airport pair, fuel prices, trade flows) comes
  from free public sources: BTS T-100, FRED, UN Comtrade.
- Quote-level bookings, shipper segments and contract yields are simulated by
  `cargokit.data.simulate`, with fixed seeds and documented parameters.
- Every result in the README states whether it comes from public or simulated data.

## Consequences
- Pricing, segmentation and A/B results show method, not real-world truth. Tests
  check that the methods recover the parameters the simulator was given (for example,
  pharma is the least price-sensitive segment).
- Swapping in real data later only changes the loaders, not the models.
