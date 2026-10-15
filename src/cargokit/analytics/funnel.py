"""Quote -> booked -> tendered -> flown funnel, overall and by any dimension."""

from __future__ import annotations

import pandas as pd

STAGES = ["quoted", "booked", "tendered", "flown"]


def funnel(bookings: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    df = bookings.assign(quoted=True)
    keys = [by] if by else []
    counts = df.groupby(keys)[STAGES].sum() if keys else df[STAGES].sum().to_frame().T
    out = counts.copy()
    for prev, cur in zip(STAGES, STAGES[1:]):
        out[f"{prev}_to_{cur}"] = (counts[cur] / counts[prev]).round(3)
    out["quote_to_flown"] = (counts["flown"] / counts["quoted"]).round(3)
    return out
