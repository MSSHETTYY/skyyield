"""Lightweight data-quality checks run on every pipeline load.

Kept dependency-free on purpose: the same checks run in pandas tests and, through
`to_pandas()` samples, after Spark jobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class QualityReport:
    table: str
    rows: int
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def check_lane_demand(df: pd.DataFrame, table: str = "gold.lane_monthly") -> QualityReport:
    report = QualityReport(table=table, rows=len(df))
    required = {"month", "lane", "tonnes", "yield_usd_kg"}
    missing = required - set(df.columns)
    if missing:
        report.failures.append(f"missing columns: {sorted(missing)}")
        return report
    if df[list(required)].isna().any().any():
        report.failures.append("null values in required columns")
    if (df["tonnes"] < 0).any():
        report.failures.append("negative tonnes")
    if not df["yield_usd_kg"].between(0.1, 20).all():
        report.failures.append("yield outside 0.1 to 20 USD/kg")
    if df.duplicated(["month", "lane"]).any():
        report.failures.append("duplicate month and lane rows")
    gaps = (
        df.sort_values("month")
        .groupby("lane")["month"]
        .apply(lambda s: (s.diff().dt.days.dropna() > 31).any())
    )
    if gaps.any():
        report.failures.append(f"month gaps in lanes: {sorted(gaps[gaps].index)}")
    return report
