"""Loaders for the free public datasets SkyYield uses.

- FRED jet fuel (DJFUELUSGULF): no key needed, CSV endpoint.
- BTS T-100 international segment: download the zip from TranStats once, then load it here.
- UN Comtrade: free API key (set COMTRADE_API_KEY), limited calls per day.

Every loader returns a tidy DataFrame and never writes to disk itself; the pipeline
decides where raw data lands.
"""

from __future__ import annotations

import io
import os
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
COMTRADE_URL = "https://comtradeapi.un.org/data/v1/get/C/M/HS"

# T-100 columns we keep. Names follow the BTS "T_T100_SEGMENT_ALL_CARRIER" export.
T100_COLUMNS = {
    "YEAR": "year",
    "MONTH": "month",
    "UNIQUE_CARRIER": "carrier",
    "ORIGIN": "origin",
    "DEST": "destination",
    "FREIGHT": "freight_lbs",
    "MAIL": "mail_lbs",
    "DEPARTURES_PERFORMED": "departures",
    "AIRCRAFT_CONFIG": "aircraft_config",
}
LBS_PER_TONNE = 2204.62


def _get(url: str, headers: dict[str, str] | None = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "skyyield/0.1", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed hosts
        return resp.read()


def load_fred_series(series: str = "DJFUELUSGULF") -> pd.DataFrame:
    """Daily FRED series as columns date, value (missing days dropped)."""
    raw = _get(FRED_CSV.format(series=series))
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().reset_index(drop=True)


def load_t100_segment(path: str | Path, airports: tuple[str, ...] = ("DXB", "DWC")) -> pd.DataFrame:
    """Load a BTS T-100 segment export (zip or csv) and keep flights touching `airports`.

    Returns monthly freight tonnes by carrier and airport pair, both directions.
    """
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            with zf.open(name) as fh:
                df = pd.read_csv(fh, usecols=lambda c: c in T100_COLUMNS)
    else:
        df = pd.read_csv(path, usecols=lambda c: c in T100_COLUMNS)
    df = df.rename(columns=T100_COLUMNS)
    mask = df["origin"].isin(airports) | df["destination"].isin(airports)
    df = df.loc[mask].copy()
    df["freight_tonnes"] = df["freight_lbs"] / LBS_PER_TONNE
    df["month_start"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return (
        df.groupby(["month_start", "carrier", "origin", "destination"], as_index=False)
        .agg(freight_tonnes=("freight_tonnes", "sum"), departures=("departures", "sum"))
        .sort_values(["month_start", "origin", "destination"])
        .reset_index(drop=True)
    )


def load_comtrade(
    reporter: str = "784",  # United Arab Emirates
    period: str = "202401",
    flow: str = "X",
    api_key: str | None = None,
) -> pd.DataFrame:
    """Monthly UAE trade by partner and HS chapter from the UN Comtrade API."""
    key = api_key or os.environ.get("COMTRADE_API_KEY")
    if not key:
        raise RuntimeError("Set COMTRADE_API_KEY (free key from comtradeplus.un.org).")
    query = urllib.parse.urlencode(
        {"reporterCode": reporter, "period": period, "flowCode": flow, "cmdCode": "AG2"}
    )
    raw = _get(f"{COMTRADE_URL}?{query}", headers={"Ocp-Apim-Subscription-Key": key})
    data = pd.read_json(io.BytesIO(raw))["data"]
    df = pd.json_normalize(data)
    keep = ["period", "partnerCode", "partnerDesc", "cmdCode", "cmdDesc", "primaryValue", "netWgt"]
    return df[[c for c in keep if c in df.columns]]
