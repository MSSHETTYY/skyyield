"""Shipper segmentation from booking behaviour (RFM-style features + k-means)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["recency_months", "frequency", "monetary_usd", "avg_kg", "conversion", "price_index"]


def shipper_features(bookings: pd.DataFrame) -> pd.DataFrame:
    last_month = bookings["month"].max()
    flown = bookings[bookings["flown"]]
    g = bookings.groupby("shipper_id")
    f = pd.DataFrame(
        {
            "recency_months": last_month - flown.groupby("shipper_id")["month"].max(),
            "frequency": flown.groupby("shipper_id").size(),
            "monetary_usd": flown.groupby("shipper_id")["revenue_usd"].sum(),
            "avg_kg": g["kg"].mean(),
            "conversion": g["booked"].mean(),
            "price_index": (flown["quoted_usd_kg"] / flown["ref_usd_kg"])
            .groupby(flown["shipper_id"])
            .mean(),
        }
    )
    f = f.fillna(
        {"recency_months": last_month, "frequency": 0, "monetary_usd": 0, "price_index": 1.0}
    )
    f["monetary_usd"] = np.log1p(f["monetary_usd"])
    f["avg_kg"] = np.log1p(f["avg_kg"])
    return f


def segment_shippers(features: pd.DataFrame, k_range: range = range(3, 8), seed: int = 0):
    """Choose k by silhouette score; return labels, chosen k, and cluster profiles."""
    X = StandardScaler().fit_transform(features[FEATURES])
    best = None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        score = silhouette_score(X, km.labels_)
        if best is None or score > best[0]:
            best = (score, k, km.labels_)
    score, k, labels = best
    profiled = features.assign(cluster=labels)
    profiles = profiled.groupby("cluster")[FEATURES].mean().round(2)
    profiles["shippers"] = profiled.groupby("cluster").size()
    return labels, k, float(score), profiles
