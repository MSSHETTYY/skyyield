"""Simulated pricing A/B test: does a 5% higher quote raise revenue per shipper?

Shippers are randomized 50/50. Each one's revenue comes from a world where all its
quotes were priced at 1.00x (control) or 1.05x (treatment). Pre-period revenue is the
CUPED covariate. Writes reports/data/pricing_experiment.csv for the R report.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cargokit.data.simulate import simulate_bookings
from cargokit.experiments.ab import cuped, msprt_p_values, welch_test

N_SHIPPERS = 2000
LIFT = 1.05


def revenue_by_shipper(df: pd.DataFrame) -> pd.Series:
    return df.groupby("shipper_id")["revenue_usd"].sum()


def main() -> None:
    pre = revenue_by_shipper(simulate_bookings(N_SHIPPERS, months=3, seed=101))
    period = dict(n_shippers=N_SHIPPERS, months=3, seed=202)
    control_world = simulate_bookings(**period, price_multiplier=1.0)
    treat_world = simulate_bookings(**period, price_multiplier=LIFT)
    segments = control_world.groupby("shipper_id")["segment"].first()

    rng = np.random.default_rng(7)
    ids = segments.index.to_numpy()
    arm = pd.Series(np.where(rng.random(len(ids)) < 0.5, "treatment", "control"), index=ids)
    rev_c, rev_t = revenue_by_shipper(control_world), revenue_by_shipper(treat_world)
    df = pd.DataFrame(
        {
            "shipper_id": ids,
            "segment": segments.to_numpy(),
            "arm": arm.to_numpy(),
            "pre_revenue": pre.reindex(ids, fill_value=0).to_numpy(),
            "revenue": np.where(
                arm == "treatment",
                rev_t.reindex(ids, fill_value=0),
                rev_c.reindex(ids, fill_value=0),
            ),
        }
    )
    out = Path("reports/data")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "pricing_experiment.csv", index=False)

    is_t = (df["arm"] == "treatment").to_numpy()
    y, x = df["revenue"].to_numpy(), df["pre_revenue"].to_numpy()
    raw = welch_test(y[~is_t], y[is_t])
    adj_y = cuped(y, x)
    adj = welch_test(adj_y[~is_t], adj_y[is_t])
    seq = msprt_p_values(adj_y[~is_t], adj_y[is_t])
    print(f"shippers: {len(df)} ({is_t.sum()} treatment)")
    print(
        f"raw:   lift {raw.relative_lift:+.1%}, 95% CI [{raw.ci_low:,.0f}, {raw.ci_high:,.0f}] "
        f"USD/shipper, p={raw.p_value:.3f}"
    )
    print(
        f"CUPED: lift {adj.relative_lift:+.1%}, 95% CI [{adj.ci_low:,.0f}, {adj.ci_high:,.0f}] "
        f"USD/shipper, p={adj.p_value:.3f}"
    )
    print(f"CUPED variance reduction: {1 - np.var(adj_y) / np.var(y):.0%}")
    print(f"always-valid p-value at end: {seq[-1]:.3f}")
    by_seg = df.groupby(["segment", "arm"])["revenue"].mean().unstack()
    print((by_seg["treatment"] / by_seg["control"] - 1).map("{:+.1%}".format).to_string())


if __name__ == "__main__":
    main()
