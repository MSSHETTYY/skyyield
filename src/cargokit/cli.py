"""Command line entry point: `skyyield <command>`.

skyyield simulate --out data/raw      write simulated lane, fuel and booking data
skyyield pipeline --lake data/lake    run the Spark bronze/silver/gold jobs
skyyield backtest                     compare forecasters on rolling folds
skyyield report                       print lane KPIs and contribution-at-risk
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cargokit.data.simulate import simulate_bookings, simulate_fuel_prices, simulate_lane_demand


def _monthly_gold() -> pd.DataFrame:
    gold = simulate_lane_demand()
    fuel = simulate_fuel_prices()
    fuel["month"] = fuel["date"].dt.to_period("M").dt.to_timestamp()
    monthly = fuel.groupby("month", as_index=False)["usd_per_gallon"].mean()
    return gold.merge(monthly.rename(columns={"usd_per_gallon": "fuel_usd_gal"}), on="month")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="skyyield")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_sim = sub.add_parser("simulate")
    p_sim.add_argument("--out", default="data/raw")
    p_pipe = sub.add_parser("pipeline")
    p_pipe.add_argument("--lake", default="data/lake")
    p_bt = sub.add_parser("backtest")
    p_bt.add_argument("--folds", type=int, default=4)
    p_bt.add_argument("--out", help="write the per-model summary as CSV")
    sub.add_parser("report")
    args = parser.parse_args(argv)

    if args.cmd == "simulate":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        simulate_lane_demand().to_parquet(out / "lane_demand.parquet")
        simulate_fuel_prices().to_parquet(out / "fuel_daily.parquet")
        simulate_bookings().to_parquet(out / "bookings.parquet")
        print(f"wrote simulated data to {out}")
    elif args.cmd == "pipeline":
        from cargokit.pipeline.spark_jobs import run

        gold = run(args.lake, simulate_lane_demand(), simulate_fuel_prices())
        print(f"gold.lane_monthly rows: {gold.count()}")
    elif args.cmd == "backtest":
        from cargokit.forecast.backtest import rolling_backtest, summarize

        summary = summarize(rolling_backtest(simulate_lane_demand(), folds=args.folds))
        print(summary.to_string())
        if args.out:
            summary.to_csv(args.out)
    elif args.cmd == "report":
        from cargokit.finance.pnl import kpi_summary, lane_pnl
        from cargokit.finance.risk import monte_carlo_contribution

        pnl = lane_pnl(_monthly_gold())
        year = int(pnl["month"].dt.year.max())
        print(f"Lane KPIs {year}\n{kpi_summary(pnl, year).to_string()}\n")
        plan = (
            pnl[pnl["month"].dt.year == year]
            .groupby("lane", as_index=False)
            .agg(
                tonnes=("tonnes", "sum"),
                yield_usd_kg=("all_in_yield_usd_kg", "mean"),
                fuel_usd_gal=("fuel_usd_gal", "mean"),
            )
        )
        print("Contribution-at-risk (95%), USD millions")
        for passthrough in (0.0, 0.7):
            for hedge in (0.0, 0.5):
                r = monte_carlo_contribution(
                    plan, hedge_ratio=hedge, surcharge_pass_through=passthrough
                )
                print(
                    f"surcharge pass-through {passthrough:.0%}, hedge {hedge:.0%}: "
                    f"plan {r['plan_contribution_usd'] / 1e6:,.1f}, "
                    f"at risk {r['contribution_at_risk_usd'] / 1e6:,.1f}"
                )


if __name__ == "__main__":
    main()
