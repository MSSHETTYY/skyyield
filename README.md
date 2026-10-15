# SkyYield

**Air cargo demand forecasting, lane economics and capacity optimization for a Dubai hub.**

SkyYield forecasts cargo tonnes on Dubai trade lanes, turns them into revenue and contribution per lane, measures revenue risk from fuel, FX and demand, decides how to allocate freighter capacity and price by customer segment, and tests pricing changes with a proper experiment. The shared code ships as `cargokit`, a tested Python package.

> **Data note.** Market history uses free public data (BTS T-100 freight statistics, FRED jet fuel prices, UN Comtrade trade flows). Airline bookings and rates are private, so quote-level data is simulated with fixed seeds and documented parameters. Every result below says which it is. See [ADR 0001](docs/adr/0001-synthetic-bookings.md).

## Results (simulated data, reproducible with the commands below)

**Forecasting:** 4 rolling folds, 6-month horizon, 8 lanes (`skyyield backtest`)

| Model | MASE (lower is better) | sMAPE % | 80% interval coverage |
| --- | --- | --- | --- |
| Ensemble (SARIMA + LightGBM) | **0.796** | **5.9** | 0.87 |
| SARIMA | 0.835 | 6.2 | 0.88 |
| Global LightGBM | 0.840 | 6.3 | 0.78 |
| Seasonal-naive baseline | 1.048 | 8.0 | n/a |

The CI pipeline fails if the best model stops beating seasonal-naive by 10%.

**Fuel risk:** 95% contribution-at-risk on a USD 337M annual plan (`skyyield report`)

| Fuel surcharge pass-through | Swap hedge | Contribution at risk |
| --- | --- | --- |
| 0% | 0% | USD 299M |
| 0% | 50% | USD 213M |
| 70% | 0% | USD 209M |
| 70% | 50% | USD 269M |

With a 70% surcharge already in place, a 50% fuel swap over-hedges and *adds* risk. Hedge ratios have to be set net of the surcharge ([ADR 0003](docs/adr/0003-fuel-risk.md)).

**Pricing experiment:** 2,000 shippers randomized, quotes +5% (`python scripts/pricing_experiment.py`)

| Analysis | Revenue lift per shipper | 95% CI (USD) | p-value |
| --- | --- | --- | --- |
| Raw Welch t-test | -1.6% | -26,141 to 16,057 | 0.64 |
| CUPED (pre-period revenue) | **-5.1%** | -27,618 to -4,376 | **0.007** |

CUPED removes 70% of the variance and turns an inconclusive test into a clear answer: a blanket 5% increase loses revenue. By segment, e-commerce (-10%) and regional forwarders (-10%) walk away while pharma (+3%) pays more, which matches the fitted price-response curves (optimal pharma price 1.29x reference, e-commerce 0.96x). An independent R analysis is in [reports/pricing_experiment.qmd](reports/pricing_experiment.qmd).

## What is inside

| Module | What it does |
| --- | --- |
| `cargokit.data.sources` | Loaders for BTS T-100 (Dubai flights), FRED jet fuel, UN Comtrade |
| `cargokit.data.simulate` | Seeded simulators for lane demand, fuel and quote-level bookings |
| `cargokit.data.quality` | Data-quality checks run on every load |
| `cargokit.pipeline.spark_jobs` | PySpark bronze, silver and gold jobs (parquet locally, Delta on Databricks) |
| `cargokit.forecast` | Seasonal-naive, SARIMA, global LightGBM with quantile intervals, Keras LSTM, rolling backtest |
| `cargokit.finance` | Lane P&L with fuel surcharge, scenarios, Monte Carlo contribution-at-risk, fuel hedge |
| `cargokit.optimize` | Freighter allocation (OR-Tools MIP), segment price response and optimal price, overbooking |
| `cargokit.analytics` | Quote-to-flown funnel, shipper segmentation (RFM features, k-means) |
| `cargokit.experiments` | Power analysis, CUPED, Welch test, always-valid sequential p-values |
| `cargokit.api` | FastAPI service: `/forecast`, `/scenario`, `/allocate`, `/pricing` |

```
public data (BTS, FRED, Comtrade) ─┐
simulated bookings ────────────────┼─> Spark bronze -> silver -> gold.lane_monthly
                                   │                              │
                                   │      forecasts <─────────────┤
                                   │      lane P&L and risk <─────┤
                                   │      capacity and pricing <──┘
                                   └─> funnel, segments, A/B tests
                                                  │
                                    FastAPI service and reports
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,spark,api]"     # Spark tests need Java 17

pytest                                # 22 tests, including a local Spark pipeline run
skyyield backtest                     # forecast comparison
skyyield report                       # lane KPIs and fuel risk
skyyield pipeline --lake data/lake    # Spark bronze/silver/gold
python scripts/pricing_experiment.py  # A/B test
uvicorn cargokit.api.app:app          # API on http://localhost:8000/docs
```

Or with Docker: `docker build -t skyyield . && docker run -p 7860:7860 skyyield`.

**Real data.** Download a T-100 International Segment export from [BTS TranStats](https://www.transtats.bts.gov/) and load it with `cargokit.data.sources.load_t100_segment(path)`; it keeps flights touching DXB and DWC. Jet fuel loads with `load_fred_series()` and needs no key. Comtrade needs a free key in `COMTRADE_API_KEY`.

## Cost

Everything runs on free tools: GitHub Actions for CI, Databricks Free Edition or local Spark, Google Colab for the Keras model, Hugging Face Spaces for the demo. No paid service is required.

## Roadmap

- [ ] Ingest real T-100 Dubai freight history and compare with the simulated lanes
- [ ] Train and backtest the Keras LSTM on Colab GPU
- [ ] MLflow experiment tracking and model registry
- [ ] Stakeholder dashboard with lane KPIs, forecasts and scenario sliders, deployed to Hugging Face Spaces
- [ ] Comtrade trade-flow features as leading indicators for demand
- [ ] Databricks notebook version of the pipeline with Delta tables

## Design decisions

- [ADR 0001: Simulated bookings, public market data](docs/adr/0001-synthetic-bookings.md)
- [ADR 0002: Forecasting approach and quality gate](docs/adr/0002-forecasting-approach.md)
- [ADR 0003: Fuel surcharge and hedging together](docs/adr/0003-fuel-risk.md)

## License

MIT
