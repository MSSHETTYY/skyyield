# ADR 0002: Global gradient boosting plus SARIMA, judged by a rolling backtest

Status: accepted

## Context
Each lane has about 130 monthly points, including a 2020 shock. Per-lane deep models
overfit at this size; a single model across lanes can share seasonality.

## Decision
- Baselines: seasonal-naive and SARIMA(1,1,1)(0,1,1)12 on log tonnes.
- Main model: one LightGBM across all lanes on lag, rolling and calendar features,
  predicting the month-on-month log change, applied recursively. Quantile models give
  80% intervals, widened with horizon.
- Ensemble: geometric mean of SARIMA and LightGBM.
- The Keras LSTM is kept as an experiment and promoted only if it wins the backtest.
- Selection metric: MASE against seasonal-naive on 4 rolling folds of 6 months, plus
  interval coverage. CI fails if the best model is not at least 10% better than naive.

## Consequences
SARIMA alone is hard to beat on clean monthly data; the ensemble wins because the two
models make different errors. This is reported as is, not hidden.
