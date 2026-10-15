"""CI quality gate for forecasts.

Fails the build if the best model does not beat seasonal-naive by at least 10% on
MASE, or if its 80% intervals cover less than 70% of actuals.
"""

import sys

import pandas as pd

summary = pd.read_csv(sys.argv[1], index_col="model")
best = summary.drop(index="seasonal_naive")["mase"].idxmin()
naive = summary.loc["seasonal_naive", "mase"]
problems = []
if summary.loc[best, "mase"] > 0.9 * naive:
    problems.append(f"{best} MASE {summary.loc[best, 'mase']} is not 10% better than {naive}")
if summary.loc[best, "coverage_80"] < 0.7:
    problems.append(f"{best} 80% interval coverage is {summary.loc[best, 'coverage_80']}")
print(f"best model: {best} (MASE {summary.loc[best, 'mase']}, naive {naive})")
if problems:
    sys.exit("forecast gate failed: " + "; ".join(problems))
