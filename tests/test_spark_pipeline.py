import shutil

import pytest

pyspark = pytest.importorskip("pyspark")
if shutil.which("java") is None:
    pytest.skip("Java not installed", allow_module_level=True)

from cargokit.data.quality import check_lane_demand  # noqa: E402
from cargokit.data.simulate import simulate_fuel_prices, simulate_lane_demand  # noqa: E402
from cargokit.pipeline.spark_jobs import run  # noqa: E402


@pytest.mark.spark
def test_pipeline_builds_clean_gold(tmp_path):
    demand = simulate_lane_demand(start="2022-01", end="2024-12")
    gold = run(tmp_path, demand, simulate_fuel_prices(start="2022-01-01", end="2024-12-31"))
    pdf = gold.toPandas()
    assert len(pdf) == len(demand)
    assert pdf["fuel_usd_gal"].notna().all()
    assert check_lane_demand(pdf.assign(month=lambda d: d["month"].astype("datetime64[ns]"))).ok
    # year-over-year growth exists only after the first 12 months of each lane
    assert pdf["tonnes_yoy"].isna().sum() == 12 * pdf["lane"].nunique()
