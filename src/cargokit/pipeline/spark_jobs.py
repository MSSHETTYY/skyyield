"""Bronze -> silver -> gold jobs in PySpark.

Runs the same locally (parquet) and on Databricks Free Edition (set fmt="delta").
Bronze keeps raw rows plus load metadata, silver is typed and deduplicated, gold is
the lane-month table every model reads.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


def get_spark(app: str = "skyyield") -> SparkSession:
    return (
        SparkSession.builder.appName(app)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def to_bronze(df: DataFrame, source: str) -> DataFrame:
    return df.withColumn("_source", F.lit(source)).withColumn("_loaded_at", F.current_timestamp())


def lane_demand_to_silver(bronze: DataFrame) -> DataFrame:
    """Type the columns, drop bad rows, keep the latest load per (month, lane)."""
    typed = (
        bronze.withColumn("month", F.to_date("month"))
        .withColumn("tonnes", F.col("tonnes").cast("double"))
        .withColumn("yield_usd_kg", F.col("yield_usd_kg").cast("double"))
        .where(F.col("tonnes").isNotNull() & (F.col("tonnes") >= 0))
        .where(F.col("yield_usd_kg").between(0.1, 20))
    )
    latest = Window.partitionBy("month", "lane").orderBy(F.col("_loaded_at").desc())
    return (
        typed.withColumn("_rn", F.row_number().over(latest))
        .where("_rn = 1")
        .drop("_rn", "_source", "_loaded_at")
    )


def fuel_to_silver(bronze: DataFrame) -> DataFrame:
    """Daily fuel -> monthly average USD per gallon."""
    return (
        bronze.withColumn("date", F.to_date("date"))
        .withColumn("month", F.trunc("date", "month"))
        .groupBy("month")
        .agg(F.avg("usd_per_gallon").alias("fuel_usd_gal"))
    )


def build_gold_lane_monthly(lane_silver: DataFrame, fuel_silver: DataFrame) -> DataFrame:
    """One row per lane and month with revenue, fuel and year-over-year growth."""
    yoy = Window.partitionBy("lane").orderBy("month")
    return (
        lane_silver.join(fuel_silver, on="month", how="left")
        .withColumn("revenue_usd", F.col("tonnes") * 1000 * F.col("yield_usd_kg"))
        .withColumn("tonnes_ly", F.lag("tonnes", 12).over(yoy))
        .withColumn(
            "tonnes_yoy",
            F.when(F.col("tonnes_ly") > 0, F.col("tonnes") / F.col("tonnes_ly") - 1),
        )
        .drop("tonnes_ly")
        .orderBy("lane", "month")
    )


def run(lake: str | Path, lane_pdf, fuel_pdf, fmt: str = "parquet") -> DataFrame:
    """Run the full pipeline from pandas inputs and write each layer under `lake`."""
    spark = get_spark()
    lake = str(lake)
    bronze_lane = to_bronze(spark.createDataFrame(lane_pdf.astype({"month": "str"})), "lane_demand")
    bronze_fuel = to_bronze(spark.createDataFrame(fuel_pdf.astype({"date": "str"})), "fuel")
    bronze_lane.write.mode("overwrite").format(fmt).save(f"{lake}/bronze/lane_demand")
    bronze_fuel.write.mode("overwrite").format(fmt).save(f"{lake}/bronze/fuel")

    lane_silver = lane_demand_to_silver(spark.read.format(fmt).load(f"{lake}/bronze/lane_demand"))
    fuel_silver = fuel_to_silver(spark.read.format(fmt).load(f"{lake}/bronze/fuel"))
    lane_silver.write.mode("overwrite").format(fmt).save(f"{lake}/silver/lane_demand")
    fuel_silver.write.mode("overwrite").format(fmt).save(f"{lake}/silver/fuel_monthly")

    gold = build_gold_lane_monthly(lane_silver, fuel_silver)
    gold.write.mode("overwrite").format(fmt).save(f"{lake}/gold/lane_monthly")
    return gold
