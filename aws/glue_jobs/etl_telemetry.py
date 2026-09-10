"""Glue PySpark job: raw parquet -> curated parquet, registered in the Catalog.

    spark-submit aws/glue_jobs/etl_telemetry.py --raw s3://.../raw --curated s3://.../curated

**Scope, and why it is this narrow.** This job cleans and aggregates. It does
*not* compute features. Feature engineering lives in ``features/build_features.py``
and is called from there by both training and scoring — writing a second, Spark
version of the same rolling windows is exactly the training/serving skew the
whole repository is arranged to prevent, and the two would diverge silently.
The rule of thumb: if a column ends up in ``FEATURE_ORDER``, it is not computed
here.

**Development.** Glue bills a minimum duration per job run, so iterating on this
by re-running it in AWS is a way to spend real money on a portfolio project. The
transform below is a pure function of a Spark DataFrame, so develop it against a
local ``pyspark`` session or a sample, and run it in Glue once it works.

``awsglue`` only exists inside the Glue container, so it is imported inside
:func:`main`. That keeps the module importable — and the transforms testable —
anywhere.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type-checking only, never imported at runtime
    from pyspark.sql import DataFrame

LOGGER = logging.getLogger(__name__)

SENSORS = ("volt", "rotate", "pressure", "vibration")

#: Physical plausibility gates, mirroring ``data_layer/schemas.py``. Duplicated
#: as literals rather than imported because a Glue job ships as a single file
#: with no access to the repository — keep the two in step by hand, and note
#: that ``tests/test_glue_etl.py`` asserts they match.
SENSOR_RANGES = {
    "volt": (0.0, 400.0),
    "rotate": (0.0, 1200.0),
    "pressure": (0.0, 400.0),
    "vibration": (0.0, 200.0),
}

CURATED_TABLES = ("telemetry", "errors", "maint", "failures", "machines")


def deduplicate_telemetry(frame: DataFrame) -> DataFrame:
    """One row per ``(machine_id, ts)``.

    An S3 object redelivered into a partition is the ordinary way duplicates
    appear, and every duplicate multiplies rows in every downstream join.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    window = Window.partitionBy("machine_id", "ts").orderBy(F.col("ts"))
    return frame.withColumn("_rn", F.row_number().over(window)).filter(F.col("_rn") == 1).drop("_rn")


def flag_out_of_range(frame: DataFrame) -> DataFrame:
    """Null out physically impossible readings and record that it happened.

    Nulling rather than dropping: the timestamp still exists and the other three
    sensors are still valid, and dropping the row would open a gap that the
    time-based windows would then quietly widen.
    """
    from pyspark.sql import functions as F

    condition = None
    for sensor, (low, high) in SENSOR_RANGES.items():
        out_of_range = F.col(sensor).isNotNull() & ((F.col(sensor) < low) | (F.col(sensor) > high))
        frame = frame.withColumn(sensor, F.when(out_of_range, F.lit(None)).otherwise(F.col(sensor)))
        condition = out_of_range if condition is None else (condition | out_of_range)
    return frame.withColumn("had_out_of_range", F.coalesce(condition, F.lit(False)))


def add_partitions(frame: DataFrame) -> DataFrame:
    """Partition by date. See the note in ``data_layer/ingest.py`` on why not
    ``(machine_id, date)``: 36,600 directories of 24 rows each is the classic
    small-files problem, and it is what makes a Glue job crawl."""
    from pyspark.sql import functions as F

    return frame.withColumn("dt", F.date_format(F.col("ts"), "yyyy-MM-dd"))


def curate_telemetry(frame: DataFrame) -> DataFrame:
    """The whole telemetry transform, as one pure function of a DataFrame."""
    from pyspark.sql import functions as F

    typed = (
        frame.select(
            F.col("machine_id").cast("int").alias("machine_id"),
            F.col("ts").cast("timestamp").alias("ts"),
            *[F.col(sensor).cast("double").alias(sensor) for sensor in SENSORS],
        )
        .filter(F.col("machine_id").isNotNull() & F.col("ts").isNotNull())
    )
    return add_partitions(flag_out_of_range(deduplicate_telemetry(typed)))


def curate_events(frame: DataFrame, code_column: str) -> DataFrame:
    """Errors, maintenance and failures share a shape: machine, timestamp, code."""
    from pyspark.sql import functions as F

    return (
        frame.select(
            F.col("machine_id").cast("int").alias("machine_id"),
            F.col("ts").cast("timestamp").alias("ts"),
            F.trim(F.col(code_column)).cast("string").alias(code_column),
        )
        .filter(F.col("machine_id").isNotNull() & F.col("ts").isNotNull())
        .dropDuplicates(["machine_id", "ts", code_column])
    )


def curate_machines(frame: DataFrame) -> DataFrame:
    from pyspark.sql import functions as F

    return frame.select(
        F.col("machine_id").cast("int").alias("machine_id"),
        F.trim(F.col("model")).cast("string").alias("model"),
        F.col("age").cast("int").alias("age"),
    ).dropDuplicates(["machine_id"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, help="s3://bucket/raw or a local path")
    parser.add_argument("--curated", required=True, help="s3://bucket/curated or a local path")
    parser.add_argument("--database", default="pdm_catalog", help="Glue Data Catalog database")
    parser.add_argument("--JOB_NAME", default="pdm-etl-telemetry", help="supplied by Glue")
    # Glue passes extra arguments the job may not know about; ignore them rather
    # than failing a paid job run on an unrecognised flag.
    known, unknown = parser.parse_known_args(argv)
    if unknown:
        LOGGER.info("Ignoring unrecognised arguments from the Glue runtime: %s", unknown)
    return known


def run(spark: Any, raw: str, curated: str) -> dict[str, int]:
    """Read raw, curate, write curated. Returns row counts per table."""
    counts: dict[str, int] = {}

    telemetry = curate_telemetry(spark.read.parquet(f"{raw}/telemetry"))
    telemetry.write.mode("overwrite").partitionBy("dt").parquet(f"{curated}/telemetry")
    counts["telemetry"] = telemetry.count()

    for table, code_column in (("errors", "error_id"), ("maint", "component"), ("failures", "component")):
        events = curate_events(spark.read.parquet(f"{raw}/{table}"), code_column)
        events.write.mode("overwrite").parquet(f"{curated}/{table}")
        counts[table] = events.count()

    machines = curate_machines(spark.read.parquet(f"{raw}/machines"))
    machines.write.mode("overwrite").parquet(f"{curated}/machines")
    counts["machines"] = machines.count()

    for table, rows in counts.items():
        LOGGER.info("Curated %-10s %d rows", table, rows)
    return counts


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs the Glue runtime
    """Glue entry point. ``awsglue`` only exists inside the Glue container."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    from awsglue.context import GlueContext
    from awsglue.job import Job
    from awsglue.utils import getResolvedOptions
    from pyspark.context import SparkContext

    _ = getResolvedOptions
    glue_context = GlueContext(SparkContext.getOrCreate())
    job = Job(glue_context)
    job.init(args.JOB_NAME, vars(args))

    run(glue_context.spark_session, args.raw, args.curated)

    job.commit()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
