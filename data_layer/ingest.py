"""Landing CSVs -> typed, partitioned parquet.

CSV is the download format, not the working format. Everything downstream reads
parquet: typed, columnar, partition-pruned, and identical in shape whether it is
sitting on this laptop or under ``s3://bucket/raw/``.

Run it with::

    python -m data_layer.ingest --download      # needs a Kaggle token
    python -m data_layer.ingest                 # CSVs already in data/landing/

On partitioning: the spec asks for ``(machine_id, date)``. That is 100 machines
x 366 days = 36,600 directories holding 24 rows each, which costs more in file
overhead and listing time than it can possibly save in scan time — on S3 it is
also the classic small-files problem that makes a Glue job crawl. Telemetry is
therefore partitioned by date alone (366 partitions, ~2,400 rows each), which is
what the scoring path prunes on anyway: batches arrive by day. The four small
tables are single files. Set ``partition_columns`` on the table below if you
want the spec's layout back.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from config import Settings, configure_logging, get_settings
from data_layer.validate import ValidationReport, validate_all

LOGGER = logging.getLogger(__name__)

#: Kaggle dataset holding the five Azure PdM CSVs. Downloading by hand and
#: dropping the files into ``data/landing/`` works just as well.
KAGGLE_DATASET = "arnabbiswas1/microsoft-azure-predictive-maintenance"

#: The Azure CSVs write timestamps as ``1/1/2015 6:00:00 AM``. Parsing that
#: without a format string means pandas guesses month-vs-day per file, and a
#: file whose first rows are all <= 12 in both positions can silently come out
#: transposed. Try the known format first, fall back only if it does not match.
CSV_TIMESTAMP_FORMAT = "%m/%d/%Y %I:%M:%S %p"


@dataclass(frozen=True)
class RawTable:
    """One source CSV and how it becomes a parquet table."""

    name: str
    csv_name: str
    #: Source column -> canonical column. Applied once, here, and never again.
    rename: dict[str, str]
    #: Canonical column order. Pinned rather than inherited from the CSV so the
    #: parquet schema does not change if the source file reorders its columns.
    columns: tuple[str, ...]
    #: Natural key. Duplicates on it are a blocking validation error.
    key_columns: tuple[str, ...]
    #: Columns the parquet is partitioned by, written as directories.
    partition_columns: tuple[str, ...] = ()
    #: Columns cast to a timestamp during read.
    timestamp_columns: tuple[str, ...] = ("ts",)
    integer_columns: tuple[str, ...] = ("machine_id",)
    float_columns: tuple[str, ...] = ()
    string_columns: tuple[str, ...] = ()
    extra: dict[str, str] = field(default_factory=dict)


RAW_TABLES: tuple[RawTable, ...] = (
    RawTable(
        name="telemetry",
        csv_name="PdM_telemetry.csv",
        rename={"datetime": "ts", "machineID": "machine_id"},
        columns=("machine_id", "ts", "volt", "rotate", "pressure", "vibration"),
        key_columns=("machine_id", "ts"),
        partition_columns=("dt",),
        float_columns=("volt", "rotate", "pressure", "vibration"),
    ),
    RawTable(
        name="errors",
        csv_name="PdM_errors.csv",
        rename={"datetime": "ts", "machineID": "machine_id", "errorID": "error_id"},
        columns=("machine_id", "ts", "error_id"),
        key_columns=("machine_id", "ts", "error_id"),
        string_columns=("error_id",),
    ),
    RawTable(
        name="maint",
        csv_name="PdM_maint.csv",
        rename={"datetime": "ts", "machineID": "machine_id", "comp": "component"},
        columns=("machine_id", "ts", "component"),
        key_columns=("machine_id", "ts", "component"),
        string_columns=("component",),
    ),
    RawTable(
        name="failures",
        csv_name="PdM_failures.csv",
        rename={"datetime": "ts", "machineID": "machine_id", "failure": "component"},
        columns=("machine_id", "ts", "component"),
        key_columns=("machine_id", "ts", "component"),
        string_columns=("component",),
    ),
    RawTable(
        name="machines",
        csv_name="PdM_machines.csv",
        rename={"machineID": "machine_id"},
        columns=("machine_id", "model", "age"),
        key_columns=("machine_id",),
        timestamp_columns=(),
        integer_columns=("machine_id", "age"),
        string_columns=("model",),
    ),
)

TABLES_BY_NAME: dict[str, RawTable] = {table.name: table for table in RAW_TABLES}


def _parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse the Azure timestamp format, falling back to inference."""
    parsed = pd.to_datetime(values, format=CSV_TIMESTAMP_FORMAT, errors="coerce")
    if parsed.notna().all():
        return parsed
    fallback = pd.to_datetime(values, errors="coerce")
    unparsed = int(fallback.isna().sum())
    if unparsed:
        LOGGER.warning("%d timestamp(s) could not be parsed and became NaT.", unparsed)
    else:
        LOGGER.info("Timestamps did not match %s; used inferred parsing.", CSV_TIMESTAMP_FORMAT)
    return fallback


def read_landing_csv(path: Path, table: RawTable) -> pd.DataFrame:
    """Read one CSV and return it with canonical names and dtypes."""
    frame = pd.read_csv(path)
    unknown = set(table.rename) - set(frame.columns)
    if unknown:
        raise ValueError(f"{path.name} is missing expected column(s) {sorted(unknown)}; got {list(frame.columns)}")
    frame = frame.rename(columns=table.rename)

    for column in table.timestamp_columns:
        frame[column] = _parse_timestamps(frame[column])
    for column in table.integer_columns:
        frame[column] = frame[column].astype("int32")
    for column in table.float_columns:
        frame[column] = frame[column].astype("float64")
    for column in table.string_columns:
        frame[column] = frame[column].astype("string").str.strip()

    missing = [column for column in table.columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing canonical column(s) {missing} after rename.")

    sort_columns = [column for column in table.key_columns if column in frame.columns]
    return frame[list(table.columns)].sort_values(sort_columns, ignore_index=True)


def _add_partition_columns(frame: pd.DataFrame, table: RawTable) -> pd.DataFrame:
    if "dt" in table.partition_columns:
        frame = frame.assign(dt=frame["ts"].dt.strftime("%Y-%m-%d"))
    return frame


def write_parquet(frame: pd.DataFrame, destination: Path, table: RawTable) -> Path:
    """Write one table, replacing whatever was there. Re-ingest is idempotent."""
    if destination.exists():
        LOGGER.info("Replacing existing %s", destination)
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    frame = _add_partition_columns(frame, table)
    if table.partition_columns:
        frame.to_parquet(
            destination,
            engine="pyarrow",
            index=False,
            partition_cols=list(table.partition_columns),
        )
    else:
        frame.to_parquet(destination / "part-0.parquet", engine="pyarrow", index=False)
    return destination


def load_raw(table_name: str, settings: Settings | None = None) -> pd.DataFrame:
    """Read one ingested table back out of parquet, with dtypes restored.

    Partition columns come back as pandas categoricals; ``dt`` is redundant with
    ``ts`` and is dropped so the frame matches what went in.
    """
    settings = settings or get_settings()
    table = TABLES_BY_NAME[table_name]
    path = settings.raw_dir / table_name
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run `python -m data_layer.ingest` first.")
    frame = pd.read_parquet(path, engine="pyarrow")
    frame = frame.drop(columns=[column for column in table.partition_columns if column in frame.columns])
    sort_columns = [column for column in table.key_columns if column in frame.columns]
    return frame.sort_values(sort_columns, ignore_index=True)


def load_all_raw(settings: Settings | None = None) -> dict[str, pd.DataFrame]:
    settings = settings or get_settings()
    return {table.name: load_raw(table.name, settings) for table in RAW_TABLES}


def download_from_kaggle(landing_dir: Path, dataset: str = KAGGLE_DATASET) -> None:
    """Shell out to the Kaggle CLI. Optional: a manual download is equivalent."""
    landing_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s into %s", dataset, landing_dir)
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(landing_dir), "--unzip"],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The kaggle CLI is not installed. Either `pip install kaggle` and put an API token "
            f"in ~/.kaggle/kaggle.json, or download the CSVs by hand into {landing_dir}."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"kaggle download failed with exit code {exc.returncode}.") from exc


def ingest(settings: Settings | None = None, *, strict: bool = True) -> ValidationReport:
    """Read every landing CSV, validate, and write parquet under ``raw/``."""
    settings = settings or get_settings()
    landing, raw = settings.landing_dir, settings.raw_dir

    missing = [table.csv_name for table in RAW_TABLES if not (landing / table.csv_name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {landing}. Run with --download, or download the dataset by hand.\n"
            f"Source: https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
        )

    frames: dict[str, pd.DataFrame] = {}
    for table in RAW_TABLES:
        frame = read_landing_csv(landing / table.csv_name, table)
        LOGGER.info("Read %-10s %8d rows x %d cols", table.name, len(frame), frame.shape[1])
        frames[table.name] = frame

    report = validate_all(frames)
    report.log()
    if strict:
        report.raise_for_errors()

    for table in RAW_TABLES:
        destination = write_parquet(frames[table.name], raw / table.name, table)
        LOGGER.info("Wrote %-10s -> %s", table.name, destination)

    telemetry = frames["telemetry"]
    LOGGER.info(
        "Telemetry spans %s .. %s across %d machines.",
        telemetry["ts"].min(),
        telemetry["ts"].max(),
        telemetry["machine_id"].nunique(),
    )
    LOGGER.info("Failure records: %d across %d machines.", len(frames["failures"]), frames["failures"]["machine_id"].nunique())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="fetch the dataset from Kaggle first")
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="report validation errors but write parquet anyway",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)
    if args.download:
        download_from_kaggle(settings.landing_dir)

    report = ingest(settings, strict=args.strict)
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
