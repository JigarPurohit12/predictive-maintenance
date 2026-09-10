"""DuckDB session plumbing: attach the raw tables, run the SQL files.

DuckDB stands in for Athena locally. The point is that the ``.sql`` files under
``sql/`` never name a file path or a literal horizon — they read relations
called ``raw_*`` and a one-row ``params`` table, both of which are created here.
Locally those relations are views over parquet; in tests they are pandas frames;
on AWS they are Glue Catalog tables. Same SQL, three backings.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import duckdb
import pandas as pd

from config import SQL_DIR, Settings, get_settings
from data_layer.ingest import RAW_TABLES

LOGGER = logging.getLogger(__name__)

STAGING_SQL = "01_staging.sql"
LABELS_SQL = "02_failure_labels.sql"


def connect(settings: Settings | None = None) -> duckdb.DuckDBPyConnection:
    """An in-memory database. Nothing here is worth persisting: the inputs are
    parquet on disk and the outputs are written back out as parquet."""
    settings = settings or get_settings()
    return duckdb.connect(database=":memory:")


def register_params(con: duckdb.DuckDBPyConnection, settings: Settings | None = None) -> None:
    """Push the problem definition from config.py into SQL as a one-row table."""
    settings = settings or get_settings()
    con.execute(
        """
        CREATE OR REPLACE TABLE params AS
        SELECT
            CAST(? AS INTEGER)   AS cadence_hours,
            CAST(? AS INTEGER)   AS horizon_hours,
            CAST(? AS INTEGER)   AS exclusion_hours,
            CAST(? AS TIMESTAMP) AS train_end_ts,
            CAST(? AS TIMESTAMP) AS val_start_ts,
            CAST(? AS TIMESTAMP) AS val_end_ts,
            CAST(? AS TIMESTAMP) AS test_start_ts
        """,
        [
            settings.feature_cadence_hours,
            settings.prediction_horizon_hours,
            settings.post_failure_exclusion_hours,
            settings.train_end_ts,
            settings.val_start_ts,
            settings.val_end_ts,
            settings.test_start_ts,
        ],
    )


def register_raw_parquet(con: duckdb.DuckDBPyConnection, settings: Settings | None = None) -> None:
    """Create ``raw_*`` views over the parquet written by data_layer.ingest."""
    settings = settings or get_settings()
    for table in RAW_TABLES:
        path = settings.raw_dir / table.name
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist. Run `python -m data_layer.ingest` first.")
        glob = (path / "**" / "*.parquet").as_posix()
        con.execute(
            f"CREATE OR REPLACE VIEW raw_{table.name} AS "
            f"SELECT * FROM read_parquet('{glob}', hive_partitioning = true)"
        )
        LOGGER.debug("Attached raw_%s from %s", table.name, glob)


def register_raw_frames(con: duckdb.DuckDBPyConnection, frames: Mapping[str, pd.DataFrame]) -> None:
    """Create ``raw_*`` views over in-memory frames — the path tests use.

    This is what lets the labelling tests assert against the production SQL on a
    hand-built ten-row frame instead of against a second Python implementation
    that would drift away from it.
    """
    for name, frame in frames.items():
        con.register(f"_frame_{name}", frame)
        con.execute(f"CREATE OR REPLACE VIEW raw_{name} AS SELECT * FROM _frame_{name}")


#: The staged tables, by the name they carry in SQL.
STAGED_TABLES: tuple[str, ...] = ("telemetry", "errors", "maint", "failures", "machines")


def load_staged(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """Pull the staged tables back into pandas for feature engineering.

    Feature code reads *staged* tables, never raw ones: the deduplication and
    the type casts in ``sql/01_staging.sql`` are the single cleaning path, and a
    second one in pandas is how training and serving start to disagree.
    """
    staged: dict[str, pd.DataFrame] = {}
    for name in STAGED_TABLES:
        order = "machine_id" if name == "machines" else "machine_id, ts"
        staged[name] = con.execute(f"SELECT * FROM stg_{name} ORDER BY {order}").df()
    return staged


def run_sql_file(con: duckdb.DuckDBPyConnection, filename: str, sql_dir: Path = SQL_DIR) -> None:
    path = sql_dir / filename
    LOGGER.debug("Executing %s", path)
    con.execute(path.read_text(encoding="utf-8"))


def build_staging(con: duckdb.DuckDBPyConnection) -> None:
    run_sql_file(con, STAGING_SQL)


def build_labels(con: duckdb.DuckDBPyConnection) -> None:
    run_sql_file(con, LABELS_SQL)


def prepare(
    settings: Settings | None = None,
    frames: Mapping[str, pd.DataFrame] | None = None,
) -> duckdb.DuckDBPyConnection:
    """A connection with params, raw views, staging and labels all in place.

    Pass ``frames`` to back the raw views with in-memory data; leave it None to
    read the parquet under ``settings.raw_dir``.
    """
    settings = settings or get_settings()
    con = connect(settings)
    register_params(con, settings)
    if frames is None:
        register_raw_parquet(con, settings)
    else:
        register_raw_frames(con, frames)
    build_staging(con)
    build_labels(con)
    return con
