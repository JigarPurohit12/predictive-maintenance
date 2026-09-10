"""Landing CSV -> parquet: the rename, the date format, and idempotent rewrites."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from config import Settings
from data_layer.ingest import (
    TABLES_BY_NAME,
    ingest,
    load_all_raw,
    load_raw,
    read_landing_csv,
)
from data_layer.validate import ValidationError
from tests.conftest import make_settings

TELEMETRY_CSV = """datetime,machineID,volt,rotate,pressure,vibration
1/2/2015 6:00:00 AM,1,170.1,450.2,100.3,40.4
1/2/2015 7:00:00 AM,1,171.1,451.2,101.3,41.4
1/2/2015 8:00:00 AM,1,172.1,452.2,102.3,42.4
"""

ERRORS_CSV = """datetime,machineID,errorID
1/2/2015 7:00:00 AM,1,error1
"""

MAINT_CSV = """datetime,machineID,comp
1/2/2015 6:00:00 AM,1,comp2
"""

FAILURES_CSV = """datetime,machineID,failure
1/2/2015 8:00:00 AM,1,comp2
"""

MACHINES_CSV = """machineID,model,age
1,model3,18
"""

CSV_BY_NAME = {
    "PdM_telemetry.csv": TELEMETRY_CSV,
    "PdM_errors.csv": ERRORS_CSV,
    "PdM_maint.csv": MAINT_CSV,
    "PdM_failures.csv": FAILURES_CSV,
    "PdM_machines.csv": MACHINES_CSV,
}


@pytest.fixture
def landed(tmp_path: Path) -> Settings:
    """A settings object whose landing directory holds the five CSVs."""
    settings = make_settings(tmp_path)
    settings.landing_dir.mkdir(parents=True, exist_ok=True)
    for name, content in CSV_BY_NAME.items():
        (settings.landing_dir / name).write_text(content, encoding="utf-8")
    return settings


def test_source_columns_are_renamed_once(landed: Settings) -> None:
    frame = read_landing_csv(landed.landing_dir / "PdM_failures.csv", TABLES_BY_NAME["failures"])
    assert list(frame.columns) == ["machine_id", "ts", "component"]
    assert frame.loc[0, "component"] == "comp2"


def test_american_date_format_is_not_transposed(landed: Settings) -> None:
    """`1/2/2015` is 2 January. Guessing month-vs-day per file is how a year of
    telemetry silently comes out reordered."""
    frame = read_landing_csv(landed.landing_dir / "PdM_telemetry.csv", TABLES_BY_NAME["telemetry"])
    assert frame.loc[0, "ts"] == datetime(2015, 1, 2, 6)
    assert frame["ts"].is_monotonic_increasing


def test_missing_source_column_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "PdM_telemetry.csv"
    path.write_text("datetime,machine,volt\n1/2/2015 6:00:00 AM,1,170.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing expected column"):
        read_landing_csv(path, TABLES_BY_NAME["telemetry"])


def test_ingest_writes_every_table_and_round_trips(landed: Settings) -> None:
    report = ingest(landed)
    assert report.ok

    frames = load_all_raw(landed)
    assert set(frames) == {"telemetry", "errors", "maint", "failures", "machines"}
    telemetry = frames["telemetry"]
    assert len(telemetry) == 3
    assert list(telemetry.columns) == ["machine_id", "ts", "volt", "rotate", "pressure", "vibration"]
    assert telemetry.loc[0, "ts"] == pd.Timestamp("2015-01-02 06:00:00")
    assert telemetry["volt"].tolist() == [170.1, 171.1, 172.1]


def test_telemetry_is_partitioned_by_date(landed: Settings) -> None:
    ingest(landed)
    partitions = sorted(path.name for path in (landed.raw_dir / "telemetry").iterdir() if path.is_dir())
    assert partitions == ["dt=2015-01-02"]
    # The partition column is redundant with ts and does not come back in the frame.
    assert "dt" not in load_raw("telemetry", landed).columns


def test_small_tables_are_written_as_a_single_file(landed: Settings) -> None:
    ingest(landed)
    assert (landed.raw_dir / "machines" / "part-0.parquet").exists()


def test_re_ingesting_replaces_rather_than_appends(landed: Settings) -> None:
    ingest(landed)
    ingest(landed)
    assert len(load_raw("telemetry", landed)) == 3


def test_duplicate_rows_block_the_ingest(landed: Settings) -> None:
    path = landed.landing_dir / "PdM_telemetry.csv"
    path.write_text(TELEMETRY_CSV + "1/2/2015 6:00:00 AM,1,170.1,450.2,100.3,40.4\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="duplicates"):
        ingest(landed, strict=True)


def test_no_strict_reports_but_still_writes(landed: Settings) -> None:
    path = landed.landing_dir / "PdM_telemetry.csv"
    path.write_text(TELEMETRY_CSV + "1/2/2015 6:00:00 AM,1,170.1,450.2,100.3,40.4\n", encoding="utf-8")
    report = ingest(landed, strict=False)
    assert not report.ok
    assert (landed.raw_dir / "telemetry").exists()


def test_missing_csv_names_the_file_and_the_source(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.landing_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match=r"PdM_telemetry\.csv"):
        ingest(settings)


def test_load_raw_before_ingest_points_at_the_fix(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"data_layer\.ingest"):
        load_raw("telemetry", settings)
