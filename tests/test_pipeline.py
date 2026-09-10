"""End to end: landing CSVs -> parquet -> DuckDB -> labels, through the CLIs.

Every other test registers pandas frames directly, which skips the parquet
round trip and the argument parsing. This one goes the whole way on a twelve-row
fixture, so a break in the on-disk path shows up here rather than the first time
the real dataset is ingested.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config import get_settings
from data_layer import ingest as ingest_module
from features import labeling as labeling_module
from features.labeling import load_training_labels

START = pd.Timestamp("2015-01-01 00:00:00")


def _format_ts(ts: pd.Timestamp) -> str:
    """The Azure CSV format: `1/1/2015 6:00:00 AM`, no zero padding."""
    return f"{ts.month}/{ts.day}/{ts.year} {ts.strftime('%I:%M:%S %p').lstrip('0')}"


def _write_landing(directory: Path, *, hours: int = 12) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    timestamps = [START + pd.Timedelta(hours=offset) for offset in range(hours)]

    telemetry = ["datetime,machineID,volt,rotate,pressure,vibration"]
    telemetry += [f"{_format_ts(ts)},1,170.0,450.0,100.0,40.0" for ts in timestamps]
    (directory / "PdM_telemetry.csv").write_text("\n".join(telemetry) + "\n", encoding="utf-8")

    (directory / "PdM_errors.csv").write_text(
        f"datetime,machineID,errorID\n{_format_ts(timestamps[2])},1,error1\n", encoding="utf-8"
    )
    (directory / "PdM_maint.csv").write_text(
        f"datetime,machineID,comp\n{_format_ts(timestamps[6])},1,comp1\n", encoding="utf-8"
    )
    (directory / "PdM_failures.csv").write_text(
        f"datetime,machineID,failure\n{_format_ts(timestamps[6])},1,comp1\n", encoding="utf-8"
    )
    (directory / "PdM_machines.csv").write_text("machineID,model,age\n1,model3,18\n", encoding="utf-8")


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the settings at a temporary data directory, via the environment.

    Going through the environment rather than constructing Settings by hand is
    the point: it exercises the same path a shell invocation takes.
    """
    data_dir = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("PREDICTION_HORIZON_HOURS", "3")
    monkeypatch.setenv("FEATURE_CADENCE_HOURS", "1")
    monkeypatch.setenv("POST_FAILURE_EXCLUSION_HOURS", "2")
    monkeypatch.setenv("GAP_HOURS", "3")
    monkeypatch.setenv("TRAIN_END", "2015-12-30")
    monkeypatch.setenv("VAL_END", "2015-12-31")
    get_settings.cache_clear()
    _write_landing(data_dir / "landing")
    return data_dir


def test_settings_reach_the_cli_from_the_environment(project: Path) -> None:
    settings = get_settings()
    assert settings.data_dir == project
    assert settings.prediction_horizon_hours == 3
    assert settings.feature_cadence_hours == 1


def test_ingest_cli_exits_clean_and_writes_parquet(project: Path) -> None:
    assert ingest_module.main([]) == 0
    assert (project / "raw" / "telemetry" / "dt=2015-01-01").exists()
    assert (project / "raw" / "failures" / "part-0.parquet").exists()


def test_labels_are_built_from_parquet_not_from_memory(project: Path) -> None:
    """The same expected rows as the in-memory fixtures, via the on-disk path."""
    assert ingest_module.main([]) == 0

    con = labeling_module.build(get_settings())
    training = load_training_labels(con)

    assert list(training["ts"]) == [START + pd.Timedelta(hours=hour) for hour in range(6)]
    assert training["label"].tolist() == [0, 0, 0, 1, 1, 1]


def test_labeling_cli_reports_and_writes(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert ingest_module.main([]) == 0
    assert labeling_module.main(["--write", "--no-results"]) == 0

    report = capsys.readouterr().out
    assert "Problem definition" in report
    assert "Class balance per split" in report

    written = pd.read_parquet(project / "curated" / "labels")
    assert len(written) == 6
    assert written["label"].sum() == 3


def test_strict_mode_fails_on_an_implausible_base_rate(project: Path) -> None:
    """50% positives on a twelve-row fixture is exactly what --strict is for."""
    assert ingest_module.main([]) == 0
    assert labeling_module.main(["--strict", "--no-results"]) == 1
