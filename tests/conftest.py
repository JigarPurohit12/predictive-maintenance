"""Shared fixtures. Tests run offline, with no AWS and no local ``.env``."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from config import Settings, get_settings

#: Sensor values that sit comfortably inside the plausibility gates, so a test
#: about labelling never fails for a reason about ranges.
NOMINAL = {"volt": 170.0, "rotate": 450.0, "pressure": 100.0, "vibration": 40.0}


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> Iterable[None]:
    """Strip every setting from the environment and clear the settings cache.

    Derived from ``Settings.model_fields`` rather than a hand-maintained list, so
    adding a setting cannot silently let a developer's shell leak into a test.
    """
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    """A Settings instance with small, exact problem parameters.

    The defaults here are deliberately not the production ones: a 3-hour horizon
    on a 1-hour grid makes boundary assertions readable in a ten-row frame.
    ``train_end`` sits far past the fixture data so everything lands in
    ``train`` unless a test says otherwise.
    """
    base: dict[str, object] = {
        "_env_file": None,
        "data_dir": tmp_path / "data",
        "prediction_horizon_hours": 3,
        "feature_cadence_hours": 1,
        "post_failure_exclusion_hours": 2,
        "gap_hours": 3,
        "train_end": date(2015, 12, 30),
        "val_end": date(2015, 12, 31),
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


# --- Frame builders --------------------------------------------------------


def telemetry_frame(
    *,
    machine_ids: Sequence[int] = (1,),
    start: str = "2015-01-01 00:00:00",
    hours: int = 12,
    **sensor_overrides: float,
) -> pd.DataFrame:
    """Hourly telemetry for one or more machines, all values nominal."""
    values = {**NOMINAL, **sensor_overrides}
    origin = pd.Timestamp(start)
    records = [
        {
            "machine_id": machine_id,
            "ts": origin + timedelta(hours=offset),
            **values,
        }
        for machine_id in machine_ids
        for offset in range(hours)
    ]
    frame = pd.DataFrame.from_records(records)
    frame["machine_id"] = frame["machine_id"].astype("int32")
    return frame


def events_frame(records: Iterable[tuple[int, str, str]], code_column: str) -> pd.DataFrame:
    """Build an errors/maint/failures frame from ``(machine_id, ts, code)`` tuples."""
    rows = [
        {"machine_id": machine_id, "ts": pd.Timestamp(ts), code_column: code}
        for machine_id, ts, code in records
    ]
    frame = pd.DataFrame(rows, columns=["machine_id", "ts", code_column])
    frame["machine_id"] = frame["machine_id"].astype("int32")
    frame["ts"] = pd.to_datetime(frame["ts"])
    frame[code_column] = frame[code_column].astype("string")
    return frame


def failures_frame(records: Iterable[tuple[int, str, str]] = ()) -> pd.DataFrame:
    return events_frame(records, "component")


def maint_frame(records: Iterable[tuple[int, str, str]] = ()) -> pd.DataFrame:
    return events_frame(records, "component")


def errors_frame(records: Iterable[tuple[int, str, str]] = ()) -> pd.DataFrame:
    return events_frame(records, "error_id")


def machines_frame(machine_ids: Sequence[int] = (1,)) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "machine_id": pd.Series(machine_ids, dtype="int32"),
            "model": pd.Series([f"model{(i % 4) + 1}" for i in machine_ids], dtype="string"),
            "age": pd.Series([10 + i for i in machine_ids], dtype="int32"),
        }
    )
    return frame


def raw_frames(
    telemetry: pd.DataFrame | None = None,
    failures: pd.DataFrame | None = None,
    *,
    errors: pd.DataFrame | None = None,
    maint: pd.DataFrame | None = None,
    machines: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """The five raw tables, ready for ``duckdb_io.register_raw_frames``."""
    telemetry = telemetry_frame() if telemetry is None else telemetry
    machine_ids = sorted(telemetry["machine_id"].unique().tolist())
    return {
        "telemetry": telemetry,
        "errors": errors_frame() if errors is None else errors,
        "maint": maint_frame() if maint is None else maint,
        "failures": failures_frame() if failures is None else failures,
        "machines": machines_frame(machine_ids) if machines is None else machines,
    }


def hours_after(start: str, hours: float) -> datetime:
    return pd.Timestamp(start) + timedelta(hours=hours)
