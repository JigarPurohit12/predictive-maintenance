"""Shared fixtures. Tests run offline, with no AWS and no local ``.env``."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
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


# --- A dataset big enough to run the whole pipeline over -------------------

DATASET_START = "2015-01-01 00:00:00"
#: Ninety days at hourly cadence. Long enough that the 3-hour scoring grid
#: yields ~720 rows per machine, so three failures per machine land the base
#: rate near 3% — the range the real dataset sits in. A shorter span makes every
#: downstream metric meaningless: at 12 days the fixture was 28% positive, which
#: is not a rare-event problem at all.
DATASET_DAYS = 90

#: One failure inside each split, so no split is all-negative.
FAILURE_DAYS = (15, 45, 75)


def pipeline_settings(tmp_path: Path, **overrides: object) -> Settings:
    """Production-shaped settings sized to :func:`synthetic_dataset`.

    The problem parameters are the real ones (24h horizon, 3h cadence, 24h
    exclusion and gap); only the split dates are moved to fit the fixture.
    """
    base: dict[str, object] = {
        "prediction_horizon_hours": 24,
        "feature_cadence_hours": 3,
        "post_failure_exclusion_hours": 24,
        "gap_hours": 24,
        "window_sizes_hours": [3, 12, 24],
        "train_end": date(2015, 1, 31),
        "val_end": date(2015, 2, 28),
    }
    base.update(overrides)
    return make_settings(tmp_path, **base)


def synthetic_dataset(
    *,
    machine_ids: Sequence[int] = (1, 2, 3, 4, 5, 6),
    days: int = DATASET_DAYS,
    failure_days: Sequence[int] = FAILURE_DAYS,
    seed: int = 7,
) -> dict[str, pd.DataFrame]:
    """Five tables with enough shape to exercise windows, splits and labels.

    Deliberately *not* realistic: the signal is a crude ramp in vibration before
    each failure. It is here so the pipeline has something to chew on, never so
    that a metric computed from it means anything.
    """
    rng = np.random.default_rng(seed)
    hours = days * 24
    telemetry = telemetry_frame(machine_ids=machine_ids, start=DATASET_START, hours=hours)

    for column, scale in (("volt", 8.0), ("rotate", 25.0), ("pressure", 5.0), ("vibration", 2.5)):
        telemetry[column] = telemetry[column] + rng.normal(0.0, scale, size=len(telemetry))

    origin = pd.Timestamp(DATASET_START)
    failure_records: list[tuple[int, str, str]] = []
    for offset, machine_id in enumerate(machine_ids):
        for day in failure_days:
            moment = origin + timedelta(hours=day * 24 + 6 + offset)
            failure_records.append((machine_id, str(moment), f"comp{(offset % 4) + 1}"))
            # A ramp in the twelve hours before the failure, so a model has
            # something to find and the leakage tests have something to move.
            ramp = (telemetry["machine_id"] == machine_id) & telemetry["ts"].between(
                moment - timedelta(hours=12), moment
            )
            telemetry.loc[ramp, "vibration"] += np.linspace(0.0, 18.0, int(ramp.sum()))

    # Errors and maintenance are scattered per machine rather than laid on a
    # fixed clock. A deterministic every-53-hours maintenance schedule makes
    # `hours_since_maint_*` a perfect proxy for the day of the week, which a
    # tree memorises instead of learning the sensor signal — an artifact of the
    # fixture that looks exactly like a modelling failure.
    error_records = [
        (machine_id, str(origin + timedelta(hours=int(hour))), f"error{rng.integers(1, 6)}")
        for machine_id in machine_ids
        for hour in np.sort(rng.choice(hours, size=hours // 18, replace=False))
    ]
    maint_records = [
        (machine_id, str(origin + timedelta(hours=int(hour))), f"comp{rng.integers(1, 5)}")
        for machine_id in machine_ids
        for hour in np.sort(rng.choice(hours, size=hours // 60, replace=False))
    ]

    return {
        "telemetry": telemetry,
        "errors": errors_frame(error_records),
        "maint": maint_frame(maint_records),
        "failures": failures_frame(failure_records),
        "machines": machines_frame(machine_ids),
    }


def _azure_timestamp(ts: pd.Timestamp) -> str:
    """`1/1/2015 6:00:00 AM` — the source format, no zero padding."""
    return f"{ts.month}/{ts.day}/{ts.year} {ts.strftime('%I:%M:%S %p').lstrip('0')}"


def write_landing(directory: Path, frames: dict[str, pd.DataFrame]) -> None:
    """Serialise the five tables back into the CSV shape ingest expects."""
    directory.mkdir(parents=True, exist_ok=True)
    layout = {
        "telemetry": ("PdM_telemetry.csv", {"ts": "datetime", "machine_id": "machineID"}),
        "errors": ("PdM_errors.csv", {"ts": "datetime", "machine_id": "machineID", "error_id": "errorID"}),
        "maint": ("PdM_maint.csv", {"ts": "datetime", "machine_id": "machineID", "component": "comp"}),
        "failures": ("PdM_failures.csv", {"ts": "datetime", "machine_id": "machineID", "component": "failure"}),
        "machines": ("PdM_machines.csv", {"machine_id": "machineID"}),
    }
    for name, (filename, rename) in layout.items():
        frame = frames[name].copy()
        if "ts" in frame.columns:
            frame["ts"] = frame["ts"].map(_azure_timestamp)
        frame.rename(columns=rename).to_csv(directory / filename, index=False)


@pytest.fixture(scope="session")
def built_project(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Ingest the synthetic dataset and build the feature matrix, once.

    Session-scoped because ingest plus feature computation over 13k telemetry
    rows costs a couple of seconds, and the phase 3 tests all want the same
    matrix. Nothing mutates it, and ``_env_file=None`` keeps it isolated from
    the environment regardless of scope.
    """
    from data_layer.ingest import ingest
    from features.build_features import build_training_matrix, write_matrix
    from features.contract import build_contract

    root = tmp_path_factory.mktemp("project")
    settings = pipeline_settings(root, mlflow_tracking_uri=f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    write_landing(settings.landing_dir, synthetic_dataset())
    ingest(settings)

    contract = build_contract(settings)
    write_matrix(build_training_matrix(settings, contract), settings, contract)
    return settings
