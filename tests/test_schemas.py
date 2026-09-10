"""The data contracts, and the plausibility gates hung off them."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from data_layer.schemas import (
    SENSOR_COLUMNS,
    SENSOR_RANGES,
    FailureEvent,
    MachineRecord,
    TelemetryRow,
    validate_rows,
)


def test_telemetry_row_accepts_a_nominal_reading() -> None:
    row = TelemetryRow(
        machine_id=1,
        ts=datetime(2015, 1, 1, 6),
        volt=170.0,
        rotate=450.0,
        pressure=100.0,
        vibration=40.0,
    )
    assert row.machine_id == 1
    assert row.ts == datetime(2015, 1, 1, 6)


@pytest.mark.parametrize("sensor", SENSOR_COLUMNS)
def test_sensor_values_outside_the_physical_range_are_rejected(sensor: str) -> None:
    """A unit change or a swapped column shows up here rather than in the model."""
    values = dict(zip(SENSOR_COLUMNS, (170.0, 450.0, 100.0, 40.0), strict=True))
    values[sensor] = SENSOR_RANGES[sensor][1] + 1
    with pytest.raises(ValidationError, match="outside the plausible range"):
        TelemetryRow(machine_id=1, ts=datetime(2015, 1, 1), **values)


def test_negative_sensor_values_are_rejected() -> None:
    with pytest.raises(ValidationError, match="outside the plausible range"):
        TelemetryRow(machine_id=1, ts=datetime(2015, 1, 1), volt=-1.0, rotate=450.0, pressure=100.0, vibration=40.0)


def test_unknown_columns_are_rejected() -> None:
    """extra='forbid' turns a silent rename drift into an import-time failure."""
    with pytest.raises(ValidationError):
        FailureEvent(machine_id=1, ts=datetime(2015, 1, 1), component="comp1", failure="comp1")


def test_machine_id_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        MachineRecord(machine_id=0, model="model1", age=10)


def test_validate_rows_reports_the_offending_row_index() -> None:
    frame = pd.DataFrame(
        [
            {"machine_id": 1, "ts": datetime(2015, 1, 1), "volt": 170.0, "rotate": 450.0, "pressure": 100.0, "vibration": 40.0},
            {"machine_id": 1, "ts": datetime(2015, 1, 2), "volt": 9_999.0, "rotate": 450.0, "pressure": 100.0, "vibration": 40.0},
        ]
    )
    with pytest.raises(ValueError, match="TelemetryRow row 1"):
        validate_rows(frame, TelemetryRow)


def test_validate_rows_honours_the_sample_limit() -> None:
    frame = pd.DataFrame(
        [
            {"machine_id": 1, "ts": datetime(2015, 1, 1), "volt": 170.0, "rotate": 450.0, "pressure": 100.0, "vibration": 40.0},
            {"machine_id": 1, "ts": datetime(2015, 1, 2), "volt": 9_999.0, "rotate": 450.0, "pressure": 100.0, "vibration": 40.0},
        ]
    )
    validate_rows(frame, TelemetryRow, limit=1)  # the bad row is past the limit
