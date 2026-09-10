"""Data contracts for the five raw tables, pinned before any logic is written.

The pydantic models are the specification and the fixture-builder for tests;
they are deliberately *not* applied row-by-row to 876k telemetry rows in the
ingest path. Row-wise validation of a frame that size costs minutes and buys
nothing that :mod:`data_layer.validate` does not check more cheaply on the
column. Use :func:`validate_rows` on samples and on hand-built frames.

Canonical column names are used everywhere downstream. The raw CSVs use
``datetime`` / ``machineID`` / ``errorID`` / ``comp`` / ``failure``; the rename
happens once, in :mod:`data_layer.ingest`, and never again.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

#: The four sensors, in the order they appear in the source data. Feature
#: ordering is owned by ``features/contract.py``; this is only the raw order.
SENSOR_COLUMNS: Final[tuple[str, ...]] = ("volt", "rotate", "pressure", "vibration")

#: Plausibility gates, not distribution gates. These are wide enough that a
#: healthy dataset never trips them, and narrow enough to catch a unit change, a
#: sentinel value or a column swapped during ingest. Observed ranges in the
#: Azure dataset sit roughly at volt 97-255, rotate 139-695, pressure 51-185,
#: vibration 14-77, so each bound has room either side.
SENSOR_RANGES: Final[dict[str, tuple[float, float]]] = {
    "volt": (0.0, 400.0),
    "rotate": (0.0, 1200.0),
    "pressure": (0.0, 400.0),
    "vibration": (0.0, 200.0),
}

#: Telemetry arrives hourly. The gap check in :mod:`data_layer.validate` is
#: written against this and nothing else assumes it.
TELEMETRY_FREQ_HOURS: Final[int] = 1


class _Row(BaseModel):
    """Shared config: reject unknown fields, so a rename drift fails loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TelemetryRow(_Row):
    """One hourly reading from one machine."""

    machine_id: int = Field(ge=1)
    ts: datetime
    volt: float
    rotate: float
    pressure: float
    vibration: float

    @field_validator("volt", "rotate", "pressure", "vibration")
    @classmethod
    def _within_physical_range(cls, value: float, info) -> float:
        low, high = SENSOR_RANGES[info.field_name]
        if not low <= value <= high:
            raise ValueError(f"{info.field_name}={value} outside the plausible range [{low}, {high}].")
        return value


class ErrorEvent(_Row):
    """A logged error code. An error is not a failure: most machines log errors
    continuously and never fail. They are predictive input, not the target."""

    machine_id: int = Field(ge=1)
    ts: datetime
    error_id: str


class MaintRecord(_Row):
    """A component replacement. Covers both scheduled maintenance and the
    replacement that follows a failure, so a maintenance record on its own does
    not imply anything broke."""

    machine_id: int = Field(ge=1)
    ts: datetime
    component: str


class FailureEvent(_Row):
    """A component failure — the thing being predicted. One row per component,
    so a single machine-moment can carry several."""

    machine_id: int = Field(ge=1)
    ts: datetime
    component: str


class MachineRecord(_Row):
    """Static machine metadata."""

    machine_id: int = Field(ge=1)
    model: str
    age: int = Field(ge=0)


def validate_rows(frame: pd.DataFrame, model: type[_Row], *, limit: int | None = None) -> None:
    """Run ``model`` over the frame, raising on the first bad row.

    ``limit`` samples the first N rows; leave it None for hand-built frames and
    set it for anything of real size.
    """
    subset = frame.head(limit) if limit is not None else frame
    for position, record in enumerate(subset.to_dict(orient="records")):
        try:
            model(**record)
        except Exception as exc:
            raise ValueError(f"{model.__name__} row {position} failed validation: {exc}") from exc
