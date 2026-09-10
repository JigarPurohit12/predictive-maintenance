"""Time-window features: rolling sensor statistics, lags, trends, error counts.

Every function here is **trailing only**. A feature at ``ts`` is computed from
observations in ``(ts - window, ts]`` — the window is closed on the right, so it
includes the reading taken at ``ts`` itself, which is exactly what the scoring
job will have in hand, and nothing after it.

The spec's prose says "closed on the right" while its pandas hint says
``closed='left'``, which in pandas means the opposite: ``[ts - window, ts)``,
excluding the current reading. The prose wins here, because excluding the
current reading throws away information that is genuinely available at scoring
time. :data:`WINDOW_CLOSED` is the one place that decision lives; flip it and
both the training and the serving path change together, which is the whole
point of this module being shared.

What matters either way is that no feature at ``ts`` may see data after ``ts``.
``tests/test_windows.py`` plants a spike and proves it.

Windows are **time-based**, not row-based: ``rolling('24h')`` rather than
``rolling(24)``. Telemetry has gaps, and a row-based window silently reaches
further back in time whenever readings are missing.
"""

from __future__ import annotations

import logging
from typing import Final, Literal

import pandas as pd

from data_layer.schemas import SENSOR_COLUMNS
from features.contract import ERROR_CODES, LAG_PERIODS, ROLLING_STATS

LOGGER = logging.getLogger(__name__)

Closed = Literal["right", "left", "both", "neither"]

#: See the module docstring. "right" includes the observation at ``ts``.
WINDOW_CLOSED: Final[Closed] = "right"

KEY_COLUMNS: Final[list[str]] = ["machine_id", "ts"]


def _require_sorted_keys(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in [*KEY_COLUMNS, *SENSOR_COLUMNS] if column not in frame.columns]
    if missing:
        raise ValueError(f"telemetry is missing column(s) {missing}")
    return frame.sort_values(KEY_COLUMNS, kind="stable", ignore_index=True)


def rolling_sensor_features(
    telemetry: pd.DataFrame,
    window_sizes_hours: list[int],
    *,
    closed: Closed = WINDOW_CLOSED,
) -> pd.DataFrame:
    """Mean, std, min and max per sensor per window, per machine.

    Returned as a frame keyed by ``(machine_id, ts)`` so the caller decides how
    to join it. Standard deviation over a window of one observation is NaN by
    definition; warm-up rows are dropped in :mod:`features.build_features`
    rather than filled, because filling them invents a stability the machine has
    not demonstrated.
    """
    frame = _require_sorted_keys(telemetry)
    indexed = frame.set_index("ts")

    blocks: list[pd.DataFrame] = []
    for window in window_sizes_hours:
        grouped = indexed.groupby("machine_id", sort=False)[list(SENSOR_COLUMNS)]
        rolled = grouped.rolling(f"{window}h", closed=closed).agg(list(ROLLING_STATS))
        rolled.columns = [f"{sensor}_{stat}_{window}h" for sensor, stat in rolled.columns]
        blocks.append(rolled)

    combined = pd.concat(blocks, axis=1).reset_index()
    return combined.sort_values(KEY_COLUMNS, kind="stable", ignore_index=True)


def lag_features(
    telemetry: pd.DataFrame,
    lag_periods: tuple[int, ...] = LAG_PERIODS,
) -> pd.DataFrame:
    """Raw sensor values shifted back N periods, per machine.

    ``shift`` with a positive period only ever reaches backwards, and grouping
    by machine stops one machine's history leaking into the next one's.
    """
    frame = _require_sorted_keys(telemetry)
    out = frame[KEY_COLUMNS].copy()
    grouped = frame.groupby("machine_id", sort=False)
    for sensor in SENSOR_COLUMNS:
        for period in lag_periods:
            out[f"{sensor}_lag_{period}"] = grouped[sensor].shift(period)
    return out


def trend_features(rolling: pd.DataFrame, window_sizes_hours: list[int]) -> pd.DataFrame:
    """Short-window mean minus long-window mean — a cheap trend proxy.

    Positive when the sensor is climbing away from its own recent baseline,
    which is what a degrading machine looks like before it looks extreme.
    """
    short, long = min(window_sizes_hours), max(window_sizes_hours)
    out = rolling[KEY_COLUMNS].copy()
    if short == long:
        return out
    for sensor in SENSOR_COLUMNS:
        out[f"{sensor}_mean_delta_{short}h_{long}h"] = (
            rolling[f"{sensor}_mean_{short}h"] - rolling[f"{sensor}_mean_{long}h"]
        )
    return out


def error_count_features(
    telemetry: pd.DataFrame,
    errors: pd.DataFrame,
    window_sizes_hours: list[int],
    *,
    error_codes: tuple[str, ...] = ERROR_CODES,
    closed: Closed = WINDOW_CLOSED,
) -> pd.DataFrame:
    """How many of each error code were logged in each trailing window.

    Errors are sparse events; they are laid onto the telemetry grid as zeros and
    ones first, so a window with no errors reads 0 rather than going missing.
    Codes outside ``error_codes`` are dropped with a warning rather than
    silently widening the matrix — see the note in :mod:`features.contract`.
    """
    frame = _require_sorted_keys(telemetry)
    grid = frame[KEY_COLUMNS].copy()

    if not errors.empty:
        unknown = sorted(set(errors["error_id"].dropna().unique()) - set(error_codes))
        if unknown:
            LOGGER.warning("Dropping %d error code(s) outside the contract: %s", len(unknown), unknown)
        counted = (
            errors[errors["error_id"].isin(error_codes)]
            .assign(_one=1)
            .pivot_table(index=KEY_COLUMNS, columns="error_id", values="_one", aggfunc="sum", observed=True)
            .reset_index()
        )
        grid = grid.merge(counted, on=KEY_COLUMNS, how="left")

    for code in error_codes:
        if code not in grid.columns:
            grid[code] = 0.0
    grid[list(error_codes)] = grid[list(error_codes)].fillna(0.0).astype("float64")

    indexed = grid.set_index("ts")
    blocks: list[pd.DataFrame] = []
    for window in window_sizes_hours:
        rolled = (
            indexed.groupby("machine_id", sort=False)[list(error_codes)]
            .rolling(f"{window}h", closed=closed)
            .sum()
        )
        rolled.columns = [f"{code}_count_{window}h" for code in rolled.columns]
        rolled[f"error_count_{window}h"] = rolled.sum(axis=1)
        blocks.append(rolled)

    combined = pd.concat(blocks, axis=1).reset_index()
    return combined.sort_values(KEY_COLUMNS, kind="stable", ignore_index=True)


def build_window_features(
    telemetry: pd.DataFrame,
    errors: pd.DataFrame,
    window_sizes_hours: list[int],
    *,
    lag_periods: tuple[int, ...] = LAG_PERIODS,
    closed: Closed = WINDOW_CLOSED,
) -> pd.DataFrame:
    """Every time-window feature, joined on ``(machine_id, ts)``."""
    rolling = rolling_sensor_features(telemetry, window_sizes_hours, closed=closed)
    trends = trend_features(rolling, window_sizes_hours)
    lags = lag_features(telemetry, lag_periods)
    error_counts = error_count_features(telemetry, errors, window_sizes_hours, closed=closed)

    raw = _require_sorted_keys(telemetry)[[*KEY_COLUMNS, *SENSOR_COLUMNS]]

    out = raw
    for block in (rolling, trends, lags, error_counts):
        out = out.merge(block, on=KEY_COLUMNS, how="left", validate="one_to_one")
    return out
