"""Leakage tests for the rolling window features.

The spike test is one of the two tests in this repository that will actually
catch a real bug (the other is parity). Everything here exists to prove one
claim: a feature at ``ts`` sees observations in ``(ts - window, ts]`` and
nothing after.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.windows import (
    build_window_features,
    error_count_features,
    lag_features,
    rolling_sensor_features,
    trend_features,
)
from tests.conftest import errors_frame, telemetry_frame

START = pd.Timestamp("2015-01-01 00:00:00")
WINDOWS = [3, 12, 24]


def at(hour: int) -> pd.Timestamp:
    return START + pd.Timedelta(hours=hour)


# --- The spike test --------------------------------------------------------


def test_a_future_spike_does_not_move_past_features() -> None:
    """Plant a spike at t; assert every feature at every earlier timestamp is
    bit-for-bit unchanged. This is the test that catches a centered window, a
    bfill, or an off-by-one in the window bound."""
    clean = telemetry_frame(hours=48)
    spiked = clean.copy()
    spike_row = spiked.index[spiked["ts"] == at(30)][0]
    spiked.loc[spike_row, "vibration"] = 195.0

    before = rolling_sensor_features(clean, WINDOWS)
    after = rolling_sensor_features(spiked, WINDOWS)

    past = before["ts"] < at(30)
    pd.testing.assert_frame_equal(before[past], after[past])


def test_the_spike_does_move_the_feature_at_its_own_timestamp() -> None:
    """The mirror of the test above, and the reason it is not vacuous: the
    window is closed on the right, so the reading at ts counts."""
    clean = telemetry_frame(hours=48)
    spiked = clean.copy()
    spiked.loc[spiked.index[spiked["ts"] == at(30)][0], "vibration"] = 195.0

    before = rolling_sensor_features(clean, WINDOWS).set_index("ts")
    after = rolling_sensor_features(spiked, WINDOWS).set_index("ts")
    assert after.loc[at(30), "vibration_max_3h"] == 195.0
    assert before.loc[at(30), "vibration_max_3h"] < 195.0


def test_the_spike_leaves_later_windows_once_it_ages_out() -> None:
    clean = telemetry_frame(hours=48)
    spiked = clean.copy()
    spiked.loc[spiked.index[spiked["ts"] == at(30)][0], "vibration"] = 195.0
    after = rolling_sensor_features(spiked, WINDOWS).set_index("ts")
    # The 3h window at t covers (t-3h, t], so the spike at hour 30 is the oldest
    # observation still inside the window at hour 32, and gone by hour 33.
    assert after.loc[at(32), "vibration_max_3h"] == 195.0
    assert after.loc[at(33), "vibration_max_3h"] < 195.0


# --- Trailing, not centered ------------------------------------------------


def test_the_window_is_trailing_not_centered() -> None:
    """A monotone ramp makes the difference obvious: a trailing 3h mean at t is
    the mean of t-2, t-1, t. A centered one would be the mean of t-1, t, t+1."""
    frame = telemetry_frame(hours=12)
    frame["volt"] = np.arange(12, dtype="float64") * 10.0

    rolling = rolling_sensor_features(frame, [3]).set_index("ts")
    # values at hours 3, 4, 5 are 30, 40, 50
    assert rolling.loc[at(5), "volt_mean_3h"] == pytest.approx((30.0 + 40.0 + 50.0) / 3)
    assert rolling.loc[at(5), "volt_max_3h"] == 50.0
    assert rolling.loc[at(5), "volt_min_3h"] == 30.0


def test_windows_are_time_based_not_row_based() -> None:
    """With a gap in the readings a row-based window silently reaches further
    back in time. A time-based one just sees fewer observations."""
    frame = telemetry_frame(hours=12)
    frame["volt"] = np.arange(12, dtype="float64") * 10.0
    with_gap = frame[~frame["ts"].isin([at(3), at(4)])].reset_index(drop=True)

    rolling = rolling_sensor_features(with_gap, [3]).set_index("ts")
    # (2, 5] contains only hour 5, because 3 and 4 are missing.
    assert rolling.loc[at(5), "volt_mean_3h"] == pytest.approx(50.0)


# --- Machine isolation -----------------------------------------------------


def test_one_machine_cannot_contaminate_another() -> None:
    frame = telemetry_frame(machine_ids=(1, 2), hours=30)
    spiked = frame.copy()
    mask = (spiked["machine_id"] == 1) & (spiked["ts"] == at(10))
    spiked.loc[mask, "vibration"] = 195.0

    before = rolling_sensor_features(frame, WINDOWS)
    after = rolling_sensor_features(spiked, WINDOWS)
    other = before["machine_id"] == 2
    pd.testing.assert_frame_equal(before[other], after[other])


def test_lags_do_not_bleed_across_the_machine_boundary() -> None:
    frame = telemetry_frame(machine_ids=(1, 2), hours=6)
    frame["volt"] = np.where(frame["machine_id"] == 1, 100.0, 200.0)

    lags = lag_features(frame, (1,)).merge(frame[["machine_id", "ts"]], on=["machine_id", "ts"])
    first_of_machine_two = lags[(lags["machine_id"] == 2) & (lags["ts"] == at(0))]
    assert first_of_machine_two["volt_lag_1"].isna().all()


# --- Lags ------------------------------------------------------------------


def test_lags_reach_backwards_only() -> None:
    frame = telemetry_frame(hours=8)
    frame["volt"] = np.arange(8, dtype="float64")

    lags = lag_features(frame, (1, 3)).set_index("ts")
    assert lags.loc[at(5), "volt_lag_1"] == 4.0
    assert lags.loc[at(5), "volt_lag_3"] == 2.0
    assert pd.isna(lags.loc[at(0), "volt_lag_1"])


# --- Trends ----------------------------------------------------------------


def test_the_trend_is_short_minus_long() -> None:
    frame = telemetry_frame(hours=36)
    frame["volt"] = np.arange(36, dtype="float64")

    rolling = rolling_sensor_features(frame, WINDOWS)
    trends = trend_features(rolling, WINDOWS).set_index("ts")
    combined = rolling.set_index("ts")
    expected = combined.loc[at(30), "volt_mean_3h"] - combined.loc[at(30), "volt_mean_24h"]
    assert trends.loc[at(30), "volt_mean_delta_3h_24h"] == pytest.approx(expected)
    assert expected > 0  # a rising sensor pulls the short window above the long one


def test_a_single_window_yields_no_trend_columns() -> None:
    frame = telemetry_frame(hours=12)
    rolling = rolling_sensor_features(frame, [3])
    trends = trend_features(rolling, [3])
    assert list(trends.columns) == ["machine_id", "ts"]


# --- Error counts ----------------------------------------------------------


def test_error_counts_are_windowed_and_sparse_windows_read_zero() -> None:
    frame = telemetry_frame(hours=24)
    errors = errors_frame(
        [
            (1, "2015-01-01 05:00:00", "error1"),
            (1, "2015-01-01 06:00:00", "error1"),
            (1, "2015-01-01 06:00:00", "error2"),
        ]
    )
    counts = error_count_features(frame, errors, [3, 12]).set_index("ts")

    assert counts.loc[at(6), "error1_count_3h"] == 2.0
    assert counts.loc[at(6), "error2_count_3h"] == 1.0
    assert counts.loc[at(6), "error_count_3h"] == 3.0
    assert counts.loc[at(0), "error_count_12h"] == 0.0  # before any error
    assert counts.loc[at(20), "error_count_12h"] == 0.0  # long after


def test_error_counts_do_not_see_the_future() -> None:
    frame = telemetry_frame(hours=24)
    errors = errors_frame([(1, "2015-01-01 20:00:00", "error1")])
    counts = error_count_features(frame, errors, [3, 12]).set_index("ts")
    assert counts.loc[at(19), "error_count_12h"] == 0.0
    assert counts.loc[at(20), "error_count_12h"] == 1.0


def test_error_codes_outside_the_contract_are_dropped_loudly(caplog: pytest.LogCaptureFixture) -> None:
    """Widening the matrix because a new code appeared would break the hash at
    inference. Dropping it is visible; widening is not."""
    frame = telemetry_frame(hours=6)
    errors = errors_frame([(1, "2015-01-01 02:00:00", "error9")])
    with caplog.at_level("WARNING"):
        counts = error_count_features(frame, errors, [3])
    assert "error9" in caplog.text
    assert "error9_count_3h" not in counts.columns
    assert counts["error_count_3h"].sum() == 0.0


def test_no_errors_at_all_still_produces_zero_columns() -> None:
    counts = error_count_features(telemetry_frame(hours=6), errors_frame(), [3, 12])
    assert counts["error1_count_3h"].eq(0.0).all()
    assert counts["error_count_12h"].eq(0.0).all()


# --- The combined block ----------------------------------------------------


def test_build_window_features_joins_one_to_one() -> None:
    frame = telemetry_frame(machine_ids=(1, 2), hours=30)
    errors = errors_frame([(1, "2015-01-01 05:00:00", "error1")])
    combined = build_window_features(frame, errors, WINDOWS)

    assert len(combined) == len(frame)
    assert not combined.duplicated(subset=["machine_id", "ts"]).any()
    for name in ("volt_mean_3h", "volt_lag_1", "volt_mean_delta_3h_24h", "error1_count_3h"):
        assert name in combined.columns
