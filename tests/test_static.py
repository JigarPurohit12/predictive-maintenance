"""Machine metadata and the maintenance clocks."""

from __future__ import annotations

import pandas as pd
import pytest

from features.static import build_static_features, machine_features, maintenance_clocks
from tests.conftest import machines_frame, maint_frame, telemetry_frame

START = pd.Timestamp("2015-01-01 00:00:00")


def at(hour: int) -> pd.Timestamp:
    return START + pd.Timedelta(hours=hour)


def grid(hours: int = 24, machine_ids=(1,)) -> pd.DataFrame:
    return telemetry_frame(machine_ids=machine_ids, hours=hours)[["machine_id", "ts"]]


# --- Machine metadata ------------------------------------------------------


def test_model_is_one_hot_not_ordinal() -> None:
    """model3 is not "more" than model1; a tree given an ordinal encoding will
    split on an ordering that does not exist."""
    machines = pd.DataFrame({"machine_id": [1], "model": ["model3"], "age": [18]})
    out = machine_features(grid(4), machines)
    assert out["model_model3"].eq(1.0).all()
    assert out["model_model1"].eq(0.0).all()
    assert out["machine_age"].eq(18.0).all()


def test_an_unknown_model_produces_all_zero_indicators(caplog: pytest.LogCaptureFixture) -> None:
    machines = pd.DataFrame({"machine_id": [1], "model": ["model9"], "age": [5]})
    with caplog.at_level("WARNING"):
        out = machine_features(grid(4), machines)
    assert "model9" in caplog.text
    assert out[[column for column in out.columns if column.startswith("model_")]].to_numpy().sum() == 0.0


def test_a_grid_row_for_an_unknown_machine_is_an_error() -> None:
    machines = pd.DataFrame({"machine_id": [2], "model": ["model1"], "age": [5]})
    with pytest.raises(ValueError, match="absent from the machines table"):
        machine_features(grid(4), machines)


# --- Maintenance clocks ----------------------------------------------------


def test_the_clock_counts_up_from_the_last_replacement() -> None:
    maint = maint_frame([(1, "2015-01-01 05:00:00", "comp1")])
    clocks = maintenance_clocks(grid(12), maint).set_index("ts")
    assert clocks.loc[at(5), "hours_since_maint_comp1"] == 0.0
    assert clocks.loc[at(8), "hours_since_maint_comp1"] == 3.0
    assert clocks.loc[at(11), "hours_since_maint_comp1"] == 6.0


def test_the_clock_resets_at_the_next_replacement() -> None:
    maint = maint_frame([(1, "2015-01-01 02:00:00", "comp1"), (1, "2015-01-01 08:00:00", "comp1")])
    clocks = maintenance_clocks(grid(12), maint).set_index("ts")
    assert clocks.loc[at(7), "hours_since_maint_comp1"] == 5.0
    assert clocks.loc[at(8), "hours_since_maint_comp1"] == 0.0
    assert clocks.loc[at(9), "hours_since_maint_comp1"] == 1.0


def test_a_future_replacement_does_not_touch_earlier_rows() -> None:
    """merge_asof in backward direction cannot reach forward, and this is the
    test that says so out loud."""
    without = maintenance_clocks(grid(12), maint_frame([(1, "2015-01-01 02:00:00", "comp1")]))
    with_future = maintenance_clocks(
        grid(12),
        maint_frame([(1, "2015-01-01 02:00:00", "comp1"), (1, "2015-01-01 09:00:00", "comp1")]),
    )
    early = without["ts"] < at(9)
    pd.testing.assert_series_equal(
        without.loc[early, "hours_since_maint_comp1"],
        with_future.loc[early, "hours_since_maint_comp1"],
    )


def test_a_component_never_serviced_is_censored_not_null() -> None:
    """All we know is "at least this long", which is true and leakage-free.
    Filling it from a later record would not be."""
    clocks = maintenance_clocks(grid(12), maint_frame([(1, "2015-01-01 03:00:00", "comp1")]))
    indexed = clocks.set_index("ts")
    assert not indexed["hours_since_maint_comp2"].isna().any()
    assert indexed.loc[at(0), "hours_since_maint_comp2"] == 0.0
    assert indexed.loc[at(7), "hours_since_maint_comp2"] == 7.0


def test_with_no_maintenance_records_at_all_every_clock_is_censored() -> None:
    clocks = maintenance_clocks(grid(6), maint_frame()).set_index("ts")
    assert not clocks.isna().to_numpy().any()
    assert clocks.loc[at(4), "hours_since_maint_comp1"] == 4.0


def test_the_any_clock_is_the_most_recently_serviced_component() -> None:
    maint = maint_frame([(1, "2015-01-01 01:00:00", "comp1"), (1, "2015-01-01 06:00:00", "comp3")])
    clocks = maintenance_clocks(grid(12), maint).set_index("ts")
    assert clocks.loc[at(8), "hours_since_maint_comp1"] == 7.0
    assert clocks.loc[at(8), "hours_since_maint_comp3"] == 2.0
    assert clocks.loc[at(8), "hours_since_maint_any"] == 2.0


def test_clocks_are_tracked_per_machine() -> None:
    maint = maint_frame([(1, "2015-01-01 04:00:00", "comp1")])
    clocks = maintenance_clocks(grid(8, machine_ids=(1, 2)), maint)
    machine_two = clocks[clocks["machine_id"] == 2].set_index("ts")
    # Machine 2 has no record of its own, so its clock is censored at its first
    # observation rather than picking up machine 1's replacement.
    assert machine_two.loc[at(6), "hours_since_maint_comp1"] == 6.0


# --- The combined block ----------------------------------------------------


def test_build_static_features_joins_one_to_one() -> None:
    maint = maint_frame([(1, "2015-01-01 02:00:00", "comp1")])
    out = build_static_features(grid(12, machine_ids=(1, 2)), machines_frame((1, 2)), maint)
    assert len(out) == 24
    assert not out.duplicated(subset=["machine_id", "ts"]).any()
    assert not out.isna().to_numpy().any()
