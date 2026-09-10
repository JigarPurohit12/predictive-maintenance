"""The labelling rules, asserted against the production SQL on hand-built frames.

There is no second Python implementation of these rules to test against — the
tests register ten-row pandas frames as the ``raw_*`` relations and run
``sql/02_failure_labels.sql`` over them, which is the same code path the real
run takes. A rule that drifts breaks a test here rather than quietly changing
the base rate three phases later.

Fixture geometry, unless a test overrides it: machine 1, hourly telemetry from
2015-01-01 00:00 to 11:00, horizon 3h, cadence 1h, post-failure exclusion 2h.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from config import Settings
from features.labeling import (
    build,
    class_balance_by_component,
    drop_accounting,
    format_report,
    load_labels,
    load_training_labels,
    summarise,
    write_labels,
)
from tests.conftest import failures_frame, make_settings, raw_frames, telemetry_frame

START = "2015-01-01 00:00:00"


def at(hour: float) -> pd.Timestamp:
    return pd.Timestamp(START) + pd.Timedelta(hours=hour)


def labelled(settings: Settings, failures: pd.DataFrame, *, hours: int = 12, machine_ids=(1,)):
    frames = raw_frames(telemetry=telemetry_frame(machine_ids=machine_ids, start=START, hours=hours), failures=failures)
    return build(settings, frames)


@pytest.fixture
def one_failure(settings: Settings):
    """A single comp1 failure at 06:00 on an otherwise clean machine."""
    return labelled(settings, failures_frame([(1, "2015-01-01 06:00:00", "comp1")]))


# --- The horizon -----------------------------------------------------------


def test_horizon_window_is_open_on_the_left_and_closed_on_the_right(one_failure) -> None:
    """Positive when a failure falls in (ts, ts + horizon].

    With the failure at 06:00 and a 3h horizon, 03:00 is the earliest positive
    (exactly 3h out) and 02:00 is negative (4h out).
    """
    labels = load_labels(one_failure).set_index("ts")["label"]
    assert labels[at(2)] == 0, "4h before the failure is outside a 3h horizon"
    assert labels[at(3)] == 1, "exactly one horizon out must be positive"
    assert labels[at(4)] == 1
    assert labels[at(5)] == 1


def test_the_failure_moment_itself_is_not_a_positive(one_failure) -> None:
    """A model scoring at the instant of failure has nothing left to warn about."""
    labels = load_labels(one_failure).set_index("ts")["label"]
    assert labels[at(6)] == 0


def test_rows_far_from_any_failure_are_negative(one_failure) -> None:
    labels = load_labels(one_failure).set_index("ts")["label"]
    assert labels[at(0)] == 0
    assert labels[at(1)] == 0


# --- The exclusion window --------------------------------------------------


def test_post_failure_rows_are_dropped_not_labelled_zero(one_failure) -> None:
    """The single most consequential rule in phase 1. A machine that has just
    had a component replaced looks nothing like a healthy machine; leaving those
    rows in as negatives teaches the model to detect repairs."""
    labels = load_labels(one_failure).set_index("ts")
    training = load_training_labels(one_failure).set_index("ts")

    for hour in (6, 7, 8):  # [failure, failure + 2h]
        assert bool(labels.loc[at(hour), "excluded"]) is True
        assert at(hour) not in training.index, f"{at(hour)} should be dropped, not present at all"


def test_the_exclusion_window_ends_where_it_says_it_does(settings: Settings) -> None:
    """09:00 is 3h after the failure and the window is 2h, so it comes back."""
    con = labelled(settings, failures_frame([(1, "2015-01-01 06:00:00", "comp1")]), hours=16)
    labels = load_labels(con).set_index("ts")
    assert bool(labels.loc[at(9), "excluded"]) is False
    assert at(9) in load_training_labels(con)["ts"].values


def test_exclusion_is_per_machine(settings: Settings) -> None:
    """Machine 2 has no failure and must keep every row."""
    con = labelled(
        settings,
        failures_frame([(1, "2015-01-01 06:00:00", "comp1")]),
        machine_ids=(1, 2),
    )
    labels = load_labels(con)
    machine_two = labels[labels["machine_id"] == 2]
    assert not machine_two["excluded"].any()
    assert machine_two["label"].sum() == 0


# --- Observability ---------------------------------------------------------


def test_the_last_horizon_of_data_cannot_be_labelled(one_failure) -> None:
    """A row at ts is only a trustworthy negative once (ts, ts+horizon] has been
    observed. Telemetry ends at 11:00, so 09:00 onwards is unlabelable."""
    labels = load_labels(one_failure).set_index("ts")
    assert bool(labels.loc[at(8), "labelable"]) is True
    for hour in (9, 10, 11):
        assert bool(labels.loc[at(hour), "labelable"]) is False

    training_timestamps = set(load_training_labels(one_failure)["ts"])
    assert training_timestamps.isdisjoint({at(9), at(10), at(11)})


def test_the_surviving_rows_are_exactly_the_expected_set(one_failure) -> None:
    """Horizon, exclusion and observability composed: 00:00-05:00 survive,
    06:00-08:00 are excluded, 09:00-11:00 are unlabelable."""
    training = load_training_labels(one_failure)
    assert list(training["ts"]) == [at(hour) for hour in range(6)]
    assert training["label"].tolist() == [0, 0, 0, 1, 1, 1]


# --- The grid --------------------------------------------------------------


def test_the_grid_follows_the_scoring_cadence(tmp_path: Path) -> None:
    """Cadence 3 keeps every third hour, anchored to the epoch rather than to
    each machine's first reading, so every machine lands on the same ticks."""
    settings = make_settings(tmp_path, feature_cadence_hours=3)
    con = labelled(settings, failures_frame(), hours=12)
    timestamps = load_labels(con)["ts"].tolist()
    assert timestamps == [at(0), at(3), at(6), at(9)]


def test_a_finer_cadence_keeps_every_hour(one_failure) -> None:
    assert len(load_labels(one_failure)) == 12


def test_the_grid_is_identical_across_machines(settings: Settings) -> None:
    con = labelled(settings, failures_frame(), machine_ids=(1, 2, 3))
    labels = load_labels(con)
    per_machine = labels.groupby("machine_id")["ts"].apply(list)
    assert per_machine[1] == per_machine[2] == per_machine[3]


# --- Multiple components ---------------------------------------------------


def test_simultaneous_component_failures_produce_one_row_not_three(settings: Settings) -> None:
    """Three components failing together is one event and one repair window."""
    failures = failures_frame(
        [
            (1, "2015-01-01 06:00:00", "comp1"),
            (1, "2015-01-01 06:00:00", "comp2"),
            (1, "2015-01-01 06:00:00", "comp3"),
        ]
    )
    con = labelled(settings, failures)
    training = load_training_labels(con)
    assert not training.duplicated(subset=["machine_id", "ts"]).any()
    assert training["label"].tolist() == [0, 0, 0, 1, 1, 1]


def test_per_component_labels_split_the_target_without_changing_the_rows(settings: Settings) -> None:
    failures = failures_frame(
        [
            (1, "2015-01-01 06:00:00", "comp1"),
            (1, "2015-01-01 05:00:00", "comp2"),
        ]
    )
    con = labelled(settings, failures)
    by_component = class_balance_by_component(con).set_index("component")
    # comp2 fails at 05:00, so 02:00-04:00 are its positives; comp1 at 06:00
    # covers 03:00-05:00 — but 05:00 onwards is inside comp2's repair window and
    # has already been dropped.
    assert by_component.loc["comp1", "positives"] == 2  # 03:00 and 04:00 survive
    assert by_component.loc["comp2", "positives"] == 3  # 02:00, 03:00, 04:00
    assert set(by_component["rows"]) == {5}


# --- Splits ----------------------------------------------------------------


@pytest.fixture
def three_day_settings(tmp_path: Path) -> Settings:
    """Train ends 1 Jan, validation ends 2 Jan, with a 3h gap either side."""
    return make_settings(
        tmp_path,
        train_end=date(2015, 1, 1),
        val_end=date(2015, 1, 2),
        prediction_horizon_hours=3,
        gap_hours=3,
    )


def test_gap_rows_belong_to_no_split_and_are_dropped(three_day_settings: Settings) -> None:
    """Without the buffer, the last training rows carry labels that depend on
    failures inside the validation window."""
    con = labelled(three_day_settings, failures_frame(), hours=80)
    labels = load_labels(con).set_index("ts")

    for hour in (24, 25, 26):  # 2015-01-02 00:00, 01:00, 02:00
        assert labels.loc[at(hour), "split"] == "gap"
    assert labels.loc[at(23), "split"] == "train"
    assert labels.loc[at(27), "split"] == "val"

    training = set(load_training_labels(con)["ts"])
    assert training.isdisjoint({at(24), at(25), at(26)})


def test_split_ranges_do_not_overlap(three_day_settings: Settings) -> None:
    con = labelled(three_day_settings, failures_frame(), hours=80)
    training = load_training_labels(con)
    bounds = training.groupby("split")["ts"].agg(["min", "max"])
    assert bounds.loc["train", "max"] < bounds.loc["val", "min"]
    assert bounds.loc["val", "max"] < bounds.loc["test", "min"]


def test_the_gap_between_splits_is_at_least_one_horizon(three_day_settings: Settings) -> None:
    con = labelled(three_day_settings, failures_frame(), hours=80)
    bounds = load_training_labels(con).groupby("split")["ts"].agg(["min", "max"])
    horizon = pd.Timedelta(hours=three_day_settings.prediction_horizon_hours)
    assert bounds.loc["val", "min"] - bounds.loc["train", "max"] > horizon
    assert bounds.loc["test", "min"] - bounds.loc["val", "max"] > horizon


# --- The report ------------------------------------------------------------


def test_drop_accounting_adds_up(one_failure) -> None:
    counts = drop_accounting(one_failure)
    assert counts["grid_rows"] == 12
    assert counts["dropped_excluded"] == 3  # 06:00, 07:00, 08:00
    assert counts["dropped_unlabelable"] == 3  # 09:00, 10:00, 11:00
    assert counts["dropped_gap"] == 0
    assert counts["kept_rows"] == 6


def test_summary_reports_the_base_rate(settings: Settings, one_failure) -> None:
    summary = summarise(one_failure, settings)
    assert summary.kept_rows == 6
    assert summary.positives == 3
    assert summary.positive_rate == pytest.approx(0.5)
    # 50% is nowhere near a real predictive-maintenance base rate, and the gate
    # is what stops that sailing through into phase 2.
    assert summary.within_expected_range is False
    assert "WARNING" in format_report(summary)


def test_report_renders_without_a_single_failure(settings: Settings) -> None:
    """An all-negative fixture must not blow up the pivot or the formatting."""
    con = labelled(settings, failures_frame())
    summary = summarise(con, settings)
    assert summary.positives == 0
    assert summary.by_component == ()
    assert "Class balance per split" in format_report(summary)


def test_written_labels_are_partitioned_by_split(settings: Settings, one_failure) -> None:
    destination = write_labels(one_failure, settings)
    partitions = sorted(path.name for path in destination.iterdir() if path.is_dir())
    assert partitions == ["split=train"]
    written = pd.read_parquet(destination)
    assert len(written) == 6
    assert set(written.columns) >= {"machine_id", "ts", "label"}
