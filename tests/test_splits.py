"""Time-based splits and the checks that prove they are what they claim."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import Settings
from training.splits import (
    SplitViolation,
    describe_splits,
    format_splits,
    holdout_machines,
    machine_overlap,
    split_frames,
    verify_time_splits,
    xy,
)

START = pd.Timestamp("2015-01-01 00:00:00")


def frame_with_splits(gap_hours: int = 24, overlap: bool = False) -> pd.DataFrame:
    """train: days 0-3, val: days 5-7, test: days 9-11, with a day of gap."""
    rows: list[dict[str, object]] = []
    spans = {"train": (0, 96), "val": (120, 192), "test": (216, 288)}
    if overlap:
        spans["val"] = (90, 192)
    for split, (start_hour, end_hour) in spans.items():
        for hour in range(start_hour, end_hour, 3):
            rows.append(
                {
                    "machine_id": (hour // 3) % 4 + 1,
                    "ts": START + pd.Timedelta(hours=hour),
                    "label": 1 if hour % 51 == 0 else 0,
                    "split": split,
                    "volt_mean_3h": float(hour),
                    "vibration_std_24h": float(hour) / 2,
                }
            )
    _ = gap_hours
    return pd.DataFrame(rows)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, prediction_horizon_hours=24, gap_hours=24)


# --- Description -----------------------------------------------------------


def test_describe_splits_reports_each_partition(settings: Settings) -> None:
    bounds = describe_splits(frame_with_splits())
    assert [bound.split for bound in bounds] == ["train", "val", "test"]
    assert all(bound.rows > 0 for bound in bounds)
    assert bounds[0].ts_min == START


def test_format_splits_renders_without_a_positive() -> None:
    frame = frame_with_splits()
    frame["label"] = 0
    rendered = format_splits(describe_splits(frame))
    assert "train" in rendered and "0.0000%" in rendered


# --- Verification ----------------------------------------------------------


def test_a_well_formed_split_passes(settings: Settings) -> None:
    bounds = verify_time_splits(frame_with_splits(), settings)
    assert len(bounds) == 3


def test_overlapping_ranges_are_rejected(settings: Settings) -> None:
    with pytest.raises(SplitViolation, match="overlap"):
        verify_time_splits(frame_with_splits(overlap=True), settings)


def test_a_gap_narrower_than_the_horizon_is_rejected(settings: Settings) -> None:
    """The whole point of the buffer: without a full horizon between them, the
    last training rows are labelled from validation-window data."""
    frame = frame_with_splits()
    # Move validation to start three hours after training ends.
    frame.loc[frame["split"] == "val", "ts"] = frame.loc[frame["split"] == "val", "ts"] - pd.Timedelta(hours=24)
    with pytest.raises(SplitViolation, match="does not clear"):
        verify_time_splits(frame, settings)


def test_gap_rows_must_already_be_dropped(settings: Settings) -> None:
    frame = frame_with_splits()
    frame.loc[0, "split"] = "gap"
    with pytest.raises(SplitViolation, match="Unexpected split label"):
        verify_time_splits(frame, settings)


def test_a_configured_gap_shorter_than_the_horizon_is_rejected() -> None:
    """config.py refuses to build such a Settings, so this guards the path
    where a frame is verified against settings constructed elsewhere."""
    settings = Settings(_env_file=None, prediction_horizon_hours=24, gap_hours=24)
    loosened = settings.model_copy(update={"gap_hours": 1})
    with pytest.raises(SplitViolation, match="shorter than the horizon"):
        verify_time_splits(frame_with_splits(), loosened)


def test_verification_tolerates_a_missing_partition(settings: Settings) -> None:
    frame = frame_with_splits()
    bounds = verify_time_splits(frame[frame["split"] != "test"], settings)
    assert [bound.split for bound in bounds] == ["train", "val"]


# --- Machine overlap -------------------------------------------------------


def test_machine_overlap_is_measured_not_assumed() -> None:
    """Overlap is expected for a purely time-based split — the same fleet runs
    throughout. Measuring it is what keeps the generalisation claim honest."""
    overlap = machine_overlap(frame_with_splits())
    assert overlap[("train", "val")] == 4
    assert overlap[("train", "test")] == 4


def test_holding_out_machines_removes_them_from_training_entirely() -> None:
    frame = frame_with_splits()
    holdout = holdout_machines(frame, fraction=0.5, seed=1)

    assert holdout.held_out
    trained_on = set(holdout.frame.loc[holdout.frame["split"].isin(("train", "val")), "machine_id"])
    assert not trained_on & set(holdout.held_out)


def test_test_keeps_seen_machines_until_you_ask_for_unseen_only() -> None:
    """Two different populations end up in test, answering two different
    questions. The stricter one has to be asked for explicitly."""
    frame = frame_with_splits()
    holdout = holdout_machines(frame, fraction=0.5, seed=1)

    assert machine_overlap(holdout.frame)[("train", "test")] > 0
    strict = holdout.unseen_only()
    assert machine_overlap(strict)[("train", "test")] == 0
    assert set(strict.loc[strict["split"] == "test", "machine_id"]) == set(holdout.held_out)


def test_the_holdout_is_stable_across_runs() -> None:
    frame = frame_with_splits()
    first = holdout_machines(frame, fraction=0.5, seed=3)
    second = holdout_machines(frame, fraction=0.5, seed=3)
    assert first.held_out == second.held_out
    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_the_holdout_is_stratified_on_whether_a_machine_ever_fails() -> None:
    """With ~8 failures across a fleet of 100, an unstratified draw can
    plausibly take every failing machine or none of them."""
    rows = []
    for machine_id in range(1, 21):
        for hour in range(0, 96, 3):
            rows.append(
                {
                    "machine_id": machine_id,
                    "ts": START + pd.Timedelta(hours=hour),
                    "label": 1 if (machine_id <= 4 and hour == 0) else 0,
                    "split": "train",
                    "volt_mean_3h": 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    held_ids = set(holdout_machines(frame, fraction=0.25, seed=0).held_out)
    failing = {1, 2, 3, 4}
    assert held_ids & failing, "at least one failing machine must be held out"
    assert held_ids - failing, "and at least one non-failing machine"


def test_an_out_of_range_fraction_is_rejected() -> None:
    with pytest.raises(ValueError, match="fraction must be"):
        holdout_machines(frame_with_splits(), fraction=1.5)


# --- Carving up ------------------------------------------------------------


def test_split_frames_partitions_without_losing_rows() -> None:
    frame = frame_with_splits()
    parts = split_frames(frame)
    assert set(parts) == {"train", "val", "test"}
    assert sum(len(part) for part in parts.values()) == len(frame)


def test_xy_returns_features_in_the_order_given() -> None:
    frame = frame_with_splits()
    order = ["vibration_std_24h", "volt_mean_3h"]
    features, labels = xy(frame, order)

    assert features.shape == (len(frame), 2)
    assert labels.dtype == np.int8
    np.testing.assert_array_equal(features[:, 0], frame["vibration_std_24h"].to_numpy())
    np.testing.assert_array_equal(features[:, 1], frame["volt_mean_3h"].to_numpy())
