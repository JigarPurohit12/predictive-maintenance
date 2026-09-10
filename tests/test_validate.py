"""Validation has to catch corrupted frames, and has to not cry wolf on clean ones."""

from __future__ import annotations

import pandas as pd
import pytest

from data_layer.validate import (
    ValidationError,
    check_machines_without_failures,
    validate_all,
    validate_events,
    validate_telemetry,
)
from tests.conftest import failures_frame, machines_frame, raw_frames, telemetry_frame


def _issue_checks(report) -> set[str]:
    return {issue.check for issue in report.issues}


def test_clean_telemetry_produces_no_issues() -> None:
    report = validate_telemetry(telemetry_frame(hours=24))
    assert report.issues == ()
    assert report.ok


def test_duplicate_machine_timestamp_pairs_are_blocking() -> None:
    frame = telemetry_frame(hours=4)
    frame = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)
    report = validate_telemetry(frame)
    assert "duplicates" in _issue_checks(report)
    assert not report.ok
    with pytest.raises(ValidationError, match="duplicates"):
        report.raise_for_errors()


def test_out_of_range_sensor_values_are_blocking() -> None:
    frame = telemetry_frame(hours=4)
    frame.loc[1, "vibration"] = 5_000.0
    report = validate_telemetry(frame)
    assert "range" in _issue_checks(report)
    assert not report.ok
    issue = next(issue for issue in report.issues if issue.check == "range")
    assert "vibration" in issue.detail
    assert issue.sample  # the offending value is carried for debugging


def test_timestamp_gaps_warn_but_do_not_block() -> None:
    """Gaps are survivable — time-based windows tolerate them — but visible."""
    frame = telemetry_frame(hours=8).drop(index=[3, 4]).reset_index(drop=True)
    report = validate_telemetry(frame)
    assert "timestamp_gaps" in _issue_checks(report)
    assert report.ok
    assert report.warnings and not report.errors


def test_gaps_are_measured_within_a_machine_not_across_the_frame() -> None:
    """Two machines interleaved in one frame must not look like a gap each."""
    frame = telemetry_frame(machine_ids=(1, 2), hours=6)
    report = validate_telemetry(frame)
    assert "timestamp_gaps" not in _issue_checks(report)


def test_unparsed_timestamps_are_blocking() -> None:
    frame = telemetry_frame(hours=4)
    frame.loc[2, "ts"] = pd.NaT
    report = validate_telemetry(frame)
    assert "timestamp_parse" in _issue_checks(report) or "nulls" in _issue_checks(report)
    assert not report.ok


def test_events_referencing_an_unknown_machine_are_blocking() -> None:
    failures = failures_frame([(99, "2015-01-01 06:00:00", "comp1")])
    report = validate_events(failures, "failures", ("machine_id", "ts", "component"), known_machine_ids=[1, 2])
    assert "known_machines" in _issue_checks(report)
    assert not report.ok


def test_machines_that_never_fail_are_a_warning_not_an_error() -> None:
    """They are normal, they stay in the data, and a machine-held-out split
    needs to know they exist before it lands every failure on one side."""
    issues = check_machines_without_failures(
        failures_frame([(1, "2015-01-01 06:00:00", "comp1")]),
        known_machine_ids=[1, 2, 3],
    )
    assert len(issues) == 1
    assert issues[0].severity == "WARN"
    assert issues[0].count == 2


def test_missing_required_columns_are_reported_by_name() -> None:
    report = validate_telemetry(telemetry_frame(hours=2).drop(columns=["pressure"]))
    issue = next(issue for issue in report.issues if issue.check == "required_columns")
    assert "pressure" in issue.detail


def test_validate_all_runs_every_table_and_the_cross_table_checks() -> None:
    frames = raw_frames(
        telemetry=telemetry_frame(machine_ids=(1, 2), hours=6),
        failures=failures_frame([(1, "2015-01-01 03:00:00", "comp1")]),
        machines=machines_frame((1, 2)),
    )
    report = validate_all(frames)
    assert report.ok
    # Machine 2 never fails, which is a warning and nothing more.
    assert _issue_checks(report) == {"machines_without_failures"}


def test_validate_all_surfaces_every_problem_at_once() -> None:
    telemetry = telemetry_frame(hours=6)
    telemetry.loc[0, "volt"] = 9_999.0
    frames = raw_frames(
        telemetry=telemetry,
        failures=failures_frame([(42, "2015-01-01 03:00:00", "comp1")]),
    )
    report = validate_all(frames)
    assert {"range", "known_machines"} <= _issue_checks(report)
    assert len(report.errors) >= 2
