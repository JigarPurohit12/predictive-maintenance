"""Parity: the training path and the scoring path must produce the same numbers.

Training/serving skew is the failure mode that does not announce itself. If
features are computed one way for training and another way for scoring, the two
drift apart and predictions quietly degrade — nobody sees an error, the model
just stops working. There is one feature module precisely so this test can be
written, and it is one of the two tests here that will catch a real bug.

The second parity concern is the scoring grid, which exists twice by necessity:
in ``sql/02_failure_labels.sql`` for training labels, and in
``features/build_features.py`` for serving, where there is no SQL engine. This
module asserts the two agree on real timestamps rather than trusting the comment
that says they do.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from config import Settings
from data_layer import ingest as ingest_module
from data_layer.duckdb_io import load_staged, prepare
from features.build_features import build_scoring_matrix, build_training_matrix, compute_features, on_cadence
from features.contract import build_contract
from tests.conftest import pipeline_settings, synthetic_dataset, write_landing


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = pipeline_settings(tmp_path)
    write_landing(settings.landing_dir, synthetic_dataset())
    monkeypatch.setattr("features.build_features.get_settings", lambda: settings)
    ingest_module.ingest(settings)
    return settings


# --- Grid parity -----------------------------------------------------------


def test_the_python_grid_matches_the_sql_grid(project: Settings) -> None:
    """`on_cadence` in pandas and the modulo in DuckDB must select exactly the
    same timestamps, or the model is scored on a grid it was never trained on."""
    con = prepare(project)
    sql_grid = con.execute("SELECT machine_id, ts FROM feature_grid ORDER BY machine_id, ts").df()

    telemetry = load_staged(con)["telemetry"]
    keep = on_cadence(telemetry["ts"], project.feature_cadence_hours)
    python_grid = telemetry.loc[keep, ["machine_id", "ts"]].sort_values(
        ["machine_id", "ts"], ignore_index=True
    )

    assert not sql_grid.empty
    pd.testing.assert_frame_equal(sql_grid, python_grid, check_dtype=False)


@pytest.mark.parametrize("cadence", [1, 2, 3, 4, 6, 12])
def test_the_two_grids_agree_at_every_cadence(project: Settings, cadence: int) -> None:
    settings = project.model_copy(update={"feature_cadence_hours": cadence})
    con = prepare(settings)
    sql_hours = {row[0] for row in con.execute("SELECT DISTINCT ts FROM feature_grid").fetchall()}

    telemetry = load_staged(con)["telemetry"]
    python_hours = set(telemetry.loc[on_cadence(telemetry["ts"], cadence), "ts"])
    assert sql_hours == python_hours


# --- Feature parity --------------------------------------------------------


def test_training_and_scoring_features_are_identical(project: Settings) -> None:
    """Same input, both paths, same numbers. The whole reason build_features is
    one module rather than one per execution path."""
    contract = build_contract(project)
    con = prepare(project)
    staged = load_staged(con)

    training = build_training_matrix(project, contract)
    scoring = build_scoring_matrix(
        staged["telemetry"], staged["errors"], staged["maint"], staged["machines"], project, contract
    )

    keys = ["machine_id", "ts"]
    shared = training[keys].merge(scoring[keys], on=keys, how="inner")
    assert len(shared) > 100, "the two paths should overlap on most of the grid"

    left = training.merge(shared, on=keys).sort_values(keys, ignore_index=True)
    right = scoring.merge(shared, on=keys).sort_values(keys, ignore_index=True)
    pd.testing.assert_frame_equal(
        left[list(contract.order)], right[list(contract.order)], check_exact=False, rtol=1e-12
    )


def test_scoring_as_of_returns_the_latest_tick_per_machine(project: Settings) -> None:
    contract = build_contract(project)
    staged = load_staged(prepare(project))
    as_of = pd.Timestamp("2015-01-08 14:00:00")

    scoring = build_scoring_matrix(
        staged["telemetry"], staged["errors"], staged["maint"], staged["machines"],
        project, contract, as_of=as_of,
    )
    assert len(scoring) == staged["telemetry"]["machine_id"].nunique()
    assert (scoring["ts"] <= as_of).all()
    # 14:00 is not on a 3h grid anchored to the epoch; 12:00 is.
    assert scoring["ts"].eq(pd.Timestamp("2015-01-08 12:00:00")).all()


def test_a_scoring_run_sees_the_same_values_as_the_full_history_run(project: Settings) -> None:
    """Serving passes a slice of recent history, not the whole year. The slice
    must produce the same features as the full run for the rows it covers."""
    contract = build_contract(project)
    staged = load_staged(prepare(project))
    as_of = pd.Timestamp("2015-01-08 12:00:00")
    window_start = as_of - timedelta(hours=project.max_window_hours * 2)

    full = build_scoring_matrix(
        staged["telemetry"], staged["errors"], staged["maint"], staged["machines"],
        project, contract, as_of=as_of,
    )
    sliced = build_scoring_matrix(
        staged["telemetry"][staged["telemetry"]["ts"] >= window_start],
        staged["errors"][staged["errors"]["ts"] >= window_start],
        staged["maint"],
        staged["machines"],
        project, contract, as_of=as_of,
    )

    keys = ["machine_id", "ts"]
    full = full.sort_values(keys, ignore_index=True)
    sliced = sliced.sort_values(keys, ignore_index=True)
    pd.testing.assert_frame_equal(full[keys], sliced[keys], check_dtype=False)

    # The maintenance clocks are censored at the first observation in the frame,
    # so a short slice legitimately reports a shorter elapsed time. Everything
    # computed from the sensor windows must match exactly.
    windowed = [name for name in contract.order if not name.startswith("hours_since_maint_")]
    pd.testing.assert_frame_equal(full[windowed], sliced[windowed], check_exact=False, rtol=1e-12)


# --- End-to-end leakage ----------------------------------------------------


def test_changing_the_future_does_not_change_a_single_past_feature(project: Settings) -> None:
    """The spike test, promoted to the whole pipeline. Rewrite every sensor
    reading after a cut-off and assert that no feature row before it moved."""
    contract = build_contract(project)
    staged = load_staged(prepare(project))
    cutoff = pd.Timestamp("2015-01-07 00:00:00")

    baseline = compute_features(
        staged["telemetry"], staged["errors"], staged["maint"], staged["machines"], project, contract
    )

    tampered_telemetry = staged["telemetry"].copy()
    future = tampered_telemetry["ts"] > cutoff
    for sensor in ("volt", "rotate", "pressure", "vibration"):
        tampered_telemetry.loc[future, sensor] = tampered_telemetry.loc[future, sensor] * 3.0 + 11.0

    tampered_errors = pd.concat(
        [
            staged["errors"],
            pd.DataFrame(
                {
                    "machine_id": pd.Series([1] * 5, dtype="int32"),
                    "ts": pd.date_range(cutoff + timedelta(hours=1), periods=5, freq="1h"),
                    "error_id": pd.Series(["error1"] * 5, dtype="string"),
                }
            ),
        ],
        ignore_index=True,
    )

    tampered = compute_features(
        tampered_telemetry, tampered_errors, staged["maint"], staged["machines"], project, contract
    )

    keys = ["machine_id", "ts"]
    past = baseline["ts"] <= cutoff
    assert past.sum() > 100
    pd.testing.assert_frame_equal(
        baseline.loc[past].sort_values(keys, ignore_index=True),
        tampered.loc[tampered["ts"] <= cutoff].sort_values(keys, ignore_index=True),
    )
