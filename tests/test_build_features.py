"""The shared feature path: warm-up, completeness, the cadence grid, and the
training matrix built end to end from parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config import Settings, get_settings
from data_layer import ingest as ingest_module
from features.build_features import (
    assert_complete,
    build_training_matrix,
    compute_features,
    drop_warmup,
    load_manifest,
    load_matrix,
    on_cadence,
    write_matrix,
)
from features.contract import FeatureContractError, build_contract
from tests.conftest import (
    errors_frame,
    machines_frame,
    maint_frame,
    make_settings,
    pipeline_settings,
    synthetic_dataset,
    telemetry_frame,
    write_landing,
)

START = pd.Timestamp("2015-01-01 00:00:00")


def at(hour: int) -> pd.Timestamp:
    return START + pd.Timedelta(hours=hour)


# --- Warm-up ---------------------------------------------------------------


def test_warmup_rows_are_dropped_per_machine() -> None:
    """A 24-hour standard deviation computed from four observations is not a
    24-hour standard deviation."""
    frame = telemetry_frame(machine_ids=(1, 2), hours=40)
    kept = drop_warmup(frame, 24)
    assert len(kept) == (40 - 24) * 2
    assert kept.groupby("machine_id")["ts"].min().eq(at(24)).all()


def test_a_machine_shorter_than_the_warmup_disappears_entirely() -> None:
    frame = telemetry_frame(hours=6)
    assert drop_warmup(frame, 24).empty


# --- The cadence grid ------------------------------------------------------


def test_the_cadence_grid_is_anchored_to_the_epoch() -> None:
    stamps = pd.Series(pd.date_range("2015-01-01 00:00:00", periods=8, freq="1h"))
    keep = on_cadence(stamps, 3)
    assert list(stamps[keep]) == [at(0), at(3), at(6)]


def test_sub_hourly_timestamps_are_never_on_the_grid() -> None:
    stamps = pd.Series([pd.Timestamp("2015-01-01 03:00:00"), pd.Timestamp("2015-01-01 03:30:00")])
    assert list(on_cadence(stamps, 3)) == [True, False]


def test_a_cadence_of_one_keeps_every_hour() -> None:
    stamps = pd.Series(pd.date_range("2015-01-01", periods=5, freq="1h"))
    assert on_cadence(stamps, 1).all()


# --- Completeness ----------------------------------------------------------


def test_assert_complete_rejects_nulls() -> None:
    contract = build_contract(Settings(_env_file=None))
    frame = pd.DataFrame({name: [1.0] for name in contract.order})
    frame.loc[0, "volt_mean_3h"] = None
    with pytest.raises(FeatureContractError, match="null value"):
        assert_complete(frame, contract)


def test_assert_complete_rejects_infinities() -> None:
    contract = build_contract(Settings(_env_file=None))
    frame = pd.DataFrame({name: [1.0] for name in contract.order})
    frame.loc[0, "volt_mean_3h"] = float("inf")
    with pytest.raises(FeatureContractError, match="Non-finite"):
        assert_complete(frame, contract)


def test_a_clean_feature_matrix_has_no_nulls() -> None:
    settings = Settings(_env_file=None)
    contract = build_contract(settings)
    features = compute_features(
        telemetry_frame(machine_ids=(1, 2), hours=60),
        errors_frame([(1, "2015-01-01 10:00:00", "error1")]),
        maint_frame([(1, "2015-01-01 05:00:00", "comp1")]),
        machines_frame((1, 2)),
        settings,
        contract,
    )
    assert_complete(features, contract)
    assert list(features.columns) == ["machine_id", "ts", *contract.order]


# --- End to end ------------------------------------------------------------


@pytest.fixture
def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Ingest a twelve-day synthetic dataset and point settings at it."""
    settings = pipeline_settings(tmp_path)
    write_landing(settings.landing_dir, synthetic_dataset())
    monkeypatch.setattr("features.build_features.get_settings", lambda: settings)
    ingest_module.ingest(settings)
    return settings


def test_the_training_matrix_carries_labels_and_every_feature(built: Settings) -> None:
    contract = build_contract(built)
    matrix = build_training_matrix(built, contract)

    assert not matrix.empty
    assert list(matrix.columns) == ["machine_id", "ts", "label", "split", *contract.order]
    assert set(matrix["split"].unique()) <= {"train", "val", "test"}
    assert matrix["label"].isin((0, 1)).all()
    assert matrix["label"].sum() > 0, "the fixture plants failures; some rows must be positive"


def test_every_matrix_row_sits_on_the_scoring_grid(built: Settings) -> None:
    matrix = build_training_matrix(built)
    assert on_cadence(matrix["ts"], built.feature_cadence_hours).all()


def test_the_matrix_has_one_row_per_machine_timestamp(built: Settings) -> None:
    matrix = build_training_matrix(built)
    assert not matrix.duplicated(subset=["machine_id", "ts"]).any()


def test_writing_records_a_manifest_with_the_feature_hash(built: Settings) -> None:
    contract = build_contract(built)
    matrix = build_training_matrix(built, contract)
    write_matrix(matrix, built, contract)

    manifest = load_manifest(built)
    assert manifest["feature_hash"] == contract.hash
    assert manifest["feature_order"] == list(contract.order)
    assert manifest["prediction_horizon_hours"] == 24
    assert manifest["rows"] == len(matrix)
    assert sum(manifest["rows_by_split"].values()) == len(matrix)


def test_the_written_matrix_round_trips(built: Settings) -> None:
    contract = build_contract(built)
    matrix = build_training_matrix(built, contract)
    write_matrix(matrix, built, contract)

    reloaded = load_matrix(built)
    assert len(reloaded) == len(matrix)
    pd.testing.assert_frame_equal(
        matrix[["machine_id", "ts", "label", *contract.order]].reset_index(drop=True),
        reloaded[["machine_id", "ts", "label", *contract.order]].reset_index(drop=True),
        check_dtype=False,
    )


def test_load_matrix_before_building_points_at_the_fix(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with pytest.raises(FileNotFoundError, match="build_features"):
        load_matrix(settings)


def test_get_settings_is_not_read_at_import_time() -> None:
    """Building the contract at import would make importing the module read the
    environment as a side effect."""
    get_settings.cache_clear()
    import features.contract as contract_module

    assert not hasattr(contract_module, "FEATURE_ORDER")
