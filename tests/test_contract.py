"""The feature contract. XGBoost fed a numpy array cares about column position,
and nothing warns you when that drifts — so the order is pinned and hashed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import Settings
from data_layer.schemas import SENSOR_COLUMNS
from features.contract import (
    ERROR_CODES,
    LAG_PERIODS,
    ROLLING_STATS,
    FeatureContractError,
    build_contract,
    hash_order,
)


@pytest.fixture
def contract():
    return build_contract(Settings(_env_file=None, window_sizes_hours="3,12,24"))


def test_the_order_is_deterministic(contract) -> None:
    again = build_contract(Settings(_env_file=None, window_sizes_hours="3,12,24"))
    assert contract.order == again.order
    assert contract.hash == again.hash


def test_names_are_unique(contract) -> None:
    assert len(set(contract.order)) == len(contract.order)


def test_the_feature_count_is_what_the_naming_rules_imply(contract) -> None:
    """A golden count. If a naming rule changes this moves, and it should."""
    sensors, windows = len(SENSOR_COLUMNS), 3
    raw = sensors  # the instantaneous reading at ts
    rolling = sensors * windows * len(ROLLING_STATS)  # 4 * 3 * 4 = 48
    trends = sensors  # one short-minus-long delta per sensor
    lags = sensors * len(LAG_PERIODS)  # 4 * 3 = 12
    errors = windows * (len(ERROR_CODES) + 1)  # per-code plus a total
    static = 1 + 4 + 4 + 1  # age, model one-hot, per-component clocks, min clock
    assert contract.n_features == raw + rolling + trends + lags + errors + static == 96


def test_every_family_is_present(contract) -> None:
    assert "volt" in contract.order
    assert "volt_mean_3h" in contract.order
    assert "vibration_std_24h" in contract.order
    assert "pressure_mean_delta_3h_24h" in contract.order
    assert "rotate_lag_12" in contract.order
    assert "error3_count_12h" in contract.order
    assert "error_count_24h" in contract.order
    assert "machine_age" in contract.order
    assert "model_model4" in contract.order
    assert "hours_since_maint_comp2" in contract.order
    assert "hours_since_maint_any" in contract.order


def test_machine_id_is_never_a_feature(contract) -> None:
    """Feeding it in lets the model memorise which machines fail rather than
    learning what failing looks like."""
    assert "machine_id" not in contract.order
    assert "ts" not in contract.order
    assert "label" not in contract.order


def test_changing_the_windows_changes_the_hash() -> None:
    narrow = build_contract(Settings(_env_file=None, window_sizes_hours="3,24"))
    wide = build_contract(Settings(_env_file=None, window_sizes_hours="3,12,24"))
    assert narrow.hash != wide.hash
    assert narrow.n_features < wide.n_features


def test_a_single_window_produces_no_trend_features() -> None:
    single = build_contract(Settings(_env_file=None, window_sizes_hours="24"))
    assert not [name for name in single.order if "_mean_delta_" in name]


def test_the_hash_is_a_plain_digest_of_the_names(contract) -> None:
    """Anyone can recompute it from the column list without this code."""
    assert contract.hash == hash_order(contract.order)
    assert hash_order(["a", "b"]) != hash_order(["b", "a"])
    assert len(contract.hash) == 16


def test_to_matrix_returns_columns_in_contract_order(contract) -> None:
    shuffled = list(reversed(contract.order))
    frame = pd.DataFrame({name: [float(index)] for index, name in enumerate(shuffled)})
    frame["machine_id"] = 1
    matrix = contract.to_matrix(frame)
    assert matrix.shape == (1, contract.n_features)
    expected = np.array([[float(shuffled.index(name)) for name in contract.order]])
    np.testing.assert_array_equal(matrix, expected)


def test_a_missing_feature_is_an_error_not_a_nan_column(contract) -> None:
    frame = pd.DataFrame({name: [0.0] for name in contract.order[:-1]})
    with pytest.raises(FeatureContractError, match="missing"):
        contract.to_matrix(frame)


def test_extra_columns_are_tolerated_unless_asked_otherwise(contract) -> None:
    frame = pd.DataFrame({name: [0.0] for name in contract.order})
    frame["something_else"] = 1.0
    contract.validate_frame(frame)
    with pytest.raises(FeatureContractError, match="unexpected"):
        contract.validate_frame(frame, allow_extra=False)


def test_id_and_label_columns_are_not_unexpected(contract) -> None:
    frame = pd.DataFrame({name: [0.0] for name in contract.order})
    frame["machine_id"], frame["ts"], frame["label"], frame["split"] = 1, pd.Timestamp("2015-01-01"), 0, "train"
    contract.validate_frame(frame, allow_extra=False)
