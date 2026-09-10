"""The feature contract: what the columns are, what order they are in, and the
hash that proves training and serving agree about both.

XGBoost fed a numpy array cares about column *position*. Nothing warns you when
the order drifts — the model simply reads vibration as pressure and the
predictions quietly stop meaning anything. So the order is derived here, once,
from the settings that define it, hashed, logged to MLflow with every run, and
asserted at inference time against the hash baked into the model artifact.

The spec calls for a module-level ``FEATURE_ORDER``. It is a function instead:
the order depends on ``WINDOW_SIZES_HOURS``, and computing it at import time
would read settings as a side effect of importing the module. Call
:func:`build_contract` and pass the result around.

The vocabularies below (error codes, components, machine models) are pinned
constants rather than values discovered from the data. Discovering them would
make the feature matrix depend on which codes happen to appear in a batch, so a
quiet Tuesday with no ``error3`` would produce a narrower matrix and a hash
mismatch at inference. Pinning them means an unseen code is dropped loudly
instead.
"""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from config import Settings, get_settings
from data_layer.schemas import SENSOR_COLUMNS

#: Rolling statistics computed for every sensor over every window. Standard
#: deviation is usually the strongest family here: degrading equipment gets
#: noisy before it gets extreme.
ROLLING_STATS: Final[tuple[str, ...]] = ("mean", "std", "min", "max")

#: Lags in *periods* of the raw hourly series, not in hours.
LAG_PERIODS: Final[tuple[int, ...]] = (1, 3, 12)

#: The Azure dataset's vocabularies. See the module docstring on why these are
#: constants and not a ``SELECT DISTINCT``.
ERROR_CODES: Final[tuple[str, ...]] = ("error1", "error2", "error3", "error4", "error5")
COMPONENTS: Final[tuple[str, ...]] = ("comp1", "comp2", "comp3", "comp4")
MACHINE_MODELS: Final[tuple[str, ...]] = ("model1", "model2", "model3", "model4")

#: Identifier columns that travel with the features but are never fed to the
#: model. Keeping them separate is what stops ``machine_id`` becoming a feature
#: and the model memorising which machines fail.
ID_COLUMNS: Final[tuple[str, ...]] = ("machine_id", "ts")
LABEL_COLUMN: Final[str] = "label"
SPLIT_COLUMN: Final[str] = "split"

#: Every feature is a float64. No nullable extension dtypes: XGBoost wants a
#: dense float matrix and pandas' nullable types silently become object arrays.
FEATURE_DTYPE: Final[str] = "float64"


#: The instantaneous readings, carried through as features in their own right.
#: A 3-hour mean is not the same signal as the value at ``ts``, and the logistic
#: baseline in phase 3 is defined on these four columns alone.
RAW_SENSOR_FEATURES: Final[tuple[str, ...]] = SENSOR_COLUMNS


def _sensor_features(window_sizes: list[int]) -> list[str]:
    names: list[str] = list(RAW_SENSOR_FEATURES)
    short, long = min(window_sizes), max(window_sizes)
    for sensor in SENSOR_COLUMNS:
        for window in window_sizes:
            names.extend(f"{sensor}_{stat}_{window}h" for stat in ROLLING_STATS)
        if short != long:
            # A cheap trend proxy: short-window mean minus long-window mean is
            # positive when the sensor is climbing away from its own baseline.
            names.append(f"{sensor}_mean_delta_{short}h_{long}h")
        names.extend(f"{sensor}_lag_{period}" for period in LAG_PERIODS)
    return names


def _error_features(window_sizes: list[int]) -> list[str]:
    names: list[str] = []
    for window in window_sizes:
        names.extend(f"{code}_count_{window}h" for code in ERROR_CODES)
        names.append(f"error_count_{window}h")
    return names


def _static_features() -> list[str]:
    names = ["machine_age"]
    names.extend(f"model_{model}" for model in MACHINE_MODELS)
    names.extend(f"hours_since_maint_{component}" for component in COMPONENTS)
    names.append("hours_since_maint_any")
    return names


class FeatureContract(BaseModel):
    """The canonical column order and everything needed to reproduce it."""

    model_config = ConfigDict(frozen=True)

    order: tuple[str, ...]
    hash: str
    window_sizes_hours: tuple[int, ...]
    lag_periods: tuple[int, ...]
    error_codes: tuple[str, ...]
    components: tuple[str, ...]
    machine_models: tuple[str, ...]

    @property
    def n_features(self) -> int:
        return len(self.order)

    def missing_from(self, frame: pd.DataFrame) -> list[str]:
        return [name for name in self.order if name not in frame.columns]

    def unexpected_in(self, frame: pd.DataFrame) -> list[str]:
        known = {*self.order, *ID_COLUMNS, LABEL_COLUMN, SPLIT_COLUMN}
        return [name for name in frame.columns if name not in known]

    def validate_frame(self, frame: pd.DataFrame, *, allow_extra: bool = True) -> None:
        """Raise unless the frame carries exactly the contracted features."""
        if missing := self.missing_from(frame):
            raise FeatureContractError(f"{len(missing)} contracted feature(s) missing: {missing[:10]}")
        if not allow_extra and (extra := self.unexpected_in(frame)):
            raise FeatureContractError(f"{len(extra)} unexpected column(s): {extra[:10]}")

    def to_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """Features as a dense float matrix in contract order.

        This is the only place a DataFrame becomes an array. Every other path
        that reaches a model goes through here, so position can only drift in
        one place and the hash catches it when it does.
        """
        self.validate_frame(frame)
        return frame.loc[:, list(self.order)].to_numpy(dtype=FEATURE_DTYPE, copy=False)


class FeatureContractError(ValueError):
    """The feature matrix does not match the contract."""


def hash_order(order: tuple[str, ...] | list[str]) -> str:
    """A short, stable digest of the column order.

    Short because it goes into every score row; stable because it is a plain
    sha256 of the joined names, so anyone can recompute it from the column list
    without this code.
    """
    digest = hashlib.sha256("|".join(order).encode("utf-8")).hexdigest()
    return digest[:16]


def build_contract(settings: Settings | None = None) -> FeatureContract:
    """Derive the contract from the settings that determine it."""
    settings = settings or get_settings()
    windows = list(settings.window_sizes_hours)
    order = tuple(_sensor_features(windows) + _error_features(windows) + _static_features())
    if len(set(order)) != len(order):
        raise FeatureContractError("Feature names are not unique; a naming rule collides.")
    return FeatureContract(
        order=order,
        hash=hash_order(order),
        window_sizes_hours=tuple(windows),
        lag_periods=LAG_PERIODS,
        error_codes=ERROR_CODES,
        components=COMPONENTS,
        machine_models=MACHINE_MODELS,
    )
