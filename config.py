"""Settings — the single source of truth for configuration.

No other module reads ``os.environ``. Import :func:`get_settings` instead.

Credentials are optional at construction time and enforced only at the point of
use, via the ``require_*`` helpers. That keeps the whole of phases 1-3 runnable
with ``LOCAL_MODE=true``, no AWS account and no keys. Tests should build their
own instance with ``Settings(_env_file=None, ...)`` so a local ``.env`` cannot
leak into them.

The four problem-definition settings — horizon, cadence, exclusion, windows —
are what the whole project hangs off. Changing any of them invalidates every
label, every feature matrix and every metric already computed, so they live here
and nowhere else.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

#: The three modelling partitions. Rows falling in a gap belong to none of them
#: and are dropped, so ``"gap"`` is a reporting value only — never a split.
SplitName = Literal["train", "val", "test"]
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

#: Repository root, resolved from this file rather than the process working
#: directory: the SQL has to be findable from pytest and from a Glue job alike.
PROJECT_ROOT = Path(__file__).resolve().parent
SQL_DIR = PROJECT_ROOT / "sql"


class MissingCredentialError(RuntimeError):
    """A credential is needed at runtime but was never configured."""


class Settings(BaseSettings):
    """Everything configurable, loaded from the environment or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Mode --------------------------------------------------------------
    #: True runs the entire training pipeline off local parquet with zero AWS
    #: calls. Phases 1-3 never need this to be False.
    local_mode: bool = True
    log_level: LogLevel = "INFO"

    # --- Local paths -------------------------------------------------------
    data_dir: Path = Path("./data")

    # --- AWS ---------------------------------------------------------------
    aws_region: str = "us-east-1"
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None
    s3_bucket: str | None = None
    s3_raw_prefix: str = "raw/"
    s3_curated_prefix: str = "curated/"
    s3_features_prefix: str = "features/"
    s3_scores_prefix: str = "scores/"

    # --- Glue --------------------------------------------------------------
    glue_database: str = "pdm_catalog"
    glue_role_arn: str | None = None

    # --- SageMaker ---------------------------------------------------------
    sagemaker_role_arn: str | None = None
    sagemaker_training_instance: str = "ml.m5.xlarge"
    sagemaker_transform_instance: str = "ml.m5.large"

    # --- MLflow ------------------------------------------------------------
    #: sqlite, not ``file:./mlruns``: MLflow 3.x refuses the filesystem store
    #: outright, and the *model registry* never worked on it at all. This is
    #: still a local file with no server and no account.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "pdm-failure-24h"
    mlflow_model_name: str = "pdm-xgb"

    # --- Problem definition ------------------------------------------------
    #: A row at ``ts`` is positive when a failure falls in ``(ts, ts + horizon]``.
    prediction_horizon_hours: int = Field(default=24, ge=1)
    #: How often the fleet is scored. Also the spacing of the training grid:
    #: training on every hourly row would mostly train on near-duplicates.
    feature_cadence_hours: int = Field(default=3, ge=1)
    #: NoDecode: pydantic-settings would otherwise JSON-decode this before the
    #: validator below sees it, so `3,12,24` in .env would be a hard error.
    window_sizes_hours: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [3, 12, 24])
    #: Rows from a failure until this many hours later are dropped, not labelled
    #: 0. A machine mid-repair or freshly repaired looks nothing like a healthy
    #: one; leaving those rows in teaches the model to detect repairs.
    post_failure_exclusion_hours: int = Field(default=24, ge=0)

    # --- Splits (time-based, with a gap >= horizon) ------------------------
    #: Inclusive last date of training. The exclusive bound is midnight after.
    train_end: date = date(2015, 8, 31)
    gap_hours: int = Field(default=24, ge=0)
    #: Inclusive last date of validation.
    val_end: date = date(2015, 10, 31)

    # --- Decision economics ------------------------------------------------
    cost_false_alarm: float = Field(default=250.0, ge=0)
    cost_missed_failure: float = Field(default=10_000.0, ge=0)
    #: 0.0 means "derive from the cost curve at eval time". Any other value pins
    #: the threshold, which is only ever right if you measured it.
    decision_threshold: float = Field(default=0.0, ge=0.0, lt=1.0)

    # --- Validation --------------------------------------------------------
    @field_validator("s3_raw_prefix", "s3_curated_prefix", "s3_features_prefix", "s3_scores_prefix")
    @classmethod
    def _normalise_prefix(cls, value: str) -> str:
        """S3 prefixes are directory-like: no leading slash, one trailing slash."""
        value = value.strip().lstrip("/")
        if value and not value.endswith("/"):
            value += "/"
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("window_sizes_hours", mode="before")
    @classmethod
    def _split_windows(cls, value: object) -> object:
        """Accept `3,12,24` from .env as well as a JSON list."""
        if isinstance(value, str):
            if value.strip().startswith("["):
                return value  # let pydantic parse the JSON form
            return [int(part.strip()) for part in value.split(",") if part.strip()]
        return value

    @field_validator("window_sizes_hours")
    @classmethod
    def _check_windows(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("WINDOW_SIZES_HOURS must list at least one window.")
        if any(hours < 1 for hours in value):
            raise ValueError(f"WINDOW_SIZES_HOURS must all be >= 1; got {value}.")
        if len(set(value)) != len(value):
            raise ValueError(f"WINDOW_SIZES_HOURS contains duplicates: {value}.")
        return sorted(value)

    @model_validator(mode="after")
    def _check_split_geometry(self) -> Settings:
        if self.gap_hours < self.prediction_horizon_hours:
            raise ValueError(
                f"GAP_HOURS ({self.gap_hours}) must be >= PREDICTION_HORIZON_HOURS "
                f"({self.prediction_horizon_hours}). With a smaller gap the last training rows "
                f"are labelled using failures that fall inside the validation window."
            )
        if self.train_end >= self.val_end:
            raise ValueError(f"TRAIN_END ({self.train_end}) must fall before VAL_END ({self.val_end}).")
        if self.val_start_ts >= self.val_end_ts:
            raise ValueError(
                f"The gap of {self.gap_hours}h swallows the validation window: it would run "
                f"from {self.val_start_ts} to {self.val_end_ts}."
            )
        return self

    # --- Derived paths -----------------------------------------------------
    @property
    def landing_dir(self) -> Path:
        """Downloaded CSVs. The only place a CSV is allowed to live."""
        return self.data_dir / "landing"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def curated_dir(self) -> Path:
        return self.data_dir / "curated"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def scores_dir(self) -> Path:
        return self.data_dir / "scores"

    # --- Derived split boundaries -----------------------------------------
    # Half-open throughout: a row belongs to the partition whose [start, end)
    # contains its timestamp. Rows landing in a gap belong to no partition.
    @property
    def train_end_ts(self) -> datetime:
        """Exclusive upper bound on training timestamps: midnight after ``train_end``."""
        return datetime.combine(self.train_end, time.min) + timedelta(days=1)

    @property
    def val_start_ts(self) -> datetime:
        return self.train_end_ts + timedelta(hours=self.gap_hours)

    @property
    def val_end_ts(self) -> datetime:
        """Exclusive upper bound on validation timestamps: midnight after ``val_end``."""
        return datetime.combine(self.val_end, time.min) + timedelta(days=1)

    @property
    def test_start_ts(self) -> datetime:
        """The same gap sits between validation and test, for the same reason."""
        return self.val_end_ts + timedelta(hours=self.gap_hours)

    # --- Other derived values ---------------------------------------------
    @property
    def max_window_hours(self) -> int:
        """Longest rolling window; also the warm-up each machine's history needs."""
        return max(self.window_sizes_hours)

    @property
    def derive_threshold(self) -> bool:
        """True when the decision threshold comes from the cost curve, not config."""
        return self.decision_threshold == 0.0

    def s3_uri(self, prefix: str, *parts: str) -> str:
        bucket = self.require_s3_bucket()
        suffix = "/".join(part.strip("/") for part in parts if part)
        return f"s3://{bucket}/{prefix}{suffix}" if suffix else f"s3://{bucket}/{prefix}"

    # --- Credential access -------------------------------------------------
    def require_s3_bucket(self) -> str:
        if not self.s3_bucket:
            raise MissingCredentialError("S3_BUCKET is not set.")
        return self.s3_bucket

    def require_glue_role_arn(self) -> str:
        if not self.glue_role_arn:
            raise MissingCredentialError("GLUE_ROLE_ARN is not set.")
        return self.glue_role_arn

    def require_sagemaker_role_arn(self) -> str:
        if not self.sagemaker_role_arn:
            raise MissingCredentialError("SAGEMAKER_ROLE_ARN is not set.")
        return self.sagemaker_role_arn


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Call ``get_settings.cache_clear()`` in tests."""
    return Settings()


def configure_logging(settings: Settings | None = None) -> None:
    """Install the standard logging config. Idempotent; safe to call at each entrypoint."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
