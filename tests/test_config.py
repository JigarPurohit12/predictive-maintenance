"""config.py is the single source of truth, so its guard rails are load-bearing."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from config import MissingCredentialError, Settings, get_settings
from tests.conftest import make_settings


def test_window_sizes_accept_comma_separated_env_form() -> None:
    settings = Settings(_env_file=None, window_sizes_hours="3,12,24")
    assert settings.window_sizes_hours == [3, 12, 24]
    assert settings.max_window_hours == 24


def test_window_sizes_are_sorted_and_deduplicated() -> None:
    assert Settings(_env_file=None, window_sizes_hours="24,3,12").window_sizes_hours == [3, 12, 24]
    with pytest.raises(ValidationError, match="duplicates"):
        Settings(_env_file=None, window_sizes_hours="3,3,12")
    with pytest.raises(ValidationError, match=">= 1"):
        Settings(_env_file=None, window_sizes_hours="0,12")


def test_gap_shorter_than_horizon_is_rejected() -> None:
    """Without a gap of at least one horizon, the last training rows carry
    labels that depend on data inside the validation window."""
    with pytest.raises(ValidationError, match="GAP_HOURS"):
        Settings(_env_file=None, prediction_horizon_hours=24, gap_hours=12)


def test_split_dates_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="TRAIN_END"):
        Settings(_env_file=None, train_end=date(2015, 10, 31), val_end=date(2015, 8, 31))


def test_gap_may_not_swallow_the_validation_window() -> None:
    with pytest.raises(ValidationError, match="swallows the validation window"):
        Settings(
            _env_file=None,
            train_end=date(2015, 8, 31),
            val_end=date(2015, 9, 1),
            prediction_horizon_hours=24,
            gap_hours=72,
        )


def test_split_boundaries_are_half_open_and_gapped() -> None:
    settings = Settings(
        _env_file=None,
        train_end=date(2015, 8, 31),
        val_end=date(2015, 10, 31),
        gap_hours=24,
        prediction_horizon_hours=24,
    )
    # train_end is inclusive as a date, so the exclusive bound is midnight after.
    assert settings.train_end_ts == datetime(2015, 9, 1)
    assert settings.val_start_ts == datetime(2015, 9, 2)
    assert settings.val_end_ts == datetime(2015, 11, 1)
    assert settings.test_start_ts == datetime(2015, 11, 2)
    # The same buffer sits on both boundaries.
    assert settings.val_start_ts - settings.train_end_ts == settings.test_start_ts - settings.val_end_ts


def test_threshold_of_zero_means_derive_from_the_cost_curve() -> None:
    assert Settings(_env_file=None, decision_threshold=0.0).derive_threshold is True
    assert Settings(_env_file=None, decision_threshold=0.4).derive_threshold is False
    with pytest.raises(ValidationError):
        Settings(_env_file=None, decision_threshold=1.0)


def test_s3_prefixes_are_normalised() -> None:
    settings = Settings(_env_file=None, s3_raw_prefix="/raw", s3_curated_prefix="curated")
    assert settings.s3_raw_prefix == "raw/"
    assert settings.s3_curated_prefix == "curated/"


def test_s3_uri_requires_a_bucket() -> None:
    settings = Settings(_env_file=None, s3_bucket=None)
    with pytest.raises(MissingCredentialError, match="S3_BUCKET"):
        settings.s3_uri(settings.s3_raw_prefix, "telemetry")
    with_bucket = Settings(_env_file=None, s3_bucket="pdm-platform")
    assert with_bucket.s3_uri(with_bucket.s3_raw_prefix, "telemetry") == "s3://pdm-platform/raw/telemetry"


def test_missing_role_arns_raise_rather_than_returning_none() -> None:
    settings = Settings(_env_file=None)
    with pytest.raises(MissingCredentialError, match="GLUE_ROLE_ARN"):
        settings.require_glue_role_arn()
    with pytest.raises(MissingCredentialError, match="SAGEMAKER_ROLE_ARN"):
        settings.require_sagemaker_role_arn()


def test_derived_paths_follow_data_dir(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert settings.landing_dir == tmp_path / "data" / "landing"
    assert settings.raw_dir == tmp_path / "data" / "raw"
    assert settings.curated_dir == tmp_path / "data" / "curated"


def test_local_mode_is_the_default_and_needs_no_credentials() -> None:
    settings = Settings(_env_file=None)
    assert settings.local_mode is True
    assert settings.aws_access_key_id is None


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
