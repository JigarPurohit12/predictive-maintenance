"""Scores become work orders. Every AWS call is mocked; nothing touches network."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from pydantic import ValidationError

from config import Settings
from serving.alerts import (
    WATCH_FRACTION,
    RiskScore,
    build_scores,
    classify,
    priority_for,
    publish,
    scores_to_frame,
    summarise,
    to_work_orders,
)

SCORED_AT = datetime(2015, 11, 1, 12, 0, 0)
THRESHOLD = 0.30


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def predictions(*probabilities: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"machine_id": range(1, len(probabilities) + 1), "probability": list(probabilities)}
    )


# --- Bands -----------------------------------------------------------------


def test_the_bands_split_at_the_threshold_and_half_of_it() -> None:
    assert classify(0.90, THRESHOLD) == "act"
    assert classify(THRESHOLD, THRESHOLD) == "act"
    assert classify(0.20, THRESHOLD) == "watch"
    assert classify(THRESHOLD * WATCH_FRACTION, THRESHOLD) == "watch"
    assert classify(0.05, THRESHOLD) == "ok"


def test_priority_rises_with_how_far_past_the_threshold_a_score_sits() -> None:
    assert priority_for(0.70, THRESHOLD) == "P1"  # >= 2x
    assert priority_for(0.40, THRESHOLD) == "P2"  # >= 1.25x
    assert priority_for(0.31, THRESHOLD) == "P3"


# --- Scores ----------------------------------------------------------------


def test_every_score_carries_its_provenance() -> None:
    """model_version and feature_hash are what let you answer "why did the model
    flag this machine in March?" in September."""
    scores = build_scores(
        predictions(0.9, 0.2, 0.01),
        threshold=THRESHOLD,
        model_version="4",
        feature_hash="abc123",
        horizon_hours=24,
        scored_at=SCORED_AT,
    )
    assert [score.decision for score in scores] == ["act", "watch", "ok"]
    assert all(score.model_version == "4" for score in scores)
    assert all(score.feature_hash == "abc123" for score in scores)
    assert all(score.horizon_hours == 24 for score in scores)


def test_drivers_are_attached_per_row() -> None:
    scores = build_scores(
        predictions(0.9, 0.8),
        threshold=THRESHOLD,
        model_version="1",
        feature_hash="h",
        horizon_hours=24,
        scored_at=SCORED_AT,
        drivers=[["vibration_std_24h", "error2_count_24h"], ["volt_mean_3h"]],
    )
    assert scores[0].top_drivers == ["vibration_std_24h", "error2_count_24h"]
    assert scores[1].top_drivers == ["volt_mean_3h"]


def test_mismatched_driver_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="drivers has"):
        build_scores(
            predictions(0.9, 0.8),
            threshold=THRESHOLD, model_version="1", feature_hash="h",
            horizon_hours=24, drivers=[["only-one"]],
        )


def test_a_probability_outside_zero_to_one_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RiskScore(
            machine_id=1, scored_at=SCORED_AT, horizon_hours=24, probability=1.4,
            decision="act", model_version="1", feature_hash="h",
        )


# --- Work orders -----------------------------------------------------------


def test_only_act_scores_become_work_orders(settings: Settings) -> None:
    """A watch is information; a work order is somebody's afternoon."""
    scores = build_scores(
        predictions(0.9, 0.2, 0.01),
        threshold=THRESHOLD, model_version="4", feature_hash="abc123",
        horizon_hours=24, scored_at=SCORED_AT,
    )
    orders = to_work_orders(scores, threshold=THRESHOLD, settings=settings)
    assert len(orders) == 1
    assert orders[0].machine_id == 1


def test_the_horizon_is_the_deadline(settings: Settings) -> None:
    """A 24-hour warning acted on in 48 hours is not a warning."""
    scores = build_scores(
        predictions(0.9), threshold=THRESHOLD, model_version="4",
        feature_hash="h", horizon_hours=24, scored_at=SCORED_AT,
    )
    order = to_work_orders(scores, threshold=THRESHOLD, settings=settings)[0]
    assert order.due_by == SCORED_AT + timedelta(hours=24)


def test_the_reason_names_the_drivers(settings: Settings) -> None:
    scores = build_scores(
        predictions(0.9), threshold=THRESHOLD, model_version="4", feature_hash="h",
        horizon_hours=24, scored_at=SCORED_AT, drivers=[["vibration_std_24h", "error2_count_24h"]],
    )
    order = to_work_orders(scores, threshold=THRESHOLD, settings=settings)[0]
    assert "vibration_std_24h" in order.reason
    assert "90.0%" in order.reason


def test_an_unexplained_score_still_produces_a_usable_reason(settings: Settings) -> None:
    scores = build_scores(
        predictions(0.9), threshold=THRESHOLD, model_version="4",
        feature_hash="h", horizon_hours=24, scored_at=SCORED_AT,
    )
    order = to_work_orders(scores, threshold=THRESHOLD, settings=settings)[0]
    assert "no per-row explanation" in order.reason


def test_the_sns_subject_fits_the_hundred_character_limit(settings: Settings) -> None:
    scores = build_scores(
        predictions(0.99), threshold=THRESHOLD, model_version="4",
        feature_hash="h", horizon_hours=24, scored_at=SCORED_AT,
    )
    subject = to_work_orders(scores, threshold=THRESHOLD, settings=settings)[0].subject()
    assert len(subject) <= 100
    assert "\n" not in subject


# --- Publishing ------------------------------------------------------------


class FakeSNS:
    """Records publish calls. No network, no moto, no credentials."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def publish(self, **kwargs: object) -> dict[str, str]:
        self.calls.append(kwargs)
        return {"MessageId": f"msg-{len(self.calls)}"}


def test_each_work_order_is_published_separately(settings: Settings) -> None:
    """A digest turns four independent jobs into one thing somebody forgets
    half of."""
    scores = build_scores(
        predictions(0.9, 0.8), threshold=THRESHOLD, model_version="4",
        feature_hash="h", horizon_hours=24, scored_at=SCORED_AT,
    )
    orders = to_work_orders(scores, threshold=THRESHOLD, settings=settings)
    client = FakeSNS()

    ids = publish(orders, "arn:aws:sns:us-east-1:123:pdm-work-orders", settings, client=client)
    assert ids == ["msg-1", "msg-2"]
    assert len(client.calls) == 2
    assert client.calls[0]["TopicArn"].endswith("pdm-work-orders")
    assert client.calls[0]["MessageAttributes"]["priority"]["StringValue"] == "P1"


def test_publishing_nothing_makes_no_calls(settings: Settings) -> None:
    client = FakeSNS()
    assert publish([], "arn:aws:sns:us-east-1:123:topic", settings, client=client) == []
    assert client.calls == []


def test_the_published_message_is_the_whole_work_order(settings: Settings) -> None:
    import json

    scores = build_scores(
        predictions(0.9), threshold=THRESHOLD, model_version="4", feature_hash="abc123",
        horizon_hours=24, scored_at=SCORED_AT, drivers=[["vibration_std_24h"]],
    )
    orders = to_work_orders(scores, threshold=THRESHOLD, settings=settings)
    client = FakeSNS()
    publish(orders, "arn:aws:sns:us-east-1:123:topic", settings, client=client)

    payload = json.loads(client.calls[0]["Message"])
    assert payload["machine_id"] == 1
    assert payload["model_version"] == "4"
    assert payload["feature_hash"] == "abc123"
    assert payload["drivers"] == ["vibration_std_24h"]


# --- Reporting -------------------------------------------------------------


def test_the_summary_counts_each_band() -> None:
    scores = build_scores(
        predictions(0.9, 0.85, 0.2, 0.01), threshold=THRESHOLD, model_version="4",
        feature_hash="h", horizon_hours=24, scored_at=SCORED_AT,
    )
    summary = summarise(scores)
    assert summary == {
        "machines": 4, "act": 2, "watch": 1, "ok": 1,
        "max_probability": 0.9, "model_version": "4", "feature_hash": "h",
    }


def test_an_empty_run_summarises_without_blowing_up() -> None:
    assert summarise([])["machines"] == 0


def test_scores_serialise_to_a_flat_frame() -> None:
    scores = build_scores(
        predictions(0.9), threshold=THRESHOLD, model_version="4", feature_hash="h",
        horizon_hours=24, scored_at=SCORED_AT, drivers=[["a", "b"]],
    )
    frame = scores_to_frame(scores)
    assert frame.loc[0, "top_drivers"] == "a,b"
    assert frame.loc[0, "model_version"] == "4"


def test_an_empty_frame_still_has_the_right_columns() -> None:
    frame = scores_to_frame([])
    assert "machine_id" in frame.columns and "feature_hash" in frame.columns
    assert frame.empty
