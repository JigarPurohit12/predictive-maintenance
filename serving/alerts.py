"""Scores become work orders, and work orders become SNS messages.

A probability is not an action. This module is where the model stops being a
model and starts being a maintenance job, which means three decisions get made
here and they are all judgement calls rather than mathematics:

* **The bands.** ``act`` above the cost-curve threshold, ``watch`` above a
  fraction of it, ``ok`` below. The watch band exists because a maintenance
  planner would rather see "this one is drifting" a day early than get a hard
  call-out with no warning.
* **The drivers.** Three SHAP contributors per score. "Why did the model flag
  this machine?" is the first thing a technician asks, and an answer they can
  act on is worth more than a probability they cannot.
* **The provenance.** ``model_version`` and ``feature_hash`` on every score.
  These are what let you answer "why did the model flag this machine in March?"
  in September, and they are not optional.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from config import Settings, get_settings

LOGGER = logging.getLogger(__name__)

Decision = Literal["ok", "watch", "act"]

#: The watch band starts at this fraction of the act threshold. Deliberately not
#: a second cost-curve optimisation: watching is free, so there is no cost to
#: minimise, and the number just needs to be low enough to give warning and high
#: enough not to list the whole fleet.
WATCH_FRACTION = 0.5

#: Priority bands for the work order, as multiples of the act threshold.
PRIORITY_BANDS: tuple[tuple[float, str], ...] = ((2.0, "P1"), (1.25, "P2"), (1.0, "P3"))


class RiskScore(BaseModel):
    """One machine, one scoring run."""

    model_config = ConfigDict(frozen=True)

    machine_id: int
    scored_at: datetime
    horizon_hours: int
    probability: float = Field(ge=0.0, le=1.0)
    decision: Decision
    #: Top 3 SHAP contributors. Empty when the model cannot be explained.
    top_drivers: list[str] = Field(default_factory=list)
    #: MLflow registry version — non-negotiable.
    model_version: str
    #: Hash of the feature order; catches skew at runtime.
    feature_hash: str


class WorkOrder(BaseModel):
    """What actually gets sent to a maintenance system."""

    model_config = ConfigDict(frozen=True)

    machine_id: int
    priority: str
    raised_at: datetime
    due_by: datetime
    probability: float
    horizon_hours: int
    reason: str
    drivers: list[str]
    model_version: str
    feature_hash: str

    def to_sns_message(self) -> str:
        return self.model_dump_json()

    def subject(self) -> str:
        # SNS subjects are capped at 100 characters and reject newlines.
        return f"[{self.priority}] Machine {self.machine_id} — failure risk {self.probability:.0%}"[:100]


def classify(probability: float, threshold: float, *, watch_fraction: float = WATCH_FRACTION) -> Decision:
    """Map a probability onto a band."""
    if probability >= threshold:
        return "act"
    if probability >= threshold * watch_fraction:
        return "watch"
    return "ok"


def priority_for(probability: float, threshold: float) -> str:
    for multiple, label in PRIORITY_BANDS:
        if probability >= threshold * multiple:
            return label
    return "P3"


def build_scores(
    predictions: pd.DataFrame,
    *,
    threshold: float,
    model_version: str,
    feature_hash: str,
    horizon_hours: int,
    scored_at: datetime | None = None,
    drivers: list[list[str]] | None = None,
    watch_fraction: float = WATCH_FRACTION,
) -> list[RiskScore]:
    """Turn a frame of probabilities into typed, provenanced scores.

    ``predictions`` needs ``machine_id`` and ``probability``; anything else it
    carries is ignored. ``drivers`` is one list per row, from
    :func:`training.evaluate.top_shap_drivers`.
    """
    scored_at = scored_at or datetime.now()
    if drivers is not None and len(drivers) != len(predictions):
        raise ValueError(f"drivers has {len(drivers)} entries for {len(predictions)} predictions.")

    scores: list[RiskScore] = []
    for position, row in enumerate(predictions.itertuples(index=False)):
        probability = float(row.probability)
        scores.append(
            RiskScore(
                machine_id=int(row.machine_id),
                scored_at=scored_at,
                horizon_hours=horizon_hours,
                probability=probability,
                decision=classify(probability, threshold, watch_fraction=watch_fraction),
                top_drivers=list(drivers[position]) if drivers is not None else [],
                model_version=model_version,
                feature_hash=feature_hash,
            )
        )
    return scores


def to_work_orders(
    scores: list[RiskScore],
    *,
    threshold: float,
    settings: Settings | None = None,
) -> list[WorkOrder]:
    """Only ``act`` scores become work orders.

    A watch is information; a work order is somebody's afternoon. Raising one
    for every drifting machine is how a maintenance team learns to ignore the
    system, which costs more than the model ever saved.
    """
    settings = settings or get_settings()
    orders: list[WorkOrder] = []
    for score in scores:
        if score.decision != "act":
            continue
        drivers = ", ".join(score.top_drivers) if score.top_drivers else "no per-row explanation available"
        orders.append(
            WorkOrder(
                machine_id=score.machine_id,
                priority=priority_for(score.probability, threshold),
                raised_at=score.scored_at,
                # The horizon is the deadline. A 24-hour warning acted on in 48
                # hours is not a warning.
                due_by=score.scored_at + timedelta(hours=score.horizon_hours),
                probability=score.probability,
                horizon_hours=score.horizon_hours,
                reason=(
                    f"Predicted failure risk {score.probability:.1%} within {score.horizon_hours}h "
                    f"(threshold {threshold:.1%}). Top drivers: {drivers}."
                ),
                drivers=list(score.top_drivers),
                model_version=score.model_version,
                feature_hash=score.feature_hash,
            )
        )
    return orders


def scores_to_frame(scores: list[RiskScore]) -> pd.DataFrame:
    """Scores as a frame, for writing to ``scores/dt=.../``."""
    if not scores:
        return pd.DataFrame(
            columns=[
                "machine_id", "scored_at", "horizon_hours", "probability",
                "decision", "top_drivers", "model_version", "feature_hash",
            ]
        )
    frame = pd.DataFrame([score.model_dump() for score in scores])
    frame["top_drivers"] = frame["top_drivers"].map(lambda drivers: ",".join(drivers))
    return frame


def publish(
    orders: list[WorkOrder],
    topic_arn: str,
    settings: Settings | None = None,
    *,
    client: Any | None = None,
) -> list[str]:
    """Publish each work order to SNS. Returns the message ids.

    One message per work order rather than one digest: they are routed and
    acknowledged individually, and a digest turns four independent jobs into one
    thing somebody forgets half of.
    """
    settings = settings or get_settings()
    if not orders:
        LOGGER.info("No work orders to publish.")
        return []

    if client is None:
        import boto3

        client = boto3.client("sns", region_name=settings.aws_region)

    message_ids: list[str] = []
    for order in orders:
        response = client.publish(
            TopicArn=topic_arn,
            Subject=order.subject(),
            Message=order.to_sns_message(),
            MessageAttributes={
                "priority": {"DataType": "String", "StringValue": order.priority},
                "machine_id": {"DataType": "Number", "StringValue": str(order.machine_id)},
            },
        )
        message_ids.append(response["MessageId"])
    LOGGER.info("Published %d work order(s) to %s.", len(orders), topic_arn)
    return message_ids


def summarise(scores: list[RiskScore]) -> dict[str, Any]:
    """A one-line view of a scoring run, for logs and for the run manifest."""
    if not scores:
        return {"machines": 0, "act": 0, "watch": 0, "ok": 0, "max_probability": None}
    probabilities = np.array([score.probability for score in scores])
    counts = pd.Series([score.decision for score in scores]).value_counts()
    return {
        "machines": len(scores),
        "act": int(counts.get("act", 0)),
        "watch": int(counts.get("watch", 0)),
        "ok": int(counts.get("ok", 0)),
        "max_probability": float(probabilities.max()),
        "model_version": scores[0].model_version,
        "feature_hash": scores[0].feature_hash,
    }


def format_summary(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, default=str)
