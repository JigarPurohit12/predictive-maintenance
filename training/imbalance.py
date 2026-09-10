"""Class imbalance, and turning a score into a decision.

Two jobs:

**Weighting.** ``scale_pos_weight`` set to the negative/positive ratio, which is
XGBoost's own lever for imbalance and costs nothing. Not SMOTE — see the note in
:mod:`training.train`.

**Thresholding.** Once ``scale_pos_weight`` is in play, 0.5 means nothing at all:
the model is no longer estimating the true posterior, and the default threshold
is an arbitrary point on a curve. The threshold has to come from somewhere real,
and the only real thing here is money::

    cost(t) = COST_FALSE_ALARM x FP(t) + COST_MISSED_FAILURE x FN(t)

Sweep t, pick the minimum, plot the curve. That plot is the most persuasive
artifact in the project because it is the one that connects the model to a
number the business already has an opinion about.

The alternative operating point — "recall at a fixed false-alarm rate" — is the
one to quote in a sentence, because "catches 71% of failures at one false alarm
per machine per month" is a thing a maintenance manager can agree or disagree
with, and an AUC is not.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import average_precision_score, roc_auc_score

LOGGER = logging.getLogger(__name__)

#: Mean hours in a month, for converting a false-alarm count into the units a
#: maintenance planner thinks in.
HOURS_PER_MONTH: Final[float] = 730.0

#: Above this many distinct scores, the cost curve is evaluated on a quantile
#: grid instead of every unique value. 2,000 points resolve the minimum far
#: finer than the noise in the estimate.
MAX_CURVE_POINTS: Final[int] = 2_000


class OperatingPoint(BaseModel):
    """One threshold and everything that follows from it."""

    model_config = ConfigDict(frozen=True)

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    recall: float
    precision: float
    false_alarm_rate: float
    expected_cost: float
    alarms_per_machine_month: float | None = None

    @property
    def confusion_matrix(self) -> dict[str, int]:
        return {
            "tp": self.true_positives,
            "fp": self.false_positives,
            "tn": self.true_negatives,
            "fn": self.false_negatives,
        }


def scale_pos_weight(labels: np.ndarray | pd.Series) -> float:
    """Negatives over positives — XGBoost's own imbalance lever.

    Returns 1.0 when there are no positives rather than dividing by zero: a
    split with no failures in it is a data problem, and the caller is told about
    it by the warning rather than by a ZeroDivisionError three frames up.
    """
    labels = np.asarray(labels)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0:
        LOGGER.warning("No positive labels; scale_pos_weight falls back to 1.0.")
        return 1.0
    return negatives / positives


def _threshold_grid(scores: np.ndarray, max_points: int = MAX_CURVE_POINTS) -> np.ndarray:
    """Candidate thresholds, from coarse quantiles when the scores are dense."""
    unique = np.unique(scores)
    if len(unique) > max_points:
        unique = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, max_points)))
    # A threshold above every score means "alarm on nothing", which is the
    # do-nothing baseline the cost curve has to be compared against.
    return np.append(unique, np.nextafter(unique[-1], np.inf))


def cost_curve(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    *,
    cost_false_alarm: float,
    cost_missed_failure: float,
    n_machines: int | None = None,
    span_hours: float | None = None,
    max_points: int = MAX_CURVE_POINTS,
) -> pd.DataFrame:
    """Confusion counts and expected cost at every candidate threshold.

    Computed with two cumulative sums rather than a Python loop over
    thresholds — at 2,000 thresholds and 200,000 rows the loop takes minutes and
    this takes milliseconds.

    ``n_machines`` and ``span_hours`` are optional; supply both to get the
    false-alarm count expressed per machine per month.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError(f"y_true {y_true.shape} and y_score {y_score.shape} differ in length.")

    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)

    order = np.argsort(-y_score, kind="stable")
    sorted_scores = y_score[order]
    sorted_true = y_true[order]
    cumulative_tp = np.cumsum(sorted_true)
    cumulative_fp = np.cumsum(1 - sorted_true)

    thresholds = _threshold_grid(y_score, max_points)
    # How many rows score >= t, for each candidate t.
    flagged = np.searchsorted(-sorted_scores, -thresholds, side="right")

    true_positives = np.where(flagged > 0, cumulative_tp[np.clip(flagged - 1, 0, None)], 0)
    false_positives = np.where(flagged > 0, cumulative_fp[np.clip(flagged - 1, 0, None)], 0)
    false_negatives = positives - true_positives
    true_negatives = negatives - false_positives

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.divide(true_positives, positives) if positives else np.zeros_like(thresholds)
        precision = np.where(flagged > 0, np.divide(true_positives, np.maximum(flagged, 1)), 0.0)
        false_alarm_rate = np.divide(false_positives, negatives) if negatives else np.zeros_like(thresholds)

    curve = pd.DataFrame(
        {
            "threshold": thresholds,
            "flagged": flagged,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "recall": recall,
            "precision": precision,
            "false_alarm_rate": false_alarm_rate,
            "expected_cost": cost_false_alarm * false_positives + cost_missed_failure * false_negatives,
        }
    )

    if n_machines and span_hours:
        machine_months = n_machines * (span_hours / HOURS_PER_MONTH)
        curve["alarms_per_machine_month"] = curve["false_positives"] / machine_months if machine_months else np.nan
    return curve


def _to_operating_point(row: pd.Series) -> OperatingPoint:
    return OperatingPoint(
        threshold=float(row["threshold"]),
        true_positives=int(row["true_positives"]),
        false_positives=int(row["false_positives"]),
        true_negatives=int(row["true_negatives"]),
        false_negatives=int(row["false_negatives"]),
        recall=float(row["recall"]),
        precision=float(row["precision"]),
        false_alarm_rate=float(row["false_alarm_rate"]),
        expected_cost=float(row["expected_cost"]),
        alarms_per_machine_month=(
            float(row["alarms_per_machine_month"]) if "alarms_per_machine_month" in row.index else None
        ),
    )


def cheapest_threshold(curve: pd.DataFrame) -> OperatingPoint:
    """The minimum of the cost curve.

    Ties go to the *higher* threshold, which is the more conservative operating
    point: fewer alarms for the same expected cost, and a maintenance team that
    keeps trusting the system.
    """
    minimum = curve["expected_cost"].min()
    tied = curve[curve["expected_cost"] == minimum]
    return _to_operating_point(tied.loc[tied["threshold"].idxmax()])


def recall_at_false_alarm_rate(curve: pd.DataFrame, max_false_alarm_rate: float) -> OperatingPoint:
    """Best recall subject to FP/(FP+TN) staying under a ceiling."""
    feasible = curve[curve["false_alarm_rate"] <= max_false_alarm_rate]
    if feasible.empty:
        raise ValueError(f"No threshold achieves a false-alarm rate <= {max_false_alarm_rate}.")
    return _to_operating_point(feasible.loc[feasible["recall"].idxmax()])


def recall_at_alarm_budget(curve: pd.DataFrame, alarms_per_machine_month: float) -> OperatingPoint:
    """Best recall subject to an alarm budget a planner would recognise.

    This is the number to quote. Requires ``n_machines`` and ``span_hours`` to
    have been passed to :func:`cost_curve`.
    """
    if "alarms_per_machine_month" not in curve.columns:
        raise ValueError("cost_curve was not given n_machines and span_hours, so the budget is not computable.")
    feasible = curve[curve["alarms_per_machine_month"] <= alarms_per_machine_month]
    if feasible.empty:
        raise ValueError(f"No threshold stays under {alarms_per_machine_month} alarms per machine per month.")
    return _to_operating_point(feasible.loc[feasible["recall"].idxmax()])


def operating_point_at(curve: pd.DataFrame, threshold: float) -> OperatingPoint:
    """The curve row closest to a threshold someone has already chosen."""
    index = (curve["threshold"] - threshold).abs().idxmin()
    return _to_operating_point(curve.loc[index])


def ranking_metrics(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series) -> dict[str, float]:
    """PR-AUC first. ROC-AUC is reported because people ask, never led with:
    on a 2%-positive target it is flattering and barely moves between a useful
    model and a useless one."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        LOGGER.warning("Only one class present; ranking metrics are undefined.")
        return {"pr_auc": float("nan"), "roc_auc": float("nan"), "base_rate": float(np.mean(y_true))}
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "base_rate": float(np.mean(y_true)),
    }
