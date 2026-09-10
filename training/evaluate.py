"""Scoring a model, and the artifacts that make the score defensible.

The word "accuracy" does not appear in this module and must not appear in its
output. With a 2% base rate, predicting "never fails" for every machine scores
98% while being completely useless, and any interviewer who has worked on
imbalanced problems asks what the base rate was within two questions.

What is reported instead:

* **PR-AUC**, primary. Threshold-free, and it degrades honestly as the target
  gets rarer.
* **Recall at a fixed alarm budget** — the sentence a maintenance planner can
  argue with: "catches 71% of failures at one false alarm per machine per month".
* **The cost curve**, and the confusion matrix at its minimum.
* **ROC-AUC**, last, because people ask. It is flattering here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display on a build box or in a SageMaker container
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from config import Settings
from training.imbalance import (
    OperatingPoint,
    cheapest_threshold,
    cost_curve,
    operating_point_at,
    ranking_metrics,
    recall_at_alarm_budget,
)

LOGGER = logging.getLogger(__name__)

#: The alarm budget the headline recall is quoted at. One call-out per machine
#: per month is roughly what a maintenance team will tolerate before they start
#: ignoring the system, which is the real failure mode of a noisy alerting model.
DEFAULT_ALARM_BUDGET = 1.0

#: Rows sampled for the SHAP summary. Exact SHAP over 200k rows takes minutes
#: and the summary plot cannot render that many points anyway.
SHAP_SAMPLE_ROWS = 2_000


class ModelResult(BaseModel):
    """Everything measured about one model on one split."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    split: str
    rows: int
    positives: int
    base_rate: float
    pr_auc: float
    roc_auc: float
    #: Lift over the trivial ranking. PR-AUC of 0.31 on a 2% base rate is a
    #: 15x lift; the same 0.31 on a 25% base rate is barely better than nothing.
    pr_auc_lift: float
    chosen_threshold: float
    recall_at_threshold: float
    precision_at_threshold: float
    expected_cost: float
    confusion_matrix: dict[str, int]
    recall_at_alarm_budget: float | None = None
    alarm_budget: float | None = None
    notes: str = ""


def evaluate_scores(
    model_name: str,
    split: str,
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    settings: Settings,
    *,
    n_machines: int | None = None,
    span_hours: float | None = None,
    threshold: float | None = None,
    alarm_budget: float = DEFAULT_ALARM_BUDGET,
    notes: str = "",
) -> tuple[ModelResult, pd.DataFrame]:
    """Score one model on one split. Returns the result and the cost curve.

    ``threshold`` pins the operating point — pass the one chosen on validation
    when evaluating test, so the test number reflects a decision made without
    seeing the test set. Leave it None to take the minimum of this split's own
    cost curve, which is only honest on validation.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    curve = cost_curve(
        y_true,
        y_score,
        cost_false_alarm=settings.cost_false_alarm,
        cost_missed_failure=settings.cost_missed_failure,
        n_machines=n_machines,
        span_hours=span_hours,
    )
    point = operating_point_at(curve, threshold) if threshold is not None else cheapest_threshold(curve)
    ranking = ranking_metrics(y_true, y_score)

    budget_recall: float | None = None
    if "alarms_per_machine_month" in curve.columns:
        try:
            budget_recall = recall_at_alarm_budget(curve, alarm_budget).recall
        except ValueError:
            LOGGER.warning("No threshold meets the %.2f alarms/machine/month budget.", alarm_budget)

    base_rate = ranking["base_rate"]
    result = ModelResult(
        model_name=model_name,
        split=split,
        rows=len(y_true),
        positives=int(y_true.sum()),
        base_rate=base_rate,
        pr_auc=ranking["pr_auc"],
        roc_auc=ranking["roc_auc"],
        pr_auc_lift=ranking["pr_auc"] / base_rate if base_rate else float("nan"),
        chosen_threshold=point.threshold,
        recall_at_threshold=point.recall,
        precision_at_threshold=point.precision,
        expected_cost=point.expected_cost,
        confusion_matrix=point.confusion_matrix,
        recall_at_alarm_budget=budget_recall,
        alarm_budget=alarm_budget if budget_recall is not None else None,
        notes=notes,
    )
    return result, curve


def plot_cost_curve(curve: pd.DataFrame, point: OperatingPoint, title: str, path: Path) -> Path:
    """The single most persuasive artifact in the project.

    It connects the model to money, which is the only axis on which anyone
    outside the team can judge whether the threshold is right.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.5, 4.5))

    axis.plot(curve["threshold"], curve["expected_cost"], linewidth=1.6)
    axis.axvline(point.threshold, linestyle="--", linewidth=1.0, color="tab:red")
    axis.annotate(
        f"min cost {point.expected_cost:,.0f}\nat t={point.threshold:.3f}\n"
        f"recall {point.recall:.1%}, {point.false_positives:,} false alarms",
        xy=(point.threshold, point.expected_cost),
        xytext=(0.55, 0.72),
        textcoords="axes fraction",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Expected cost")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    LOGGER.info("Wrote %s", path)
    return path


def plot_precision_recall(y_true: np.ndarray, y_score: np.ndarray, title: str, path: Path) -> Path:
    """PR rather than ROC, deliberately: ROC hides what a 2% base rate does."""
    from sklearn.metrics import PrecisionRecallDisplay

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=axis)
    axis.axhline(float(np.mean(y_true)), linestyle=":", linewidth=1.0, color="grey")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def shap_summary(
    model: Any,
    features: pd.DataFrame,
    feature_order: list[str],
    path: Path,
    *,
    sample_rows: int = SHAP_SAMPLE_ROWS,
    seed: int = 0,
) -> Path | None:
    """SHAP summary for a tree model, sampled.

    Returns None rather than raising when SHAP cannot explain the model: an
    explanation is worth having but never worth failing a training run over.
    """
    try:
        import shap
    except ImportError:  # pragma: no cover - shap is a declared dependency
        LOGGER.warning("shap is not installed; skipping the summary plot.")
        return None

    sample = features.loc[:, feature_order]
    if len(sample) > sample_rows:
        sample = sample.sample(sample_rows, random_state=seed)

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)
    except Exception as exc:  # noqa: BLE001 - never fail a run over a plot
        LOGGER.warning("SHAP could not explain this model (%s); skipping.", exc)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(values, sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    LOGGER.info("Wrote %s", path)
    return path


def top_shap_drivers(
    model: Any,
    features: pd.DataFrame,
    feature_order: list[str],
    *,
    top_n: int = 3,
) -> list[list[str]]:
    """Per-row top contributors, for the ``top_drivers`` field on a score.

    "Why did the model flag this machine?" is the first question a technician
    asks, and "vibration_std_24h, error2_count_24h, hours_since_maint_comp2" is
    an answer they can act on.
    """
    try:
        import shap
    except ImportError:  # pragma: no cover
        return [[] for _ in range(len(features))]

    matrix = features.loc[:, feature_order]
    try:
        values = shap.TreeExplainer(model).shap_values(matrix)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("SHAP could not explain this model (%s); no drivers attached.", exc)
        return [[] for _ in range(len(matrix))]

    values = np.asarray(values)
    if values.ndim == 3:  # some explainers return one array per class
        values = values[..., -1]
    ranked = np.argsort(-np.abs(values), axis=1)[:, :top_n]
    return [[feature_order[index] for index in row] for row in ranked]


def results_table(results: list[ModelResult]) -> pd.DataFrame:
    """The phase 3 acceptance artifact: one row per model, sorted by PR-AUC."""
    frame = pd.DataFrame([result.model_dump() for result in results])
    if frame.empty:
        return frame
    frame = frame.drop(columns=["confusion_matrix"])
    return frame.sort_values(["split", "pr_auc"], ascending=[True, False], ignore_index=True)


def format_results(results: list[ModelResult]) -> str:
    frame = results_table(results)
    if frame.empty:
        return "(no results)"
    shown = frame[
        [
            "split",
            "model_name",
            "rows",
            "positives",
            "base_rate",
            "pr_auc",
            "pr_auc_lift",
            "recall_at_threshold",
            "precision_at_threshold",
            "recall_at_alarm_budget",
            "expected_cost",
        ]
    ].copy()
    for column in ("base_rate", "recall_at_threshold", "precision_at_threshold", "recall_at_alarm_budget"):
        shown[column] = shown[column].map(lambda value: "-" if pd.isna(value) else f"{value:.2%}")
    shown["pr_auc"] = shown["pr_auc"].map("{:.4f}".format)
    shown["pr_auc_lift"] = shown["pr_auc_lift"].map("{:.1f}x".format)
    shown["expected_cost"] = shown["expected_cost"].map("{:,.0f}".format)
    return shown.to_string(index=False)


def write_results(results: list[ModelResult], path: Path, extra: dict[str, Any] | None = None) -> Path:
    """Commit the numbers. Opening the actual JSON in an interview is worth more
    than any figure on a slide."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"results": [result.model_dump() for result in results]}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Wrote %s", path)
    return path
