"""Imbalance handling and the cost curve.

The cost curve is the artifact the whole of phase 3 builds towards, so its
arithmetic is checked against hand-computed numbers rather than against itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.imbalance import (
    HOURS_PER_MONTH,
    cheapest_threshold,
    cost_curve,
    operating_point_at,
    ranking_metrics,
    recall_at_alarm_budget,
    recall_at_false_alarm_rate,
    scale_pos_weight,
)

COSTS = {"cost_false_alarm": 250.0, "cost_missed_failure": 10_000.0}


# --- scale_pos_weight ------------------------------------------------------


def test_scale_pos_weight_is_the_negative_positive_ratio() -> None:
    labels = np.array([0] * 98 + [1] * 2)
    assert scale_pos_weight(labels) == pytest.approx(49.0)


def test_a_balanced_target_weights_at_one() -> None:
    assert scale_pos_weight(np.array([0, 0, 1, 1])) == pytest.approx(1.0)


def test_no_positives_falls_back_rather_than_dividing_by_zero(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        assert scale_pos_weight(np.zeros(10)) == 1.0
    assert "No positive labels" in caplog.text


# --- The cost curve --------------------------------------------------------


@pytest.fixture
def tiny_curve() -> pd.DataFrame:
    """Four rows, two positives, scores that separate cleanly enough to reason
    about by hand."""
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.6, 0.4, 0.1])
    return cost_curve(y_true, y_score, **COSTS)


def test_the_curve_covers_every_distinct_score_plus_alarm_on_nothing(tiny_curve: pd.DataFrame) -> None:
    assert len(tiny_curve) == 5
    assert tiny_curve["flagged"].min() == 0
    assert tiny_curve["flagged"].max() == 4


def test_confusion_counts_are_right_at_a_known_threshold(tiny_curve: pd.DataFrame) -> None:
    row = tiny_curve[tiny_curve["threshold"] == 0.6].iloc[0]
    assert row["flagged"] == 2  # 0.9 and 0.6
    assert row["true_positives"] == 2
    assert row["false_positives"] == 0
    assert row["false_negatives"] == 0
    assert row["true_negatives"] == 2
    assert row["recall"] == pytest.approx(1.0)
    assert row["precision"] == pytest.approx(1.0)


def test_expected_cost_is_the_stated_formula(tiny_curve: pd.DataFrame) -> None:
    row = tiny_curve[tiny_curve["threshold"] == 0.1].iloc[0]
    # Everything flagged: 2 false alarms, no misses.
    assert row["false_positives"] == 2
    assert row["false_negatives"] == 0
    assert row["expected_cost"] == pytest.approx(2 * 250.0)


def test_alarming_on_nothing_costs_every_missed_failure(tiny_curve: pd.DataFrame) -> None:
    """The do-nothing baseline the curve has to beat."""
    row = tiny_curve.loc[tiny_curve["flagged"].idxmin()]
    assert row["flagged"] == 0
    assert row["false_negatives"] == 2
    assert row["expected_cost"] == pytest.approx(2 * 10_000.0)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        cost_curve(np.array([1, 0]), np.array([0.5]), **COSTS)


# --- Picking a threshold ---------------------------------------------------


def test_the_minimum_of_the_curve_is_found() -> None:
    """A missed failure costs 40 false alarms, so the cheapest point flags
    generously — but not so generously that it flags the clear negatives."""
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_score = np.array([0.95, 0.80, 0.55, 0.30, 0.20, 0.05])
    curve = cost_curve(y_true, y_score, **COSTS)
    best = cheapest_threshold(curve)

    assert best.threshold == pytest.approx(0.80)
    assert best.recall == pytest.approx(1.0)
    assert best.false_positives == 0
    assert best.expected_cost == pytest.approx(0.0)


def test_ties_resolve_to_the_more_conservative_threshold() -> None:
    """Same expected cost, fewer alarms, a maintenance team that keeps
    trusting the system."""
    y_true = np.array([1, 0, 0, 0])
    y_score = np.array([0.9, 0.5, 0.4, 0.3])
    curve = cost_curve(y_true, y_score, cost_false_alarm=0.0, cost_missed_failure=0.0)
    assert cheapest_threshold(curve).threshold == curve["threshold"].max()


def test_a_missed_failure_priced_at_zero_stops_anyone_being_flagged() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.6, 0.4, 0.1])
    curve = cost_curve(y_true, y_score, cost_false_alarm=250.0, cost_missed_failure=0.0)
    best = cheapest_threshold(curve)
    assert best.false_positives == 0
    assert best.true_positives == 0


def test_recall_at_a_false_alarm_ceiling() -> None:
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.array([0.95, 0.55, 0.90, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10])
    curve = cost_curve(y_true, y_score, **COSTS)

    # Allowing one false alarm out of eight negatives (12.5%) lets the
    # threshold drop to 0.55, which catches both positives.
    generous = recall_at_false_alarm_rate(curve, 0.125)
    assert generous.false_alarm_rate == pytest.approx(0.125)
    assert generous.recall == pytest.approx(1.0)

    # Allowing none forces the threshold above 0.90, and half the failures are
    # missed. This is the trade the cost curve prices.
    strict = recall_at_false_alarm_rate(curve, 0.0)
    assert strict.false_positives == 0
    assert strict.recall == pytest.approx(0.5)


def test_an_impossible_false_alarm_ceiling_is_an_error() -> None:
    curve = cost_curve(np.array([1, 0]), np.array([0.4, 0.9]), **COSTS)
    with pytest.raises(ValueError, match="No threshold achieves"):
        recall_at_false_alarm_rate(curve, -0.1)


def test_operating_point_at_snaps_to_the_nearest_row(tiny_curve: pd.DataFrame) -> None:
    point = operating_point_at(tiny_curve, 0.62)
    assert point.threshold == pytest.approx(0.6)


# --- The quotable number ---------------------------------------------------


def test_alarms_are_expressed_per_machine_per_month() -> None:
    """"One false alarm per machine per month" is a claim a maintenance planner
    can agree or disagree with. An AUC is not."""
    y_true = np.array([1] * 10 + [0] * 90)
    y_score = np.concatenate([np.linspace(0.9, 0.6, 10), np.linspace(0.59, 0.01, 90)])
    span_hours = HOURS_PER_MONTH * 2  # two months of data
    curve = cost_curve(y_true, y_score, n_machines=5, span_hours=span_hours, **COSTS)

    flagged_all = curve.loc[curve["flagged"].idxmax()]
    # 90 false alarms over 5 machines x 2 months = 9 per machine per month.
    assert flagged_all["alarms_per_machine_month"] == pytest.approx(9.0)


def test_recall_under_an_alarm_budget() -> None:
    y_true = np.array([1] * 10 + [0] * 90)
    y_score = np.concatenate([np.linspace(0.9, 0.6, 10), np.linspace(0.59, 0.01, 90)])
    curve = cost_curve(y_true, y_score, n_machines=5, span_hours=HOURS_PER_MONTH * 2, **COSTS)

    point = recall_at_alarm_budget(curve, 1.0)
    assert point.alarms_per_machine_month <= 1.0
    assert point.recall == pytest.approx(1.0)  # this fixture separates perfectly


def test_the_budget_needs_the_fleet_size_to_be_computable() -> None:
    curve = cost_curve(np.array([1, 0]), np.array([0.9, 0.1]), **COSTS)
    with pytest.raises(ValueError, match="n_machines and span_hours"):
        recall_at_alarm_budget(curve, 1.0)


# --- Ranking metrics -------------------------------------------------------


def test_pr_auc_is_reported_and_beats_the_base_rate_on_a_good_ranking() -> None:
    y_true = np.array([1] * 5 + [0] * 95)
    y_score = np.concatenate([np.linspace(0.99, 0.9, 5), np.linspace(0.5, 0.0, 95)])
    metrics = ranking_metrics(y_true, y_score)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["base_rate"] == pytest.approx(0.05)


def test_a_random_ranking_scores_near_the_base_rate() -> None:
    rng = np.random.default_rng(0)
    y_true = np.array([1] * 20 + [0] * 980)
    metrics = ranking_metrics(y_true, rng.random(1000))
    assert metrics["pr_auc"] < 0.1


def test_one_class_only_is_reported_as_nan_not_a_crash(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        metrics = ranking_metrics(np.zeros(10), np.linspace(0, 1, 10))
    assert np.isnan(metrics["pr_auc"])
    assert "one class" in caplog.text.lower()


# --- Scale ------------------------------------------------------------------


def test_the_curve_stays_fast_on_a_realistic_number_of_rows() -> None:
    """Two cumulative sums, not a Python loop over thresholds."""
    rng = np.random.default_rng(1)
    y_true = (rng.random(200_000) < 0.02).astype(int)
    y_score = rng.random(200_000)
    curve = cost_curve(y_true, y_score, **COSTS)
    assert len(curve) <= 2_001
    assert curve["expected_cost"].notna().all()
