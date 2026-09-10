"""Phase 5: drift detection and delayed-label evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import Settings
from features.contract import build_contract
from monitoring.drift import (
    PSI_SIGNIFICANT,
    build_reference,
    compute_drift,
    format_report,
    load_reference,
    psi,
    save_reference,
    severity_for,
)
from monitoring.performance import (
    PerformanceRecord,
    append_record,
    attach_outcomes,
    evaluate_period,
    format_record,
    load_history,
    ripe,
)
from tests.conftest import failures_frame

START = pd.Timestamp("2015-11-01 00:00:00")


# --- PSI arithmetic --------------------------------------------------------


def test_identical_distributions_have_zero_psi() -> None:
    proportions = np.array([0.2, 0.3, 0.3, 0.2])
    assert psi(proportions, proportions) == pytest.approx(0.0, abs=1e-12)


def test_psi_grows_as_the_distributions_separate() -> None:
    expected = np.array([0.25, 0.25, 0.25, 0.25])
    mild = psi(expected, np.array([0.30, 0.25, 0.25, 0.20]))
    severe = psi(expected, np.array([0.70, 0.15, 0.10, 0.05]))
    assert 0.0 < mild < severe


def test_psi_is_symmetric() -> None:
    """Worth knowing when reading one: it says "these differ by this much", not
    "this one moved in that direction"."""
    a = np.array([0.1, 0.4, 0.5])
    b = np.array([0.5, 0.3, 0.2])
    assert psi(a, b) == pytest.approx(psi(b, a))


def test_an_empty_bin_does_not_send_psi_to_infinity() -> None:
    value = psi(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
    assert np.isfinite(value)
    assert value > PSI_SIGNIFICANT


def test_the_severity_bands() -> None:
    assert severity_for(0.05) == "stable"
    assert severity_for(0.15) == "moderate"
    assert severity_for(0.30) == "significant"


# --- Reference profiles ----------------------------------------------------


@pytest.fixture
def contract():
    return build_contract(Settings(_env_file=None, window_sizes_hours="3,12,24"))


def synthetic_features(contract, rows: int = 500, *, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {feature: rng.normal(shift, 1.0, rows) for feature in contract.order}
    )


def test_a_batch_from_the_same_distribution_shows_no_drift(contract) -> None:
    profile = build_reference(synthetic_features(contract, seed=0), contract)
    report = compute_drift(profile, synthetic_features(contract, seed=1))

    assert report.ok
    assert report.worst.psi < PSI_SIGNIFICANT


def test_a_shifted_batch_is_flagged(contract) -> None:
    """A recalibrated sensor, a firmware change, a new batch of machines — all
    of them look like this."""
    profile = build_reference(synthetic_features(contract, seed=0), contract)
    report = compute_drift(profile, synthetic_features(contract, shift=3.0, seed=1))

    assert not report.ok
    assert len(report.drifted) == contract.n_features
    assert report.worst.psi > PSI_SIGNIFICANT


def test_only_the_drifted_feature_is_flagged(contract) -> None:
    reference = synthetic_features(contract, seed=0)
    batch = synthetic_features(contract, seed=1)
    batch["vibration_std_24h"] = batch["vibration_std_24h"] + 5.0

    report = compute_drift(build_reference(reference, contract), batch)
    assert [drift.feature for drift in report.drifted] == ["vibration_std_24h"]


def test_features_are_reported_worst_first(contract) -> None:
    profile = build_reference(synthetic_features(contract, seed=0), contract)
    batch = synthetic_features(contract, seed=1)
    batch["volt_mean_3h"] += 4.0
    report = compute_drift(profile, batch)

    scores = [drift.psi for drift in report.features]
    assert scores == sorted(scores, reverse=True)
    assert report.features[0].feature == "volt_mean_3h"


def test_quantile_bins_survive_a_near_constant_feature(contract) -> None:
    """Equal-width bins on a constant column produce zero-width bins; quantile
    bins collapse to fewer instead."""
    reference = synthetic_features(contract, seed=0)
    reference["machine_age"] = 12.0
    profile = build_reference(reference, contract)

    batch = synthetic_features(contract, seed=1)
    batch["machine_age"] = 12.0
    report = compute_drift(profile, batch)
    assert np.isfinite([drift.psi for drift in report.features]).all()


def test_a_batch_missing_a_profiled_feature_is_refused(contract) -> None:
    profile = build_reference(synthetic_features(contract), contract)
    with pytest.raises(ValueError, match="missing"):
        compute_drift(profile, synthetic_features(contract).drop(columns=["volt_mean_3h"]))


def test_a_profile_round_trips_through_json(tmp_path: Path, contract) -> None:
    profile = build_reference(synthetic_features(contract), contract)
    reloaded = load_reference(save_reference(profile, tmp_path / "reference.json"))

    assert reloaded.feature_hash == profile.feature_hash
    assert reloaded.rows == profile.rows
    assert reloaded.edges.keys() == profile.edges.keys()


def test_a_missing_profile_points_at_the_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build-reference"):
        load_reference(tmp_path / "nothing.json")


def test_the_report_says_drift_is_not_the_same_as_damage(contract) -> None:
    profile = build_reference(synthetic_features(contract, seed=0), contract)
    rendered = format_report(compute_drift(profile, synthetic_features(contract, shift=3.0, seed=1)))
    assert "not that the model got worse" in rendered


# --- Delayed labels --------------------------------------------------------


def scores_frame(count: int = 8, *, probability: float | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "machine_id": [(index % 4) + 1 for index in range(count)],
            "scored_at": [START + timedelta(hours=3 * index) for index in range(count)],
            "probability": rng.random(count) if probability is None else [probability] * count,
            "decision": ["ok"] * count,
            "model_version": ["4"] * count,
            "feature_hash": ["abc123"] * count,
        }
    )


def test_only_predictions_whose_horizon_has_closed_are_ripe() -> None:
    """A machine that has not failed yet may still fail in four hours. Counting
    it as a false alarm now makes a working model look broken."""
    scores = scores_frame(8)  # 00:00 to 21:00
    as_of = START + timedelta(hours=30)
    mature = ripe(scores, as_of, horizon_hours=24)

    # Ripe when scored_at + 24h <= as_of, i.e. scored_at <= 06:00.
    assert list(scores.loc[mature, "scored_at"]) == [
        START, START + timedelta(hours=3), START + timedelta(hours=6)
    ]


def test_the_outcome_rule_matches_the_training_label() -> None:
    """Positive when a failure falls in (ts, ts + horizon], open on the left —
    the same rule as sql/02_failure_labels.sql. Evaluating against a different
    rule measures the difference between the rules, not the model."""
    scores = pd.DataFrame(
        {
            "machine_id": [1, 1, 1, 1],
            "scored_at": [START, START + timedelta(hours=6), START + timedelta(hours=12), START + timedelta(hours=36)],
            "probability": [0.9, 0.8, 0.7, 0.6],
        }
    )
    failures = failures_frame([(1, str(START + timedelta(hours=12)), "comp1")])
    joined = attach_outcomes(scores, failures, horizon_hours=12).sort_values("scored_at")

    assert joined["actual"].tolist() == [
        1,  # 00:00, failure exactly 12h later — inside (ts, ts+12h]
        1,  # 06:00, 6h before
        0,  # 12:00, the failure moment itself — open on the left
        0,  # 36:00, long after
    ]


def test_a_failure_on_another_machine_is_not_this_machine_s_outcome() -> None:
    scores = pd.DataFrame(
        {"machine_id": [1, 2], "scored_at": [START, START], "probability": [0.9, 0.9]}
    )
    failures = failures_frame([(2, str(START + timedelta(hours=6)), "comp1")])
    joined = attach_outcomes(scores, failures, horizon_hours=24).sort_values("machine_id")
    assert joined["actual"].tolist() == [0, 1]


def test_no_failures_at_all_means_every_outcome_is_negative() -> None:
    joined = attach_outcomes(scores_frame(4), failures_frame(), horizon_hours=24)
    assert joined["actual"].sum() == 0


# --- The metrics record ----------------------------------------------------


def test_an_evaluation_counts_pending_rows_separately() -> None:
    scores = scores_frame(8)
    failures = failures_frame([(1, str(START + timedelta(hours=5)), "comp1")])
    record = evaluate_period(scores, failures, as_of=START + timedelta(hours=30), horizon_hours=24)

    assert record is not None
    assert record.scored_rows == 3
    assert record.pending_rows == 5
    assert record.scored_rows + record.pending_rows == len(scores)


def test_nothing_ripe_yet_is_not_an_error() -> None:
    assert evaluate_period(scores_frame(4), failures_frame(), as_of=START, horizon_hours=24) is None


def test_no_predictions_at_all_is_not_an_error() -> None:
    assert evaluate_period(pd.DataFrame(), failures_frame(), as_of=START, horizon_hours=24) is None


def test_the_period_is_evaluated_at_its_own_threshold() -> None:
    """Re-scoring history at today's threshold is how a regression gets hidden."""
    scores = scores_frame(6)
    scores["threshold"] = 0.42
    record = evaluate_period(
        scores, failures_frame(), as_of=START + timedelta(hours=48), horizon_hours=24
    )
    assert record is not None
    assert record.threshold == pytest.approx(0.42)


def test_the_threshold_is_recovered_from_the_decisions_when_unrecorded() -> None:
    scores = scores_frame(6)
    scores["probability"] = [0.1, 0.2, 0.6, 0.7, 0.8, 0.9]
    scores["decision"] = ["ok", "ok", "act", "act", "act", "act"]
    record = evaluate_period(
        scores, failures_frame(), as_of=START + timedelta(hours=48), horizon_hours=24
    )
    assert record is not None
    assert record.threshold == pytest.approx(0.6)


def test_the_confusion_matrix_adds_up() -> None:
    scores = scores_frame(6, probability=0.9)
    scores["threshold"] = 0.5
    failures = failures_frame([(1, str(START + timedelta(hours=2)), "comp1")])
    record = evaluate_period(scores, failures, as_of=START + timedelta(hours=48), horizon_hours=24)

    assert record is not None
    total = (
        record.true_positives + record.false_positives + record.true_negatives + record.false_negatives
    )
    assert total == record.scored_rows


def test_provenance_travels_into_the_metrics_row() -> None:
    scores = scores_frame(6)
    record = evaluate_period(
        scores, failures_frame(), as_of=START + timedelta(hours=48), horizon_hours=24
    )
    assert record is not None
    assert record.model_version == "4"
    assert record.feature_hash == "abc123"


# --- The running table -----------------------------------------------------


def test_the_history_is_appended_never_rewritten(tmp_path: Path) -> None:
    """The record of how the model performed is not something a later run should
    be able to edit."""
    path = tmp_path / "history.jsonl"
    for index in range(3):
        record = PerformanceRecord(
            evaluated_at=datetime(2015, 11, 5 + index),
            period_start=START, period_end=START + timedelta(hours=24),
            model_version="4", feature_hash="abc123", threshold=0.5, horizon_hours=24,
            scored_rows=100, pending_rows=10, positives=4, base_rate=0.04, pr_auc=0.3 + index / 100,
            true_positives=3, false_positives=9, true_negatives=87, false_negatives=1,
            recall=0.75, precision=0.25,
        )
        append_record(record, path)

    history = load_history(path)
    assert len(history) == 3
    assert history["pr_auc"].tolist() == [0.30, 0.31, 0.32]


def test_an_absent_history_reads_as_empty(tmp_path: Path) -> None:
    assert load_history(tmp_path / "nothing.jsonl").empty


def test_the_record_renders_without_an_alarm_rate() -> None:
    record = PerformanceRecord(
        evaluated_at=datetime(2015, 11, 5),
        period_start=START, period_end=START + timedelta(hours=24),
        model_version="4", feature_hash="abc", threshold=0.5, horizon_hours=24,
        scored_rows=10, pending_rows=0, positives=1, base_rate=0.1, pr_auc=0.5,
        true_positives=1, false_positives=1, true_negatives=8, false_negatives=0,
        recall=1.0, precision=0.5, alarms_per_machine_month=None,
    )
    rendered = format_record(record)
    assert "n/a" in rendered
    assert "accuracy" not in rendered.lower()
