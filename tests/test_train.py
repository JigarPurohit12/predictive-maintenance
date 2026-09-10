"""Phase 3: the models, the threshold discipline, and MLflow tracking.

These run against the synthetic fixture, so no *metric* here means anything —
the assertions are about mechanics: that each model fits, that the threshold is
chosen on validation and reused unchanged on test, that the training-set
quantile in the rules baseline never sees validation, and that everything an
audit needs reaches MLflow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import Settings
from features.build_features import load_matrix
from features.contract import LABEL_COLUMN, RAW_SENSOR_FEATURES, build_contract
from training.evaluate import format_results, results_table
from training.splits import split_frames
from training.train import (
    DEFAULT_MODELS,
    MODEL_BUILDERS,
    RULES_FEATURE,
    RULES_PERCENTILE,
    best_run,
    fit_logistic,
    fit_rules,
    fit_xgboost_tuned,
    train,
    train_one,
)


@pytest.fixture
def parts(built_project: Settings) -> dict[str, pd.DataFrame]:
    return split_frames(load_matrix(built_project))


@pytest.fixture
def contract(built_project: Settings):
    return build_contract(built_project)


# --- The fixture is a sane test bed ----------------------------------------


def test_the_fixture_has_a_plausible_base_rate(parts: dict[str, pd.DataFrame]) -> None:
    """Not an assertion about the real data — an assertion that the fixture is
    a rare-event problem at all, because at 30% positives none of the imbalance
    machinery below would be exercised."""
    for name, frame in parts.items():
        rate = frame[LABEL_COLUMN].mean()
        assert 0.005 < rate < 0.10, f"{name} base rate {rate:.2%} is not a rare-event problem"


def test_every_split_has_positives(parts: dict[str, pd.DataFrame]) -> None:
    for name, frame in parts.items():
        assert frame[LABEL_COLUMN].sum() > 0, f"{name} has no positives"


# --- The baselines ---------------------------------------------------------


def test_the_rules_quantile_is_taken_on_train_only(parts, contract, built_project: Settings) -> None:
    """Computing the cutoff over the whole dataset leaks the test distribution
    into the baseline and quietly flatters it."""
    fitted = fit_rules(parts["train"], contract, built_project)
    expected = float(np.percentile(parts["train"][RULES_FEATURE], RULES_PERCENTILE))
    assert fitted.params["cutoff"] == pytest.approx(expected)

    everything = pd.concat(parts.values(), ignore_index=True)
    over_everything = float(np.percentile(everything[RULES_FEATURE], RULES_PERCENTILE))
    assert fitted.params["cutoff"] != pytest.approx(over_everything)


def test_the_rules_baseline_ranks_by_how_far_above_the_cutoff(parts, contract, built_project) -> None:
    fitted = fit_rules(parts["train"], contract, built_project)
    scores = fitted.predict_proba(parts["val"])
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
    # Monotone in the underlying feature, which is what makes a PR curve meaningful.
    order = np.argsort(parts["val"][RULES_FEATURE].to_numpy())
    assert np.all(np.diff(scores[order]) >= -1e-12)


def test_the_logistic_baseline_uses_the_raw_sensors_alone(parts, contract, built_project) -> None:
    fitted = fit_logistic(parts["train"], contract, built_project)
    assert fitted.params["n_features"] == len(RAW_SENSOR_FEATURES) == 4


def test_the_logistic_scaler_is_fitted_inside_the_pipeline(parts, contract, built_project) -> None:
    """Fitting a scaler on the full dataset is leakage — quieter than a random
    split, and just as real."""
    fitted = fit_logistic(parts["train"], contract, built_project)
    scaler = fitted.estimator.named_steps["scale"]
    expected = parts["train"][list(RAW_SENSOR_FEATURES)].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected, rtol=1e-9)


# --- Every model runs ------------------------------------------------------


@pytest.mark.parametrize("name", DEFAULT_MODELS)
def test_each_model_produces_probabilities_in_range(name, parts, contract, built_project) -> None:
    fitted = MODEL_BUILDERS[name](parts["train"], contract, built_project)
    scores = fitted.predict_proba(parts["val"])
    assert scores.shape == (len(parts["val"]),)
    assert np.isfinite(scores).all()
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_xgboost_gets_the_negative_positive_ratio_as_its_weight(parts, contract, built_project) -> None:
    fitted = fit_xgboost_tuned(parts["train"], contract, built_project)
    labels = parts["train"][LABEL_COLUMN]
    expected = (labels == 0).sum() / (labels == 1).sum()
    assert fitted.params["scale_pos_weight"] == pytest.approx(expected)


def test_the_smote_ablation_is_absent_rather_than_fatal(parts, contract, built_project) -> None:
    """imbalanced-learn is optional on purpose. SMOTE is the ablation you run to
    make a point, never part of the comparison."""
    fitted = MODEL_BUILDERS["smote"](parts["train"], contract, built_project)
    assert fitted is None or fitted.params["resampler"] == "SMOTE"
    assert "smote" not in DEFAULT_MODELS


# --- Threshold discipline --------------------------------------------------


def test_the_threshold_is_chosen_on_validation_and_reused_on_test(parts, contract, built_project, tmp_path) -> None:
    """Picking the threshold on test is how a cost curve turns into a number
    nobody should believe."""
    run = train_one("xgboost", parts, contract, built_project, tmp_path)
    assert run is not None

    val_result = next(result for result in run.results if result.split == "val")
    test_result = next(result for result in run.results if result.split == "test")

    assert val_result.chosen_threshold == pytest.approx(run.validation_threshold)
    assert test_result.chosen_threshold == pytest.approx(run.validation_threshold, abs=1e-6)
    assert "validation" in test_result.notes


def test_a_pinned_threshold_in_config_overrides_the_cost_curve(parts, contract, built_project, tmp_path) -> None:
    pinned = built_project.model_copy(update={"decision_threshold": 0.42})
    run = train_one("rules", parts, contract, pinned, tmp_path)
    assert run is not None
    assert run.validation_threshold == pytest.approx(0.42)


def test_each_run_writes_its_cost_curve_and_pr_curve(parts, contract, built_project, tmp_path) -> None:
    """The cost curve is the artifact that connects the model to money."""
    run = train_one("rules", parts, contract, built_project, tmp_path)
    assert run is not None
    assert run.artifacts["cost_curve"].exists()
    assert run.artifacts["pr_curve"].exists()


def test_tree_models_get_a_shap_summary(parts, contract, built_project, tmp_path) -> None:
    run = train_one("xgboost", parts, contract, built_project, tmp_path)
    assert run is not None
    assert "shap" in run.artifacts and run.artifacts["shap"].exists()


# --- The full comparison ---------------------------------------------------


@pytest.fixture(scope="module")
def full_run(built_project: Settings, tmp_path_factory: pytest.TempPathFactory):
    return train(built_project, DEFAULT_MODELS, track=True, artifact_dir=tmp_path_factory.mktemp("plots"))


def test_the_comparison_has_a_row_per_model_per_split(full_run) -> None:
    """The phase 3 acceptance criterion: at least five models, side by side."""
    results, runs = full_run
    assert set(runs) == set(DEFAULT_MODELS)
    assert len(DEFAULT_MODELS) >= 5

    table = results_table(results)
    for split in ("val", "test"):
        assert len(table[table["split"] == split]) == len(DEFAULT_MODELS)


def test_the_word_accuracy_appears_nowhere_in_the_output(full_run) -> None:
    """With a 3% base rate, predicting "never fails" scores 97%. If any output
    of this project contains the word, something has gone wrong."""
    results, _ = full_run
    rendered = format_results(results).lower()
    assert "accuracy" not in rendered
    assert not any("accuracy" in field.lower() for field in results_table(results).columns)


def test_pr_auc_lift_puts_the_score_in_context(full_run) -> None:
    """0.31 on a 2% base rate is a 15x lift; the same 0.31 on a 25% base rate is
    barely better than nothing."""
    results, _ = full_run
    for result in results:
        assert result.pr_auc_lift == pytest.approx(result.pr_auc / result.base_rate)


def test_a_winner_is_chosen_by_validation_pr_auc(full_run) -> None:
    _, runs = full_run
    winner = best_run(runs)
    val_scores = {
        name: next(result.pr_auc for result in run.results if result.split == "val")
        for name, run in runs.items()
    }
    assert winner == max(val_scores, key=lambda name: val_scores[name])


def test_every_model_is_at_least_compared_against_the_rules_baseline(full_run) -> None:
    """A gradient-boosted model that barely beats a threshold rule is a finding,
    not a failure — but you only know that if you measured the rule."""
    _, runs = full_run
    assert "rules" in runs


# --- MLflow ----------------------------------------------------------------


def test_mlflow_captures_what_an_audit_needs(built_project: Settings, full_run) -> None:
    import mlflow

    mlflow.set_tracking_uri(built_project.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(built_project.mlflow_experiment)
    assert experiment is not None

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    runs = runs[runs["tags.mlflow.runName"].isin(DEFAULT_MODELS)]
    assert len(runs) >= len(DEFAULT_MODELS)

    contract = build_contract(built_project)
    assert (runs["params.feature_hash"] == contract.hash).all()
    for column in (
        "params.decision_threshold",
        "params.horizon_hours",
        "params.train_end_ts",
        "params.cost_missed_failure",
        "metrics.val.pr_auc",
        "metrics.val.recall",
    ):
        assert column in runs.columns, f"{column} was not logged"
        assert runs[column].notna().all()


def test_tracking_can_be_switched_off(built_project: Settings, tmp_path: Path) -> None:
    results, runs = train(built_project, ("rules",), track=False, artifact_dir=tmp_path)
    assert runs and results


def test_a_matrix_with_no_validation_rows_is_rejected(
    built_project: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without validation there is nowhere honest to choose the threshold, so
    this fails loudly rather than falling back to test."""
    train_only = load_matrix(built_project).query("split == 'train'")
    monkeypatch.setattr("training.train.load_matrix", lambda _settings=None: train_only)

    with pytest.raises(ValueError, match="no 'val' rows"):
        train(built_project, ("rules",), track=False, artifact_dir=tmp_path)
