"""Promotion to the MLflow registry.

A registry with every experiment in it is a list, not a registry. The point is
that "which model is in production" has exactly one answer, and that the answer
carries the two things inference cannot reconstruct: the feature hash and the
decision threshold.
"""

from __future__ import annotations

import pytest

from config import Settings
from features.contract import build_contract
from training.register import (
    SELECTION_METRIC,
    RegistrationError,
    find_best_run,
    register,
)
from training.train import train


@pytest.fixture(scope="module")
def tracked(built_project: Settings, tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Two runs in the experiment, so "the best" is a real choice."""
    import mlflow

    mlflow.set_tracking_uri(built_project.mlflow_tracking_uri)
    train(built_project, ("rules", "xgboost"), track=True, artifact_dir=tmp_path_factory.mktemp("plots"))
    return built_project


def test_the_best_run_is_found_by_validation_pr_auc(tracked: Settings) -> None:
    """Best among the runs that produced a deployable artifact.

    A rules baseline can top the PR-AUC table and still be nothing you can
    register — the selection has to respect that rather than silently promoting
    the runner-up as though it had won.
    """
    import mlflow

    mlflow.set_tracking_uri(tracked.mlflow_tracking_uri)
    run_id = find_best_run(tracked)

    experiment = mlflow.get_experiment_by_name(tracked.mlflow_experiment)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    registrable = runs[runs["params.has_model"] == "true"]
    best = registrable.loc[registrable[f"metrics.{SELECTION_METRIC}"].idxmax()]
    assert run_id == best["run_id"]


def test_a_run_without_a_model_artifact_is_refused_with_an_explanation(tracked: Settings) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracked.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(tracked.mlflow_experiment)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    rules_run = runs[runs["params.has_model"] == "false"].iloc[0]

    with pytest.raises(RegistrationError, match="logged no model artifact"):
        register(rules_run["run_id"], tracked)


def test_the_champion_loads_through_the_flavor_that_wrote_it(tracked: Settings) -> None:
    """The pyfunc wrapper around an XGBClassifier returns class labels, not
    probabilities — exactly the wrong thing for a ranked risk score."""
    from training.register import load_champion

    register(find_best_run(tracked), tracked)
    model, tags = load_champion(tracked)
    assert hasattr(model, "predict_proba")
    assert tags["model_flavor"] in {"sklearn", "xgboost"}


def test_registering_carries_the_hash_and_the_threshold(tracked: Settings) -> None:
    """A model without its threshold is not deployable: 0.5 is meaningless once
    scale_pos_weight is in play."""
    registered = register(find_best_run(tracked), tracked)

    assert registered.name == tracked.mlflow_model_name
    assert registered.feature_hash == build_contract(tracked).hash
    assert 0.0 <= registered.decision_threshold <= 1.0
    assert registered.selection_metric == SELECTION_METRIC


def test_the_version_is_tagged_for_a_later_audit(tracked: Settings) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracked.mlflow_tracking_uri)
    registered = register(find_best_run(tracked), tracked)

    version = MlflowClient().get_model_version(registered.name, registered.version)
    assert version.tags["feature_hash"] == registered.feature_hash
    assert version.tags["horizon_hours"] == str(tracked.prediction_horizon_hours)
    assert version.tags["train_end_ts"] == tracked.train_end_ts.isoformat()


def test_a_run_without_a_feature_hash_cannot_be_registered(tracked: Settings) -> None:
    """The hash is what inference asserts against. Without it a reordered
    feature matrix is undetectable."""
    import mlflow

    mlflow.set_tracking_uri(tracked.mlflow_tracking_uri)
    mlflow.set_experiment(tracked.mlflow_experiment)
    with mlflow.start_run() as bare:
        mlflow.log_param("something", "else")

    with pytest.raises(RegistrationError, match="feature_hash"):
        register(bare.info.run_id, tracked)


def test_an_unknown_experiment_is_reported_clearly(built_project: Settings) -> None:
    missing = built_project.model_copy(update={"mlflow_experiment": "no-such-experiment"})
    with pytest.raises(RegistrationError, match="No MLflow experiment"):
        find_best_run(missing)
