"""Promote a winning run to the MLflow model registry.

    python -m training.register --run-id <id>
    python -m training.register --best            # highest val PR-AUC in the experiment

Only the winner gets registered. A registry with every experiment in it is a
list, not a registry — the point is that "which model is in production" has
exactly one answer.

What is attached to the version matters as much as the model:

* ``feature_hash`` — asserted at inference against the incoming batch. Without
  it, a reordered feature matrix is undetectable.
* ``decision_threshold`` — a model without its threshold is not deployable, and
  0.5 is meaningless once ``scale_pos_weight`` is in play.
* the split dates and the horizon, so a score can be explained six months later.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from typing import Any

from pydantic import BaseModel, ConfigDict

from config import Settings, configure_logging, get_settings

LOGGER = logging.getLogger(__name__)

#: Run tags that must be present before a run may be registered. Each one is
#: something inference or an audit needs and cannot reconstruct.
REQUIRED_PARAMS = ("feature_hash", "decision_threshold")

#: Set by ``training.train`` when the run logged a serialisable estimator.
HAS_MODEL_PARAM = "has_model"

#: The metric a winner is chosen by. Not accuracy, not ROC-AUC.
SELECTION_METRIC = "val.pr_auc"


class RegisteredModel(BaseModel):
    """What was registered, and enough to find it again."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    run_id: str
    feature_hash: str
    decision_threshold: float
    selection_metric: str
    selection_value: float


class RegistrationError(RuntimeError):
    """A run cannot be registered."""


def _client() -> Any:
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as exc:  # pragma: no cover - mlflow is a declared dependency
        raise RegistrationError("mlflow is not installed.") from exc
    return MlflowClient()


def find_best_run(settings: Settings | None = None, metric: str = SELECTION_METRIC) -> str:
    """The highest-scoring run in the configured experiment."""
    import mlflow

    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    experiment = mlflow.get_experiment_by_name(settings.mlflow_experiment)
    if experiment is None:
        raise RegistrationError(f"No MLflow experiment named {settings.mlflow_experiment!r}.")

    # Only runs that actually logged a model. A rules baseline can win on
    # PR-AUC and still not be a registrable artifact — if it does win, that is a
    # finding to act on, not something to paper over by registering the
    # runner-up silently.
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.has_model = 'true'",
        order_by=[f"metrics.`{metric}` DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RegistrationError(
            f"Experiment {settings.mlflow_experiment!r} has no runs with a logged model. "
            f"If the rules baseline is winning, deploy it as a rule or beat it — do not "
            f"register the runner-up and call it the champion."
        )
    return str(runs.iloc[0]["run_id"])


def register(
    run_id: str,
    settings: Settings | None = None,
    *,
    metric: str = SELECTION_METRIC,
    stage_alias: str = "champion",
) -> RegisteredModel:
    """Register one run's model, carrying the contract forward as tags."""
    import mlflow

    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = _client()

    run = client.get_run(run_id)
    missing = [name for name in REQUIRED_PARAMS if name not in run.data.params]
    if missing:
        raise RegistrationError(
            f"Run {run_id} is missing {missing}. A model without its feature hash and its "
            f"threshold is not deployable — 0.5 means nothing once scale_pos_weight is in play."
        )

    if run.data.params.get("has_model") != "true":
        raise RegistrationError(
            f"Run {run_id} ({run.data.tags.get('mlflow.runName', 'unnamed')}) logged no model "
            f"artifact — a threshold rule is not a deployable model object."
        )

    model_uri = f"runs:/{run_id}/model"
    version = mlflow.register_model(model_uri=model_uri, name=settings.mlflow_model_name)

    tags = {
        "feature_hash": run.data.params["feature_hash"],
        "decision_threshold": run.data.params["decision_threshold"],
        "horizon_hours": run.data.params.get("horizon_hours", ""),
        "cadence_hours": run.data.params.get("cadence_hours", ""),
        "train_end_ts": run.data.params.get("train_end_ts", ""),
        "val_end_ts": run.data.params.get("val_end_ts", ""),
        "version": str(version.version),
        "model_flavor": run.data.params.get("model_flavor", "sklearn"),
        "selection_metric": metric,
        "selection_value": str(run.data.metrics.get(metric, "")),
    }
    for key, value in tags.items():
        client.set_model_version_tag(settings.mlflow_model_name, version.version, key, value)

    try:
        client.set_registered_model_alias(settings.mlflow_model_name, stage_alias, version.version)
    except Exception as exc:  # noqa: BLE001 - aliases are unsupported on some backends
        LOGGER.warning("Could not set the %r alias: %s", stage_alias, exc)

    LOGGER.info("Registered %s version %s from run %s.", settings.mlflow_model_name, version.version, run_id)
    return RegisteredModel(
        name=settings.mlflow_model_name,
        version=str(version.version),
        run_id=run_id,
        feature_hash=tags["feature_hash"],
        decision_threshold=float(tags["decision_threshold"]),
        selection_metric=metric,
        selection_value=float(run.data.metrics.get(metric, float("nan"))),
    )


def load_champion(settings: Settings | None = None, alias: str = "champion") -> tuple[Any, dict[str, str]]:
    """Load the aliased model and its tags — what the scoring job calls."""
    import mlflow

    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = _client()

    version = client.get_model_version_by_alias(settings.mlflow_model_name, alias)
    tags = {**version.tags, "version": str(version.version)}
    # Load through the flavor that wrote it. The pyfunc wrapper around an
    # XGBClassifier returns class labels, not probabilities, which is exactly
    # the wrong thing for a ranked risk score.
    flavor = importlib.import_module(f"mlflow.{tags.get('model_flavor', 'sklearn')}")
    model = flavor.load_model(f"models:/{settings.mlflow_model_name}@{alias}")
    return model, tags


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="register this run")
    group.add_argument("--best", action="store_true", help="register the highest val PR-AUC run")
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)

    run_id = args.run_id or find_best_run(settings)
    registered = register(run_id, settings, stage_alias=args.alias)
    print(
        f"Registered {registered.name} v{registered.version} "
        f"({registered.selection_metric}={registered.selection_value:.4f}, "
        f"threshold={registered.decision_threshold:.4f}, features={registered.feature_hash})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
