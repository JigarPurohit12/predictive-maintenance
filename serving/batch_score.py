"""The scoring driver: fresh telemetry in, work orders out.

    python -m serving.batch_score --local            # score in-process
    python -m serving.batch_score --as-of 2015-11-01T12:00:00

Two ways to run the same pipeline:

* ``--local`` loads the registered model and calls the same handlers the
  container calls. This is what runs in phases 1-3, and what the tests exercise.
* the default launches a SageMaker **batch transform** — a job that spins up,
  scores, writes to S3 and shuts down.

There is no real-time endpoint, and adding one would be a mistake. Failure
horizons here are measured in hours; an endpoint bills 24/7 to answer a question
nobody asks more than once every three. Leaving one running is the single most
common way a portfolio project produces a surprise bill.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings, configure_logging, get_settings
from data_layer.duckdb_io import load_staged, prepare
from features.build_features import build_scoring_matrix
from features.contract import build_contract
from serving.alerts import (
    RiskScore,
    WorkOrder,
    build_scores,
    format_summary,
    publish,
    scores_to_frame,
    summarise,
    to_work_orders,
)
from serving.inference import ModelBundle, build_payload, input_fn, predict_fn

LOGGER = logging.getLogger(__name__)

RUN_MANIFEST = "run.json"


def load_bundle_from_registry(settings: Settings | None = None, alias: str = "champion") -> ModelBundle:
    """Assemble a :class:`ModelBundle` from the MLflow registry.

    The threshold and the feature hash come off the registered *version's* tags,
    not from config: the model was calibrated with a specific threshold and a
    specific column order, and re-deriving either at scoring time would let them
    drift apart from the artifact they belong to.
    """
    from training.register import load_champion

    settings = settings or get_settings()
    model, tags = load_champion(settings, alias=alias)
    contract = build_contract(settings)

    if tags.get("feature_hash") != contract.hash:
        LOGGER.warning(
            "The registered model was trained on feature hash %s but this checkout builds %s. "
            "Scoring will refuse the batch; retrain or check out the matching revision.",
            tags.get("feature_hash"),
            contract.hash,
        )

    return ModelBundle(
        model=model,
        feature_order=list(contract.order),
        feature_hash=tags.get("feature_hash", contract.hash),
        decision_threshold=float(tags.get("decision_threshold", settings.decision_threshold)),
        # The registry version, not the metric it won on. This is what goes on
        # every score row and what answers "which model said that, in March?".
        model_version=tags.get("version", "unknown"),
        horizon_hours=int(tags.get("horizon_hours", settings.prediction_horizon_hours)),
        flavor=tags.get("model_flavor", "sklearn"),
    )


def build_batch(
    settings: Settings | None = None,
    *,
    as_of: datetime | None = None,
) -> pd.DataFrame:
    """The feature matrix for one scoring run, from the curated tables.

    Uses :func:`features.build_features.build_scoring_matrix` — the same module
    training uses. There is no second implementation and there must never be
    one; ``tests/test_parity.py`` is what keeps that true.
    """
    settings = settings or get_settings()
    contract = build_contract(settings)
    staged = load_staged(prepare(settings))
    return build_scoring_matrix(
        staged["telemetry"], staged["errors"], staged["maint"], staged["machines"],
        settings, contract, as_of=as_of,
    )


def score_locally(features: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """Round-trip the batch through the real handlers, in process.

    Deliberately not a shortcut that calls ``model.predict_proba`` directly:
    going through serialise -> input_fn -> predict_fn means the local path
    exercises the feature-hash assertion exactly as the container does, so a
    payload bug surfaces here rather than in a Glue job at 3am.
    """
    payload = build_payload(features, bundle.feature_order)
    frame = input_fn(payload, "application/json")
    return predict_fn(frame, bundle)


def run(
    settings: Settings | None = None,
    *,
    as_of: datetime | None = None,
    alias: str = "champion",
    topic_arn: str | None = None,
    sns_client: Any | None = None,
    bundle: ModelBundle | None = None,
) -> tuple[list[RiskScore], list[WorkOrder]]:
    """Score the fleet and raise work orders. The whole scoring path."""
    settings = settings or get_settings()
    bundle = bundle or load_bundle_from_registry(settings, alias=alias)

    features = build_batch(settings, as_of=as_of)
    if features.empty:
        LOGGER.warning("No machines have enough history to score at %s.", as_of)
        return [], []

    predictions = score_locally(features, bundle)
    scored_at = as_of or pd.Timestamp(features["ts"].max()).to_pydatetime()

    drivers = _drivers(bundle, features)
    scores = build_scores(
        predictions,
        threshold=bundle.decision_threshold,
        model_version=bundle.model_version,
        feature_hash=bundle.feature_hash,
        horizon_hours=bundle.horizon_hours,
        scored_at=scored_at,
        drivers=drivers,
    )
    orders = to_work_orders(scores, threshold=bundle.decision_threshold, settings=settings)

    if orders and topic_arn:
        publish(orders, topic_arn, settings, client=sns_client)
    elif orders:
        LOGGER.info("%d work order(s) raised but no SNS topic configured.", len(orders))

    return scores, orders


def _drivers(bundle: ModelBundle, features: pd.DataFrame) -> list[list[str]] | None:
    from training.evaluate import top_shap_drivers

    try:
        return top_shap_drivers(bundle.model, features, bundle.feature_order)
    except Exception as exc:  # noqa: BLE001 - an explanation is never worth failing a score over
        LOGGER.warning("Could not compute drivers: %s", exc)
        return None


def write_scores(scores: list[RiskScore], settings: Settings, scored_at: datetime) -> Path:
    """Write to ``scores/dt=.../`` — the same layout batch transform writes on S3."""
    partition = settings.scores_dir / f"dt={scored_at:%Y-%m-%d}"
    partition.mkdir(parents=True, exist_ok=True)
    path = partition / f"scores-{scored_at:%H%M%S}.parquet"
    scores_to_frame(scores).to_parquet(path, engine="pyarrow", index=False)

    (partition / RUN_MANIFEST).write_text(
        json.dumps({"scored_at": scored_at.isoformat(), **summarise(scores)}, indent=2, default=str),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %d score(s) to %s", len(scores), path)
    return path


def launch_transform(settings: Settings | None = None, *, as_of: datetime | None = None) -> dict[str, Any]:
    """Launch the managed batch transform. See ``aws/sagemaker/transform_job.py``."""
    from aws.sagemaker.transform_job import launch

    settings = settings or get_settings()
    return launch(settings, as_of=as_of)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true", help="score in-process instead of on SageMaker")
    parser.add_argument("--as-of", type=datetime.fromisoformat, default=None)
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--topic-arn", default=None, help="publish work orders to this SNS topic")
    parser.add_argument("--no-write", dest="write", action="store_false")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)

    if not args.local:
        response = launch_transform(settings, as_of=args.as_of)
        print(json.dumps(response, indent=2, default=str))
        return 0

    scores, orders = run(settings, as_of=args.as_of, alias=args.alias, topic_arn=args.topic_arn)
    print(format_summary(summarise(scores)))
    for order in orders:
        print(f"  {order.priority} machine {order.machine_id}: {order.reason}")

    if args.write and scores:
        write_scores(scores, settings, scores[0].scored_at)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
