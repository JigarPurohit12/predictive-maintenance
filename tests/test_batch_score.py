"""The scoring driver, end to end: curated tables in, work orders out."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from config import Settings
from features.contract import build_contract
from serving.batch_score import build_batch, run, score_locally, write_scores
from serving.inference import FeatureHashMismatch, ModelBundle
from tests.test_alerts import FakeSNS

AS_OF = datetime(2015, 3, 20, 12, 0, 0)


@pytest.fixture(scope="module")
def bundle(built_project: Settings) -> ModelBundle:
    """A model trained on the fixture, wrapped the way serving expects."""
    from features.build_features import load_matrix
    from training.splits import split_frames
    from training.train import fit_xgboost_tuned

    contract = build_contract(built_project)
    parts = split_frames(load_matrix(built_project))
    fitted = fit_xgboost_tuned(parts["train"], contract, built_project)

    return ModelBundle(
        model=fitted.estimator,
        feature_order=list(contract.order),
        feature_hash=contract.hash,
        decision_threshold=0.5,
        model_version="1",
        horizon_hours=built_project.prediction_horizon_hours,
        flavor="xgboost",
    )


# --- Building the batch ----------------------------------------------------


def test_the_batch_is_one_row_per_machine_at_the_latest_tick(built_project: Settings) -> None:
    features = build_batch(built_project, as_of=AS_OF)
    assert len(features) == 6  # the fixture fleet
    assert features["ts"].nunique() == 1
    assert features["ts"].max() <= AS_OF


def test_the_batch_carries_exactly_the_contracted_features(built_project: Settings) -> None:
    contract = build_contract(built_project)
    features = build_batch(built_project, as_of=AS_OF)
    assert list(features.columns) == ["machine_id", "ts", *contract.order]


# --- Scoring ---------------------------------------------------------------


def test_scoring_goes_through_the_real_handlers(built_project: Settings, bundle: ModelBundle) -> None:
    """Not a shortcut to predict_proba: the local path serialises and
    deserialises so the feature-hash assertion runs exactly as it will in the
    container. A payload bug surfaces here, not in a Glue job at 3am."""
    features = build_batch(built_project, as_of=AS_OF)
    scored = score_locally(features, bundle)

    assert len(scored) == len(features)
    assert scored["probability"].between(0.0, 1.0).all()
    assert (scored["feature_hash"] == bundle.feature_hash).all()


def test_a_batch_built_against_a_different_contract_is_refused(
    built_project: Settings, bundle: ModelBundle
) -> None:
    """The end-to-end version of the skew check: retrain the windows, forget to
    retrain the model, and the scoring job stops rather than lying."""
    features = build_batch(built_project, as_of=AS_OF)
    stale = ModelBundle(
        model=bundle.model,
        feature_order=bundle.feature_order,
        feature_hash="0000000000000000",
        decision_threshold=0.5,
        model_version="1",
        horizon_hours=24,
    )
    with pytest.raises(FeatureHashMismatch):
        score_locally(features, stale)


# --- The whole run ---------------------------------------------------------


def test_a_run_produces_scores_for_every_machine(built_project: Settings, bundle: ModelBundle) -> None:
    scores, _ = run(built_project, as_of=AS_OF, bundle=bundle)
    assert len(scores) == 6
    assert {score.machine_id for score in scores} == set(range(1, 7))
    assert all(score.model_version == "1" for score in scores)
    assert all(score.horizon_hours == built_project.prediction_horizon_hours for score in scores)


def test_a_low_threshold_raises_work_orders_and_publishes_them(
    built_project: Settings, bundle: ModelBundle
) -> None:
    generous = ModelBundle(**{**bundle.__dict__, "decision_threshold": 0.0})
    client = FakeSNS()

    scores, orders = run(
        built_project,
        as_of=AS_OF,
        bundle=generous,
        topic_arn="arn:aws:sns:us-east-1:123:pdm-work-orders",
        sns_client=client,
    )
    assert len(orders) == len(scores) == 6
    assert len(client.calls) == 6


def test_a_threshold_of_one_raises_nothing(built_project: Settings, bundle: ModelBundle) -> None:
    strict = ModelBundle(**{**bundle.__dict__, "decision_threshold": 1.0})
    client = FakeSNS()

    scores, orders = run(
        built_project, as_of=AS_OF, bundle=strict,
        topic_arn="arn:aws:sns:us-east-1:123:topic", sns_client=client,
    )
    assert scores and not orders
    assert client.calls == []


def test_work_orders_without_a_topic_are_logged_not_lost(
    built_project: Settings, bundle: ModelBundle, caplog: pytest.LogCaptureFixture
) -> None:
    generous = ModelBundle(**{**bundle.__dict__, "decision_threshold": 0.0})
    with caplog.at_level("INFO"):
        _, orders = run(built_project, as_of=AS_OF, bundle=generous, topic_arn=None)
    assert orders
    assert "no SNS topic configured" in caplog.text


def test_scoring_a_moment_with_no_history_returns_nothing(
    built_project: Settings, bundle: ModelBundle, caplog: pytest.LogCaptureFixture
) -> None:
    """Before any machine has cleared its warm-up there is nothing to score, and
    that is a warning rather than a crash."""
    with caplog.at_level("WARNING"):
        scores, orders = run(built_project, as_of=datetime(2015, 1, 1, 3), bundle=bundle)
    assert scores == [] and orders == []


# --- Output ----------------------------------------------------------------


def test_scores_are_written_to_a_dated_partition(built_project: Settings, bundle: ModelBundle) -> None:
    """The same layout batch transform writes on S3, so the local run and the
    managed run are readable by the same code."""
    scores, _ = run(built_project, as_of=AS_OF, bundle=bundle)
    path = write_scores(scores, built_project, AS_OF)

    assert path.parent.name == "dt=2015-03-20"
    written = pd.read_parquet(path)
    assert len(written) == 6
    assert {"machine_id", "probability", "decision", "model_version", "feature_hash"} <= set(written.columns)


def test_the_run_manifest_records_what_happened(built_project: Settings, bundle: ModelBundle) -> None:
    import json

    scores, _ = run(built_project, as_of=AS_OF, bundle=bundle)
    path = write_scores(scores, built_project, AS_OF)
    manifest = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))

    assert manifest["machines"] == 6
    assert manifest["feature_hash"] == bundle.feature_hash
    assert manifest["act"] + manifest["watch"] + manifest["ok"] == 6


# --- Registry integration --------------------------------------------------


def test_the_bundle_can_be_assembled_from_the_registry(built_project: Settings, tmp_path: Path) -> None:
    """The threshold and the hash come off the registered version's tags, not
    from config: the model was calibrated with a specific threshold and a
    specific column order, and re-deriving either would let them drift."""
    from serving.batch_score import load_bundle_from_registry
    from training.register import find_best_run, register
    from training.train import train

    train(built_project, ("xgboost",), track=True, artifact_dir=tmp_path)
    register(find_best_run(built_project), built_project)

    loaded = load_bundle_from_registry(built_project)
    assert loaded.feature_hash == build_contract(built_project).hash
    assert 0.0 <= loaded.decision_threshold <= 1.0
    assert loaded.horizon_hours == built_project.prediction_horizon_hours
    # The registry version, not the metric it won on: this string ends up on
    # every score row and has to identify the model.
    assert loaded.model_version.isdigit()


def test_a_stale_registered_hash_warns_before_it_fails(
    built_project: Settings, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from serving import batch_score

    monkeypatch.setattr(
        batch_score, "load_champion",
        lambda *_args, **_kwargs: (object(), {"feature_hash": "stale00000000000"}),
        raising=False,
    )
    monkeypatch.setattr(
        "training.register.load_champion",
        lambda *_args, **_kwargs: (object(), {"feature_hash": "stale00000000000"}),
    )
    with caplog.at_level("WARNING"):
        batch_score.load_bundle_from_registry(built_project)
    assert "retrain or check out the matching revision" in caplog.text
