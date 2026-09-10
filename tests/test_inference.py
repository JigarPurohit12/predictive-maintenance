"""The SageMaker handlers, and the feature-hash assertion they exist for.

The mismatch tests are the point of this file. A reordered feature matrix
raises nothing on its own: XGBoost reads column 7 as column 7, the numbers stay
plausible, and the model quietly answers a different question. The only defence
is refusing the batch, and these tests prove it refuses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from features.contract import hash_order
from serving.inference import (
    CSV_CONTENT_TYPE,
    JSON_CONTENT_TYPE,
    JSONLINES_CONTENT_TYPE,
    FeatureHashMismatch,
    ModelBundle,
    build_payload,
    input_fn,
    model_fn,
    output_fn,
    predict_fn,
    save_bundle,
)

FEATURES = ["alpha", "beta", "gamma"]


def make_frame(rows: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.random((rows, len(FEATURES))), columns=FEATURES)
    frame.insert(0, "machine_id", range(1, rows + 1))
    frame.insert(1, "ts", pd.date_range("2015-11-01", periods=rows, freq="3h"))
    return frame


@pytest.fixture
def bundle() -> ModelBundle:
    rng = np.random.default_rng(1)
    features = rng.random((200, len(FEATURES)))
    labels = (features[:, 0] > 0.7).astype(int)
    model = LogisticRegression().fit(pd.DataFrame(features, columns=FEATURES), labels)
    return ModelBundle(
        model=model,
        feature_order=FEATURES,
        feature_hash=hash_order(FEATURES),
        decision_threshold=0.4,
        model_version="3",
        horizon_hours=24,
    )


# --- Payload round trip -----------------------------------------------------


def test_a_json_payload_round_trips_with_its_column_names() -> None:
    frame = make_frame()
    restored = input_fn(build_payload(frame, FEATURES), JSON_CONTENT_TYPE)
    assert list(restored.columns) == ["machine_id", "ts", *FEATURES]
    assert len(restored) == len(frame)
    assert restored.attrs["declared_feature_hash"] == hash_order(FEATURES)


def test_a_csv_payload_needs_a_header() -> None:
    """A headerless batch cannot be checked, and an unchecked batch is exactly
    what these handlers exist to prevent."""
    headerless = "1,2015-11-01,0.1,0.2,0.3\n2,2015-11-01,0.4,0.5,0.6\n"
    with pytest.raises(ValueError, match="header row"):
        input_fn(headerless, CSV_CONTENT_TYPE)


def test_a_csv_payload_with_a_header_is_accepted() -> None:
    frame = make_frame(2)
    restored = input_fn(frame.to_csv(index=False), CSV_CONTENT_TYPE)
    assert list(restored.columns) == ["machine_id", "ts", *FEATURES]


def test_an_unknown_content_type_is_refused() -> None:
    with pytest.raises(ValueError, match="Unsupported content type"):
        input_fn("{}", "application/xml")


def test_content_type_parameters_are_ignored() -> None:
    frame = make_frame(2)
    restored = input_fn(build_payload(frame, FEATURES), "application/json; charset=utf-8")
    assert len(restored) == 2


# --- Scoring ---------------------------------------------------------------


def test_predict_returns_a_probability_and_provenance_per_row(bundle: ModelBundle) -> None:
    frame = input_fn(build_payload(make_frame(), FEATURES), JSON_CONTENT_TYPE)
    scored = predict_fn(frame, bundle)

    assert list(scored.columns) == [
        "machine_id", "ts", "probability", "decision", "model_version", "feature_hash",
    ]
    assert len(scored) == 4
    assert scored["probability"].between(0.0, 1.0).all()
    assert (scored["model_version"] == "3").all()
    assert (scored["feature_hash"] == bundle.feature_hash).all()


def test_the_decision_follows_the_bundled_threshold(bundle: ModelBundle) -> None:
    frame = input_fn(build_payload(make_frame(20), FEATURES), JSON_CONTENT_TYPE)
    scored = predict_fn(frame, bundle)
    expected = np.where(scored["probability"] >= bundle.decision_threshold, "act", "ok")
    assert (scored["decision"].to_numpy() == expected).all()


# --- The assertion that matters --------------------------------------------


def test_a_reordered_batch_is_refused(bundle: ModelBundle) -> None:
    """The important case. The columns are all present and all correct — only
    the order differs, so nothing else in the stack would ever notice."""
    frame = make_frame()
    reordered = frame[["machine_id", "ts", "gamma", "alpha", "beta"]]
    payload = input_fn(build_payload(reordered, ["gamma", "alpha", "beta"]), JSON_CONTENT_TYPE)

    with pytest.raises(FeatureHashMismatch, match="different order"):
        predict_fn(payload, bundle)


def test_a_missing_feature_is_refused_and_named(bundle: ModelBundle) -> None:
    frame = make_frame().drop(columns=["beta"])
    payload = input_fn(build_payload(frame, ["alpha", "gamma"]), JSON_CONTENT_TYPE)

    with pytest.raises(FeatureHashMismatch, match="beta"):
        predict_fn(payload, bundle)


def test_an_extra_feature_is_refused_and_named(bundle: ModelBundle) -> None:
    frame = make_frame()
    frame["delta"] = 1.0
    payload = input_fn(build_payload(frame, [*FEATURES, "delta"]), JSON_CONTENT_TYPE)

    with pytest.raises(FeatureHashMismatch, match="delta"):
        predict_fn(payload, bundle)


def test_a_payload_that_lies_about_its_own_hash_is_refused(bundle: ModelBundle) -> None:
    payload = json.loads(build_payload(make_frame(), FEATURES))
    payload["feature_hash"] = "0000000000000000"
    frame = input_fn(json.dumps(payload), JSON_CONTENT_TYPE)

    with pytest.raises(FeatureHashMismatch, match="assembled inconsistently"):
        predict_fn(frame, bundle)


def test_a_matching_batch_passes(bundle: ModelBundle) -> None:
    frame = input_fn(build_payload(make_frame(), FEATURES), JSON_CONTENT_TYPE)
    assert len(predict_fn(frame, bundle)) == 4


# --- Serialisation out ------------------------------------------------------


def test_output_defaults_to_json_lines(bundle: ModelBundle) -> None:
    scored = predict_fn(input_fn(build_payload(make_frame(), FEATURES), JSON_CONTENT_TYPE), bundle)
    body, content_type = output_fn(scored)

    assert content_type == JSONLINES_CONTENT_TYPE
    lines = body.splitlines()
    assert len(lines) == 4
    assert json.loads(lines[0])["machine_id"] == 1


def test_output_can_be_csv(bundle: ModelBundle) -> None:
    scored = predict_fn(input_fn(build_payload(make_frame(), FEATURES), JSON_CONTENT_TYPE), bundle)
    body, content_type = output_fn(scored, CSV_CONTENT_TYPE)
    assert content_type == CSV_CONTENT_TYPE
    assert body.splitlines()[0].startswith("machine_id,ts,probability")


def test_an_unknown_accept_type_is_refused(bundle: ModelBundle) -> None:
    scored = predict_fn(input_fn(build_payload(make_frame(), FEATURES), JSON_CONTENT_TYPE), bundle)
    with pytest.raises(ValueError, match="Unsupported accept type"):
        output_fn(scored, "application/xml")


# --- The artifact bundle ----------------------------------------------------


def test_a_saved_bundle_reloads_with_its_contract(tmp_path: Path, bundle: ModelBundle) -> None:
    save_bundle(
        bundle.model,
        tmp_path / "artifact",
        feature_order=FEATURES,
        feature_hash=hash_order(FEATURES),
        decision_threshold=0.37,
        model_version="7",
        horizon_hours=24,
    )
    reloaded = model_fn(str(tmp_path / "artifact"))

    assert reloaded.feature_order == FEATURES
    assert reloaded.feature_hash == hash_order(FEATURES)
    assert reloaded.decision_threshold == pytest.approx(0.37)
    assert reloaded.model_version == "7"


def test_an_internally_inconsistent_bundle_is_refused(tmp_path: Path, bundle: ModelBundle) -> None:
    """A hash that does not match the order it claims to describe means the
    artifact was assembled by hand, and nothing downstream can be trusted."""
    directory = tmp_path / "artifact"
    save_bundle(
        bundle.model, directory,
        feature_order=FEATURES, feature_hash=hash_order(FEATURES),
        decision_threshold=0.4, model_version="1", horizon_hours=24,
    )
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    metadata["feature_hash"] = "deadbeefdeadbeef"
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(FeatureHashMismatch, match="internally inconsistent"):
        model_fn(str(directory))


def test_a_reloaded_bundle_scores_identically(tmp_path: Path, bundle: ModelBundle) -> None:
    save_bundle(
        bundle.model, tmp_path / "artifact",
        feature_order=FEATURES, feature_hash=hash_order(FEATURES),
        decision_threshold=0.4, model_version="1", horizon_hours=24,
    )
    reloaded = model_fn(str(tmp_path / "artifact"))

    frame = input_fn(build_payload(make_frame(), FEATURES), JSON_CONTENT_TYPE)
    np.testing.assert_allclose(
        predict_fn(frame, bundle)["probability"], predict_fn(frame, reloaded)["probability"], rtol=1e-9
    )
