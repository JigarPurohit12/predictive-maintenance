"""SageMaker script-mode inference handlers.

SageMaker calls four functions, in this order, for every invocation::

    model_fn(model_dir)                -> once, at container start
    input_fn(body, content_type)       -> bytes  -> DataFrame
    predict_fn(data, model)            -> DataFrame -> scores
    output_fn(scores, accept)          -> scores -> bytes

The load-bearing line in this module is the ``feature_hash`` assertion in
:func:`predict_fn`. XGBoost fed a numpy array reads column 7 as column 7 — if
the batch was built with the columns in a different order, or with a feature
added, nothing raises, the numbers stay plausible, and the model quietly starts
answering a different question. The bundle carries the hash of the column order
it was trained on, every batch carries the hash of the column order it was built
with, and a mismatch fails the job loudly.

**Payload format.** The primary content type is ``application/json``::

    {"feature_hash": "...", "columns": [...], "rows": [[...], ...]}

carrying the column names explicitly, because that is what makes the hash check
possible. Bare ``text/csv`` is accepted with a header row for manual testing,
and refused without one — a headerless CSV cannot be checked, and an unchecked
batch is the failure mode this module exists to prevent. Configure the transform
job with ``SplitType: None`` so each file arrives as one payload with its header
intact; at fleet scale a scoring batch is a few hundred rows.
"""

from __future__ import annotations

import importlib
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.contract import hash_order

LOGGER = logging.getLogger(__name__)

JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"
JSONLINES_CONTENT_TYPE = "application/jsonlines"

METADATA_FILENAME = "metadata.json"
ID_COLUMN = "machine_id"
TIMESTAMP_COLUMN = "ts"


class FeatureHashMismatch(ValueError):
    """The batch was not built with the column order the model was trained on."""


@dataclass(frozen=True)
class ModelBundle:
    """A model plus the two things it cannot be deployed without."""

    model: Any
    feature_order: list[str]
    feature_hash: str
    decision_threshold: float
    model_version: str
    horizon_hours: int
    flavor: str = "sklearn"

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        ordered = frame.loc[:, self.feature_order]
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(ordered))[:, 1]
        return np.asarray(self.model.predict(ordered), dtype="float64")


def save_bundle(
    model: Any,
    model_dir: Path,
    *,
    feature_order: list[str],
    feature_hash: str,
    decision_threshold: float,
    model_version: str,
    horizon_hours: int,
    flavor: str = "sklearn",
) -> Path:
    """Write the artifact SageMaker will untar into ``/opt/ml/model``.

    The metadata sits next to the model rather than inside it so it can be read
    without deserialising anything — useful when the thing you are debugging is
    the deserialisation.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    mlflow_flavor = importlib.import_module(f"mlflow.{flavor}")
    mlflow_flavor.save_model(model, path=str(model_dir / "model"))

    metadata = {
        "feature_order": list(feature_order),
        "feature_hash": feature_hash,
        "decision_threshold": decision_threshold,
        "model_version": model_version,
        "horizon_hours": horizon_hours,
        "flavor": flavor,
    }
    (model_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_dir


# --- The four SageMaker handlers -------------------------------------------


def model_fn(model_dir: str) -> ModelBundle:
    """Called once at container start."""
    path = Path(model_dir)
    metadata = json.loads((path / METADATA_FILENAME).read_text(encoding="utf-8"))

    flavor = metadata.get("flavor", "sklearn")
    mlflow_flavor = importlib.import_module(f"mlflow.{flavor}")
    model = mlflow_flavor.load_model(str(path / "model"))

    expected = hash_order(metadata["feature_order"])
    if expected != metadata["feature_hash"]:
        raise FeatureHashMismatch(
            f"The bundle is internally inconsistent: metadata records {metadata['feature_hash']} "
            f"but its own feature_order hashes to {expected}."
        )

    LOGGER.info(
        "Loaded model version %s (%d features, hash %s, threshold %.4f).",
        metadata["model_version"],
        len(metadata["feature_order"]),
        metadata["feature_hash"],
        metadata["decision_threshold"],
    )
    return ModelBundle(
        model=model,
        feature_order=list(metadata["feature_order"]),
        feature_hash=metadata["feature_hash"],
        decision_threshold=float(metadata["decision_threshold"]),
        model_version=str(metadata["model_version"]),
        horizon_hours=int(metadata["horizon_hours"]),
        flavor=flavor,
    )


def input_fn(request_body: bytes | str, content_type: str = JSON_CONTENT_TYPE) -> pd.DataFrame:
    """Deserialise a batch, keeping the column names.

    The frame carries a ``feature_hash`` attribute where the payload declared
    one; :func:`predict_fn` checks it. A payload that declares nothing is still
    checked, against the hash of the column names it actually shipped.
    """
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    content_type = (content_type or JSON_CONTENT_TYPE).split(";")[0].strip()

    if content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        frame = pd.DataFrame(payload["rows"], columns=payload["columns"])
        declared = payload.get("feature_hash")
    elif content_type == JSONLINES_CONTENT_TYPE:
        records = [json.loads(line) for line in request_body.splitlines() if line.strip()]
        frame = pd.DataFrame(records)
        declared = None
    elif content_type == CSV_CONTENT_TYPE:
        frame = pd.read_csv(io.StringIO(request_body))
        if frame.columns.str.match(r"^(Unnamed|\d)").any():
            raise ValueError(
                "text/csv payloads must carry a header row. A headerless batch cannot be checked "
                "against the model's feature hash, and an unchecked batch is exactly what this "
                "handler exists to prevent — send application/json instead."
            )
        declared = None
    else:
        raise ValueError(f"Unsupported content type {content_type!r}.")

    frame.attrs["declared_feature_hash"] = declared
    return frame


def predict_fn(data: pd.DataFrame, model: ModelBundle) -> pd.DataFrame:
    """Score a batch, refusing anything whose columns do not match the model."""
    identifiers = [column for column in (ID_COLUMN, TIMESTAMP_COLUMN) if column in data.columns]
    feature_columns = [column for column in data.columns if column not in identifiers]

    declared = data.attrs.get("declared_feature_hash")
    observed = hash_order(feature_columns)
    if declared is not None and declared != observed:
        raise FeatureHashMismatch(
            f"The payload declares feature_hash {declared} but its own columns hash to {observed}. "
            f"The batch was assembled inconsistently."
        )
    if observed != model.feature_hash:
        missing = sorted(set(model.feature_order) - set(feature_columns))
        extra = sorted(set(feature_columns) - set(model.feature_order))
        raise FeatureHashMismatch(
            f"Batch feature hash {observed} does not match the model's {model.feature_hash}. "
            f"Missing: {missing[:5] or 'none'}. Unexpected: {extra[:5] or 'none'}. "
            f"If both are empty the columns are the same but in a different order, which is worse: "
            f"nothing would have raised and every score would have been wrong."
        )

    probabilities = model.probabilities(data)
    out = pd.DataFrame({"probability": probabilities})
    for column in identifiers:
        out[column] = data[column].to_numpy()
    out["decision"] = np.where(probabilities >= model.decision_threshold, "act", "ok")
    out["model_version"] = model.model_version
    out["feature_hash"] = model.feature_hash
    return out[[*identifiers, "probability", "decision", "model_version", "feature_hash"]]


def output_fn(prediction: pd.DataFrame, accept: str = JSONLINES_CONTENT_TYPE) -> tuple[str, str]:
    """Serialise scores. Returns ``(body, content_type)``, as SageMaker expects."""
    accept = (accept or JSONLINES_CONTENT_TYPE).split(";")[0].strip()
    if accept in (JSONLINES_CONTENT_TYPE, JSON_CONTENT_TYPE):
        body = "\n".join(prediction.to_json(orient="records", lines=True).splitlines())
        return body, JSONLINES_CONTENT_TYPE
    if accept == CSV_CONTENT_TYPE:
        return prediction.to_csv(index=False), CSV_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type {accept!r}.")


def build_payload(frame: pd.DataFrame, feature_order: list[str]) -> str:
    """Serialise a feature matrix into the format :func:`input_fn` expects.

    Used by the batch driver and by tests. Keeping it here rather than in the
    driver means the writer and the reader of the payload cannot drift apart.
    """
    identifiers = [column for column in (ID_COLUMN, TIMESTAMP_COLUMN) if column in frame.columns]
    columns = [*identifiers, *feature_order]
    ordered = frame.loc[:, columns].copy()
    if TIMESTAMP_COLUMN in ordered.columns:
        ordered[TIMESTAMP_COLUMN] = ordered[TIMESTAMP_COLUMN].astype("string")
    return json.dumps(
        {
            "feature_hash": hash_order(feature_order),
            "columns": columns,
            "rows": ordered.to_numpy(dtype=object).tolist(),
        },
        default=str,
    )
