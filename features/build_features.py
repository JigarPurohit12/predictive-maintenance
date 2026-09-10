"""The one feature path. Training uses it; scoring uses it; nothing else exists.

This is the single most important structural decision in the repository. If
features are computed in pandas for training and rewritten in PySpark for
scoring, the two implementations diverge, and nothing tells you — the model just
gets quietly worse. So the Glue job in phase 4 cleans and aggregates, and calls
:func:`compute_features` for anything that becomes a model input.

    python -m features.build_features            # training matrix -> features/

Two entry points over the same core:

* :func:`build_training_matrix` — full history, joined to the labels from
  ``sql/02_failure_labels.sql``, split into train/val/test parquet.
* :func:`build_scoring_matrix` — a window of recent history, no labels, filtered
  to the timestamps a scoring run is asked about.

``tests/test_parity.py`` runs both over the same input and asserts the feature
values are identical.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import Settings, configure_logging, get_settings
from data_layer.duckdb_io import load_staged, prepare
from features.contract import (
    ID_COLUMNS,
    LABEL_COLUMN,
    SPLIT_COLUMN,
    FeatureContract,
    FeatureContractError,
    build_contract,
)
from features.static import build_static_features
from features.windows import build_window_features
from training.splits import verify_time_splits

LOGGER = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"

#: The epoch anchor for the scoring grid. Identical to the one in
#: ``sql/02_failure_labels.sql``; ``tests/test_parity.py`` asserts the two
#: agree on real timestamps rather than trusting the comment.
GRID_EPOCH = pd.Timestamp("1970-01-01 00:00:00")


def on_cadence(timestamps: pd.Series, cadence_hours: int) -> pd.Series:
    """Which timestamps fall on the scoring grid.

    Anchored to the Unix epoch rather than to each machine's first reading, so
    every machine lands on the same ticks and a scoring run months later
    produces the same grid the training data was built on.
    """
    elapsed_hours = (timestamps - GRID_EPOCH) // pd.Timedelta(hours=1)
    return (elapsed_hours % cadence_hours == 0) & (timestamps.dt.minute == 0) & (timestamps.dt.second == 0)


def drop_warmup(frame: pd.DataFrame, max_window_hours: int) -> pd.DataFrame:
    """Remove each machine's first ``max_window_hours``.

    A 24-hour standard deviation computed from four observations is not a
    24-hour standard deviation. Dropping the warm-up is honest; filling it
    forwards would invent a stability the machine has not demonstrated.
    """
    first_seen = frame.groupby("machine_id", sort=False)["ts"].transform("min")
    keep = frame["ts"] >= first_seen + pd.Timedelta(hours=max_window_hours)
    dropped = int((~keep).sum())
    if dropped:
        LOGGER.info("Dropped %d warm-up row(s) (< %dh of history).", dropped, max_window_hours)
    return frame.loc[keep].reset_index(drop=True)


def compute_features(
    telemetry: pd.DataFrame,
    errors: pd.DataFrame,
    maint: pd.DataFrame,
    machines: pd.DataFrame,
    settings: Settings | None = None,
    contract: FeatureContract | None = None,
    *,
    drop_warmup_rows: bool = True,
) -> pd.DataFrame:
    """Telemetry plus context in, contracted feature matrix out.

    Computed on the **full hourly series**, not on the scoring grid: a 24-hour
    rolling mean sampled every three hours still needs all 24 hourly
    observations underneath it. Grid filtering happens afterwards.
    """
    settings = settings or get_settings()
    contract = contract or build_contract(settings)
    windows = list(settings.window_sizes_hours)

    LOGGER.info("Computing features over %d telemetry rows.", len(telemetry))
    window_features = build_window_features(telemetry, errors, windows, lag_periods=contract.lag_periods)
    static_features = build_static_features(
        window_features[list(ID_COLUMNS)],
        machines,
        maint,
        components=contract.components,
        machine_models=contract.machine_models,
    )

    frame = window_features.merge(static_features, on=list(ID_COLUMNS), how="left", validate="one_to_one")
    if drop_warmup_rows:
        frame = drop_warmup(frame, settings.max_window_hours)

    contract.validate_frame(frame)
    frame = frame[[*ID_COLUMNS, *contract.order]]
    return frame.sort_values(list(ID_COLUMNS), kind="stable", ignore_index=True)


def assert_complete(frame: pd.DataFrame, contract: FeatureContract) -> None:
    """No nulls and no infinities in the feature block.

    Allowed nowhere: every null here is either a warm-up row that should have
    been dropped or a join that silently missed. Both are bugs, and both look
    identical to the model.
    """
    block = frame.loc[:, list(contract.order)]
    null_counts = block.isna().sum()
    offenders = null_counts[null_counts > 0]
    if not offenders.empty:
        raise FeatureContractError(
            f"{int(offenders.sum())} null value(s) across {len(offenders)} feature(s): "
            f"{offenders.head(10).to_dict()}"
        )
    numeric = block.to_numpy(dtype="float64", copy=False)
    if not np.isfinite(numeric).all():
        columns = [name for name, ok in zip(contract.order, np.isfinite(numeric).all(axis=0), strict=True) if not ok]
        raise FeatureContractError(f"Non-finite values in feature(s): {columns[:10]}")


def build_training_matrix(
    settings: Settings | None = None,
    contract: FeatureContract | None = None,
) -> pd.DataFrame:
    """Features joined to the labels, restricted to the labelled grid."""
    settings = settings or get_settings()
    contract = contract or build_contract(settings)

    con = prepare(settings)
    staged = load_staged(con)
    labels = con.execute("SELECT machine_id, ts, label, split FROM labels_training").df()
    LOGGER.info("Loaded %d labelled grid rows.", len(labels))

    features = compute_features(
        staged["telemetry"], staged["errors"], staged["maint"], staged["machines"], settings, contract
    )
    matrix = labels.merge(features, on=list(ID_COLUMNS), how="inner", validate="one_to_one")
    LOGGER.info(
        "Feature matrix: %d rows (%d labelled rows had no feature row, usually warm-up).",
        len(matrix),
        len(labels) - len(matrix),
    )
    assert_complete(matrix, contract)
    # Cheap, and it catches the most expensive mistake in the project: a split
    # whose ranges overlap or whose gap does not clear the label horizon.
    verify_time_splits(matrix, settings)
    # Sorted, not merge-ordered: the parquet written from this must be
    # byte-reproducible across runs, and every reader sorts the same way.
    matrix = matrix.sort_values(list(ID_COLUMNS), kind="stable", ignore_index=True)
    return matrix[[*ID_COLUMNS, LABEL_COLUMN, SPLIT_COLUMN, *contract.order]]


def build_scoring_matrix(
    telemetry: pd.DataFrame,
    errors: pd.DataFrame,
    maint: pd.DataFrame,
    machines: pd.DataFrame,
    settings: Settings | None = None,
    contract: FeatureContract | None = None,
    *,
    as_of: datetime | None = None,
) -> pd.DataFrame:
    """Unlabelled features for a scoring run.

    ``telemetry`` must carry at least ``max_window_hours`` of history before the
    timestamps you want scored, or those rows are dropped as warm-up. ``as_of``
    keeps only the single most recent grid tick at or before it — what an
    hourly scoring job actually asks for.
    """
    settings = settings or get_settings()
    contract = contract or build_contract(settings)

    features = compute_features(telemetry, errors, maint, machines, settings, contract)
    features = features.loc[on_cadence(features["ts"], settings.feature_cadence_hours)].reset_index(drop=True)

    if as_of is not None:
        as_of = pd.Timestamp(as_of)
        features = features.loc[features["ts"] <= as_of]
        latest = features.groupby("machine_id", sort=False)["ts"].transform("max")
        features = features.loc[features["ts"] == latest].reset_index(drop=True)

    assert_complete(features, contract)
    return features


def write_matrix(matrix: pd.DataFrame, settings: Settings, contract: FeatureContract) -> Path:
    """Write the matrix partitioned by split, next to a manifest.

    The manifest is the thing that makes a matrix reproducible six months later:
    the feature hash, the ordered column list, the window sizes and the split
    dates that produced it.
    """
    destination = settings.features_dir
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    matrix.to_parquet(destination / "matrix", engine="pyarrow", index=False, partition_cols=[SPLIT_COLUMN])

    manifest = {
        "feature_hash": contract.hash,
        "n_features": contract.n_features,
        "feature_order": list(contract.order),
        "window_sizes_hours": list(contract.window_sizes_hours),
        "lag_periods": list(contract.lag_periods),
        "prediction_horizon_hours": settings.prediction_horizon_hours,
        "feature_cadence_hours": settings.feature_cadence_hours,
        "post_failure_exclusion_hours": settings.post_failure_exclusion_hours,
        "gap_hours": settings.gap_hours,
        "train_end_ts": settings.train_end_ts.isoformat(),
        "val_start_ts": settings.val_start_ts.isoformat(),
        "val_end_ts": settings.val_end_ts.isoformat(),
        "test_start_ts": settings.test_start_ts.isoformat(),
        "rows": len(matrix),
        "rows_by_split": {str(key): int(value) for key, value in matrix[SPLIT_COLUMN].value_counts().items()},
        "positives_by_split": {
            str(key): int(value) for key, value in matrix.groupby(SPLIT_COLUMN)[LABEL_COLUMN].sum().items()
        },
    }
    (destination / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d rows and %s to %s", len(matrix), MANIFEST_FILENAME, destination)
    return destination


def load_matrix(settings: Settings | None = None) -> pd.DataFrame:
    """Read the training matrix back, with the split column restored."""
    settings = settings or get_settings()
    path = settings.features_dir / "matrix"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run `python -m features.build_features` first.")
    frame = pd.read_parquet(path, engine="pyarrow")
    frame[SPLIT_COLUMN] = frame[SPLIT_COLUMN].astype("string")
    return frame.sort_values(list(ID_COLUMNS), kind="stable", ignore_index=True)


def load_manifest(settings: Settings | None = None) -> dict[str, object]:
    settings = settings or get_settings()
    path = settings.features_dir / MANIFEST_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", dest="write", action="store_false", help="compute and report, write nothing")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)
    contract = build_contract(settings)

    LOGGER.info("Feature contract %s: %d features.", contract.hash, contract.n_features)
    matrix = build_training_matrix(settings, contract)

    counts = matrix.groupby(SPLIT_COLUMN)[LABEL_COLUMN].agg(["count", "sum"])
    counts["rate"] = counts["sum"] / counts["count"]
    print(f"\nFeature contract {contract.hash} — {contract.n_features} features\n")
    print(counts.to_string())

    if args.write:
        write_matrix(matrix, settings, contract)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
