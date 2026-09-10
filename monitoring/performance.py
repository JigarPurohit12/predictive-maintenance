"""Delayed-label evaluation: how the model actually did, once the answers arrive.

    python -m monitoring.performance --as-of 2015-11-05

A prediction made at time *t* cannot be scored at time *t*. It says something
about the window ``(t, t + horizon]``, and until that window has fully elapsed
the outcome is not known — a machine that has not failed yet may still fail in
four hours. Scoring predictions before their horizon closes counts every
not-yet-failure as a false alarm and makes a working model look broken.

So this job runs on a lag: it takes yesterday's predictions, joins them to the
failures that have since been recorded, and appends a row to a running metrics
table. That table is the only honest answer to "is the model still working",
and almost nobody builds it in a portfolio project.

Two things it is careful about:

* **Ripeness.** Only predictions whose horizon has closed are scored. The rest
  wait, and are counted as pending rather than dropped.
* **The threshold.** Recall and precision are computed at the threshold that was
  actually in force when the prediction was made, which is carried on the score
  row — not at whatever the current champion uses. Re-scoring history at today's
  threshold is how a regression gets hidden.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from config import Settings, configure_logging, get_settings
from data_layer.duckdb_io import load_staged, prepare
from training.imbalance import ranking_metrics

LOGGER = logging.getLogger(__name__)

METRICS_FILENAME = "performance_history.jsonl"


class PerformanceRecord(BaseModel):
    """One evaluation period, ready to append to the running table."""

    model_config = ConfigDict(frozen=True)

    evaluated_at: datetime
    period_start: datetime
    period_end: datetime
    model_version: str
    feature_hash: str
    threshold: float
    horizon_hours: int
    scored_rows: int
    pending_rows: int
    positives: int
    base_rate: float
    pr_auc: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    recall: float
    precision: float
    alarms_per_machine_month: float | None = None

    def to_json_line(self) -> str:
        return self.model_dump_json()


def ripe(scores: pd.DataFrame, as_of: datetime, horizon_hours: int) -> pd.Series:
    """Which predictions have had their whole horizon observed by ``as_of``.

    A prediction at ``ts`` is answerable once ``ts + horizon <= as_of``. Anything
    later is still pending — not a negative.
    """
    deadline = pd.Timestamp(as_of) - pd.Timedelta(hours=horizon_hours)
    return pd.to_datetime(scores["scored_at"]) <= deadline


def attach_outcomes(
    scores: pd.DataFrame,
    failures: pd.DataFrame,
    *,
    horizon_hours: int,
) -> pd.DataFrame:
    """Join each prediction to what actually happened in its horizon.

    The outcome rule is deliberately identical to the training label in
    ``sql/02_failure_labels.sql``: positive when a failure falls in
    ``(ts, ts + horizon]``, open on the left. Evaluating against a different
    rule than the one the model was trained on measures the difference between
    the rules, not the model.
    """
    out = scores.copy()
    out["scored_at"] = pd.to_datetime(out["scored_at"])
    # Score parquet and the staged tables do not agree on the width of
    # machine_id, and merge_asof refuses to join int32 to int64 rather than
    # coercing. Normalise both sides here instead of hoping they match.
    out["machine_id"] = out["machine_id"].astype("int64")

    if failures.empty:
        out["actual"] = 0
        out["next_failure_ts"] = pd.NaT
        return out

    moments = (
        failures[["machine_id", "ts"]]
        .drop_duplicates()
        .assign(machine_id=lambda frame: frame["machine_id"].astype("int64"))
        .sort_values("ts")
        .rename(columns={"ts": "next_failure_ts"})
    )
    merged = pd.merge_asof(
        out.sort_values("scored_at"),
        moments,
        left_on="scored_at",
        right_on="next_failure_ts",
        by="machine_id",
        direction="forward",
        allow_exact_matches=False,  # open on the left, as the training label is
    )
    elapsed = (merged["next_failure_ts"] - merged["scored_at"]).dt.total_seconds() / 3_600.0
    merged["actual"] = ((elapsed > 0) & (elapsed <= horizon_hours)).fillna(False).astype(int)
    return merged


def evaluate_period(
    scores: pd.DataFrame,
    failures: pd.DataFrame,
    *,
    as_of: datetime,
    horizon_hours: int,
    settings: Settings | None = None,
) -> PerformanceRecord | None:
    """Score every ripe prediction. Returns None when nothing is ripe yet."""
    settings = settings or get_settings()
    if scores.empty:
        LOGGER.info("No predictions to evaluate.")
        return None

    is_ripe = ripe(scores, as_of, horizon_hours)
    pending = int((~is_ripe).sum())
    mature = scores.loc[is_ripe]
    if mature.empty:
        LOGGER.info("None of the %d prediction(s) have had their horizon close yet.", pending)
        return None

    joined = attach_outcomes(mature, failures, horizon_hours=horizon_hours)
    actual = joined["actual"].to_numpy()
    probability = joined["probability"].to_numpy(dtype="float64")

    # The threshold in force when the prediction was made, not today's.
    threshold = float(joined["threshold"].iloc[0]) if "threshold" in joined.columns else _implied_threshold(joined)
    predicted = probability >= threshold

    true_positives = int(np.sum(predicted & (actual == 1)))
    false_positives = int(np.sum(predicted & (actual == 0)))
    false_negatives = int(np.sum(~predicted & (actual == 1)))
    true_negatives = int(np.sum(~predicted & (actual == 0)))

    span_hours = float(
        (joined["scored_at"].max() - joined["scored_at"].min()).total_seconds() / 3_600.0
    )
    machines = int(joined["machine_id"].nunique())
    machine_months = machines * (span_hours / 730.0) if span_hours else 0.0

    ranking = ranking_metrics(actual, probability)
    return PerformanceRecord(
        evaluated_at=datetime.now(),
        period_start=joined["scored_at"].min().to_pydatetime(),
        period_end=joined["scored_at"].max().to_pydatetime(),
        model_version=str(joined["model_version"].iloc[0]) if "model_version" in joined else "unknown",
        feature_hash=str(joined["feature_hash"].iloc[0]) if "feature_hash" in joined else "unknown",
        threshold=threshold,
        horizon_hours=horizon_hours,
        scored_rows=len(joined),
        pending_rows=pending,
        positives=int(actual.sum()),
        base_rate=float(actual.mean()),
        pr_auc=ranking["pr_auc"],
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        recall=true_positives / max(true_positives + false_negatives, 1),
        precision=true_positives / max(true_positives + false_positives, 1),
        alarms_per_machine_month=false_positives / machine_months if machine_months else None,
    )


def _implied_threshold(joined: pd.DataFrame) -> float:
    """Recover the threshold from the decisions the scoring run recorded.

    Score rows carry ``decision``, not the threshold that produced it. The
    lowest probability that was called ``act`` is the threshold, to within the
    resolution of the batch — and reading it back this way means a historical
    period is always evaluated at its own threshold even when nobody wrote it
    down.
    """
    if "decision" not in joined.columns:
        return 0.5
    acted = joined.loc[joined["decision"] == "act", "probability"]
    return float(acted.min()) if not acted.empty else float("inf")


def load_scores(settings: Settings, *, since: datetime | None = None) -> pd.DataFrame:
    """Read every written scoring run from ``scores/dt=.../``."""
    root = settings.scores_dir
    if not root.exists():
        return pd.DataFrame()

    frames = [pd.read_parquet(path) for path in sorted(root.glob("dt=*/*.parquet"))]
    if not frames:
        return pd.DataFrame()

    scores = pd.concat(frames, ignore_index=True)
    scores["scored_at"] = pd.to_datetime(scores["scored_at"])
    if since is not None:
        scores = scores[scores["scored_at"] >= pd.Timestamp(since)]
    return scores.reset_index(drop=True)


def append_record(record: PerformanceRecord, path: Path) -> Path:
    """Append one row to the running metrics table.

    JSON Lines, appended never rewritten: the history of how the model performed
    is not something a later run should be able to edit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.to_json_line() + "\n")
    LOGGER.info("Appended a performance record to %s", path)
    return path


def load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(records)


def format_record(record: PerformanceRecord) -> str:
    return "\n".join(
        [
            f"Delayed-label evaluation — {record.period_start:%Y-%m-%d %H:%M} to {record.period_end:%Y-%m-%d %H:%M}",
            f"  model version      {record.model_version} (features {record.feature_hash})",
            f"  threshold          {record.threshold:.4f}",
            f"  scored / pending   {record.scored_rows:,} / {record.pending_rows:,}",
            f"  base rate          {record.base_rate:.4%} ({record.positives} failures)",
            f"  PR-AUC             {record.pr_auc:.4f}",
            f"  recall             {record.recall:.2%}",
            f"  precision          {record.precision:.2%}",
            f"  confusion          tp={record.true_positives} fp={record.false_positives} "
            f"tn={record.true_negatives} fn={record.false_negatives}",
            (
                f"  alarms/machine/mo  {record.alarms_per_machine_month:.2f}"
                if record.alarms_per_machine_month is not None
                else "  alarms/machine/mo  n/a"
            ),
        ]
    )


def run(
    settings: Settings | None = None,
    *,
    as_of: datetime | None = None,
    since: datetime | None = None,
) -> PerformanceRecord | None:
    settings = settings or get_settings()
    as_of = as_of or datetime.now()

    scores = load_scores(settings, since=since)
    failures = load_staged(prepare(settings))["failures"]
    return evaluate_period(
        scores, failures, as_of=as_of, horizon_hours=settings.prediction_horizon_hours, settings=settings
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-of", type=datetime.fromisoformat, default=None)
    parser.add_argument("--since", type=datetime.fromisoformat, default=None)
    parser.add_argument("--history", type=Path, default=Path("results") / METRICS_FILENAME)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)

    record = run(settings, as_of=args.as_of, since=args.since)
    if record is None:
        print("Nothing ripe to evaluate yet — every prediction's horizon is still open.")
        return 0

    print(format_record(record))
    append_record(record, args.history)

    history = load_history(args.history)
    if len(history) > 1:
        print(f"\n{len(history)} evaluation periods recorded:")
        print(history[["period_end", "model_version", "pr_auc", "recall", "precision"]].to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


def default_history_path() -> Path:
    return Path("results") / METRICS_FILENAME
