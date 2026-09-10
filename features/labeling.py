"""Build the labelled feature grid and report on it.

The labelling rules themselves live in ``sql/02_failure_labels.sql`` and nowhere
else. This module runs that SQL, writes the result to parquet for phase 2, and
prints the class balance that phase 1 is judged on::

    python -m features.labeling                 # report only
    python -m features.labeling --write         # also write curated/labels/

Three rules decide which grid rows survive, and the report accounts for each of
them separately because getting any one wrong shows up the same way — a positive
rate that is obviously wrong:

* **horizon** — positive when a failure falls in ``(ts, ts + horizon]``
* **exclusion** — rows in ``[failure, failure + exclusion]`` are dropped
* **observability** — the last ``horizon`` hours cannot be labelled at all
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
from pydantic import BaseModel, ConfigDict

from config import SPLIT_NAMES, Settings, configure_logging, get_settings
from data_layer.duckdb_io import prepare

LOGGER = logging.getLogger(__name__)

#: The positive rate this problem should land in. The spec says 1-3%; the gate
#: is wider so it flags a mistake rather than a rounding difference. A rate of
#: 30% means the horizon or the exclusion logic is inverted — see the module
#: docstring for the three rules and check them in that order.
EXPECTED_POSITIVE_RATE = (0.005, 0.05)

RESULTS_FILENAME = "phase1_label_balance.json"


class SplitBalance(BaseModel):
    """Class balance for one split period."""

    model_config = ConfigDict(frozen=True)

    split: str
    rows: int
    positives: int
    positive_rate: float
    ts_min: datetime | None
    ts_max: datetime | None
    machines: int


class LabelSummary(BaseModel):
    """Everything phase 1 has to be able to answer with a number."""

    model_config = ConfigDict(frozen=True)

    horizon_hours: int
    cadence_hours: int
    exclusion_hours: int
    gap_hours: int
    grid_rows: int
    dropped_excluded: int
    dropped_unlabelable: int
    dropped_gap: int
    kept_rows: int
    positives: int
    positive_rate: float
    by_split: tuple[SplitBalance, ...]
    by_component: tuple[dict[str, object], ...]

    @property
    def within_expected_range(self) -> bool:
        low, high = EXPECTED_POSITIVE_RATE
        return low <= self.positive_rate <= high


def build(settings: Settings | None = None, frames: Mapping[str, pd.DataFrame] | None = None) -> duckdb.DuckDBPyConnection:
    """A connection with staging and the label views built. See :mod:`data_layer.duckdb_io`."""
    return prepare(settings, frames)


def load_labels(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Every grid row with its flags, before anything is dropped."""
    return con.execute("SELECT * FROM labels ORDER BY machine_id, ts").df()


def load_training_labels(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """The rows that reach phase 2: exclusions, unlabelable tail and gaps removed."""
    return con.execute("SELECT * FROM labels_training ORDER BY machine_id, ts").df()


def drop_accounting(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """How many grid rows each rule removes.

    The counts overlap — a row can be both excluded and in a gap — so they are
    reported as "removed by this rule", not as a partition of the total.
    """
    row = con.execute(
        """
        SELECT
            count(*)                                              AS grid_rows,
            sum(CASE WHEN excluded THEN 1 ELSE 0 END)             AS dropped_excluded,
            sum(CASE WHEN NOT labelable THEN 1 ELSE 0 END)        AS dropped_unlabelable,
            sum(CASE WHEN split = 'gap' THEN 1 ELSE 0 END)        AS dropped_gap,
            sum(CASE WHEN NOT excluded AND labelable AND split <> 'gap' THEN 1 ELSE 0 END) AS kept_rows
        FROM labels
        """
    ).fetchone()
    assert row is not None  # a single-row aggregate always returns a row
    keys = ("grid_rows", "dropped_excluded", "dropped_unlabelable", "dropped_gap", "kept_rows")
    return {key: int(value or 0) for key, value in zip(keys, row, strict=True)}


def class_balance_by_split(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Rows, positives and positive rate per split period."""
    frame = con.execute(
        """
        SELECT
            split,
            count(*)                        AS "rows",
            sum(label)                      AS positives,
            min(ts)                         AS ts_min,
            max(ts)                         AS ts_max,
            count(DISTINCT machine_id)      AS machines
        FROM labels_training
        GROUP BY split
        """
    ).df()
    frame["positives"] = frame["positives"].fillna(0).astype(int)
    frame["positive_rate"] = frame["positives"] / frame["rows"]
    order = {name: index for index, name in enumerate(SPLIT_NAMES)}
    frame = frame.sort_values("split", key=lambda column: column.map(order))
    return frame.reset_index(drop=True)


def class_balance_by_component(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Positive rate per component per split.

    The modelled target is any-component failure; this is a diagnostic. A
    component whose rate is zero in one split has been eaten by the exclusion or
    horizon logic, or genuinely never fails in that period — either way you want
    to know before phase 2, not after.
    """
    frame = con.execute(
        """
        SELECT
            component,
            split,
            count(*)   AS "rows",
            sum(label) AS positives
        FROM labels_by_component
        GROUP BY component, split
        ORDER BY component, split
        """
    ).df()
    frame["positives"] = frame["positives"].fillna(0).astype(int)
    frame["positive_rate"] = frame["positives"] / frame["rows"]
    return frame


def summarise(con: duckdb.DuckDBPyConnection, settings: Settings | None = None) -> LabelSummary:
    settings = settings or get_settings()
    counts = drop_accounting(con)
    by_split = class_balance_by_split(con)
    by_component = class_balance_by_component(con)

    positives = int(by_split["positives"].sum())
    kept = counts["kept_rows"]
    return LabelSummary(
        horizon_hours=settings.prediction_horizon_hours,
        cadence_hours=settings.feature_cadence_hours,
        exclusion_hours=settings.post_failure_exclusion_hours,
        gap_hours=settings.gap_hours,
        **counts,
        positives=positives,
        positive_rate=positives / kept if kept else 0.0,
        by_split=tuple(
            SplitBalance(
                split=row.split,
                rows=int(row.rows),
                positives=int(row.positives),
                positive_rate=float(row.positive_rate),
                ts_min=row.ts_min,
                ts_max=row.ts_max,
                machines=int(row.machines),
            )
            for row in by_split.itertuples(index=False)
        ),
        by_component=tuple(by_component.to_dict(orient="records")),
    )


def format_report(summary: LabelSummary) -> str:
    """The phase 1 acceptance output, as text."""
    lines = [
        "",
        "Problem definition",
        "------------------",
        f"  horizon            {summary.horizon_hours}h   (positive if a failure falls in (ts, ts+{summary.horizon_hours}h])",
        f"  scoring cadence    {summary.cadence_hours}h",
        f"  post-failure drop  {summary.exclusion_hours}h",
        f"  train/val gap      {summary.gap_hours}h",
        "",
        "Grid accounting",
        "---------------",
        f"  grid rows                {summary.grid_rows:>10,}",
        f"  dropped: excluded        {summary.dropped_excluded:>10,}   (in a repair window)",
        f"  dropped: unlabelable     {summary.dropped_unlabelable:>10,}   (horizon runs past the data)",
        f"  dropped: split gap       {summary.dropped_gap:>10,}   (buffer between partitions)",
        f"  kept                     {summary.kept_rows:>10,}",
        "",
        f"  positives                {summary.positives:>10,}",
        f"  base rate                {summary.positive_rate:>10.4%}",
        "",
        "Class balance per split",
        "-----------------------",
    ]

    split_frame = pd.DataFrame([row.model_dump() for row in summary.by_split])
    if not split_frame.empty:
        split_frame["positive_rate"] = split_frame["positive_rate"].map("{:.4%}".format)
        lines.append(split_frame.to_string(index=False))

    lines += ["", "Class balance per component per split", "-------------------------------------"]
    component_frame = pd.DataFrame(list(summary.by_component))
    if not component_frame.empty:
        pivot = component_frame.pivot(index="component", columns="split", values="positive_rate")
        pivot = pivot.reindex(columns=[name for name in SPLIT_NAMES if name in pivot.columns])
        lines.append(pivot.map("{:.4%}".format).to_string())

    low, high = EXPECTED_POSITIVE_RATE
    if summary.within_expected_range:
        lines += ["", f"Base rate {summary.positive_rate:.4%} is inside the expected {low:.1%}-{high:.1%} band."]
    else:
        lines += [
            "",
            f"WARNING: base rate {summary.positive_rate:.4%} is outside the expected {low:.1%}-{high:.1%} band.",
            "The horizon or the exclusion logic is wrong. Stop and debug before phase 2.",
        ]
    return "\n".join(lines)


def write_labels(con: duckdb.DuckDBPyConnection, settings: Settings | None = None) -> Path:
    """Write the training labels to ``curated/labels/``, partitioned by split."""
    settings = settings or get_settings()
    destination = settings.curated_dir / "labels"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    frame = load_training_labels(con)
    frame.to_parquet(destination, engine="pyarrow", index=False, partition_cols=["split"])
    LOGGER.info("Wrote %d labelled rows to %s", len(frame), destination)
    return destination


def write_results(summary: LabelSummary, path: Path) -> Path:
    """Commit the numbers, not just the printout. See spec section 11."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="write curated/labels/ as well as reporting")
    parser.add_argument(
        "--no-results",
        dest="results",
        action="store_false",
        help=f"skip writing results/{RESULTS_FILENAME}",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when the base rate falls outside the expected band",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)

    con = build(settings)
    summary = summarise(con, settings)
    print(format_report(summary))

    if args.write:
        write_labels(con, settings)
    if args.results:
        write_results(summary, Path("results") / RESULTS_FILENAME)

    if args.strict and not summary.within_expected_range:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
