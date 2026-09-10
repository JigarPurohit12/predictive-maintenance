"""Time-based splits, and the checks that prove they are what they claim.

There is no ``train_test_split`` here and there never will be. Shuffling
time-ordered rows means training on the future to predict the past; the PR-AUC
comes out beautiful and means nothing, and it is the first thing a good
interviewer checks.

The split *assignment* is made once, in ``sql/02_failure_labels.sql``, from the
boundaries in config.py. This module verifies that assignment rather than
recomputing it — one implementation, checked from the outside — and adds the
optional machine holdout, which SQL has no business deciding.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from config import SPLIT_NAMES, Settings, get_settings
from features.contract import ID_COLUMNS, LABEL_COLUMN, SPLIT_COLUMN

LOGGER = logging.getLogger(__name__)


class SplitViolation(ValueError):
    """A split is not what it claims to be."""


class SplitBounds(BaseModel):
    """Observed extent of one split."""

    model_config = ConfigDict(frozen=True)

    split: str
    rows: int
    positives: int
    ts_min: datetime
    ts_max: datetime
    machines: int

    @property
    def positive_rate(self) -> float:
        return self.positives / self.rows if self.rows else 0.0


def describe_splits(frame: pd.DataFrame) -> list[SplitBounds]:
    bounds: list[SplitBounds] = []
    for name in SPLIT_NAMES:
        subset = frame[frame[SPLIT_COLUMN] == name]
        if subset.empty:
            continue
        bounds.append(
            SplitBounds(
                split=name,
                rows=len(subset),
                positives=int(subset[LABEL_COLUMN].sum()),
                ts_min=subset["ts"].min(),
                ts_max=subset["ts"].max(),
                machines=int(subset["machine_id"].nunique()),
            )
        )
    return bounds


def verify_time_splits(frame: pd.DataFrame, settings: Settings | None = None) -> list[SplitBounds]:
    """Check the ranges are disjoint, ordered, and separated by the full gap.

    Raises on the first violation. Called from ``build_features`` and from
    ``train`` — it costs microseconds and it is the check that stops the most
    expensive mistake in the project.
    """
    settings = settings or get_settings()
    bounds = describe_splits(frame)
    by_name = {bound.split: bound for bound in bounds}

    if unknown := set(frame[SPLIT_COLUMN].unique()) - set(SPLIT_NAMES):
        raise SplitViolation(f"Unexpected split label(s) {sorted(unknown)}; gap rows should already be dropped.")

    gap = pd.Timedelta(hours=settings.gap_hours)
    horizon = pd.Timedelta(hours=settings.prediction_horizon_hours)
    if gap < horizon:
        raise SplitViolation(f"Configured gap {gap} is shorter than the horizon {horizon}.")

    for earlier, later in (("train", "val"), ("val", "test")):
        if earlier not in by_name or later not in by_name:
            continue
        left, right = by_name[earlier], by_name[later]
        if left.ts_max >= right.ts_min:
            raise SplitViolation(
                f"{earlier} runs to {left.ts_max} but {later} starts at {right.ts_min}; the ranges overlap."
            )
        observed = right.ts_min - left.ts_max
        if observed <= horizon:
            raise SplitViolation(
                f"Only {observed} separates {earlier} from {later}, which does not clear the "
                f"{horizon} horizon. The last {earlier} rows are labelled from {later} data."
            )
    return bounds


def machine_overlap(frame: pd.DataFrame) -> dict[tuple[str, str], int]:
    """How many machines appear in both of each pair of splits.

    Overlap is the *expected* state for a purely time-based split — the same
    fleet runs throughout — and is only a problem if you are claiming
    generalisation to unseen equipment. Measure it so the claim you make is the
    one you can defend.
    """
    machines = {
        name: set(frame.loc[frame[SPLIT_COLUMN] == name, "machine_id"].unique())
        for name in SPLIT_NAMES
        if (frame[SPLIT_COLUMN] == name).any()
    }
    overlap: dict[tuple[str, str], int] = {}
    names = list(machines)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap[(left, right)] = len(machines[left] & machines[right])
    return overlap


@dataclass(frozen=True)
class MachineHoldout:
    """A split frame plus the identity of the machines held out of training."""

    frame: pd.DataFrame
    held_out: tuple[int, ...]

    def unseen_only(self) -> pd.DataFrame:
        """Test restricted to held-out machines: unseen kit *and* unseen time.

        This is the frame to evaluate on when the claim is "it generalises to
        equipment it has never seen". The full ``frame`` keeps the seen machines
        in the test period too, which measures something different and easier —
        report whichever you claim, not whichever is higher.
        """
        held = set(self.held_out)
        keep = (self.frame[SPLIT_COLUMN] != "test") | self.frame["machine_id"].isin(held)
        return self.frame.loc[keep].reset_index(drop=True)


def holdout_machines(
    frame: pd.DataFrame,
    fraction: float = 0.2,
    *,
    seed: int = 0,
) -> MachineHoldout:
    """Additionally hold out whole machines, to test generalisation to unseen kit.

    Machines are sampled by identity, not by row, and the sampling is seeded, so
    the holdout is stable across runs. Every row belonging to a held-out machine
    moves to ``test`` wherever it came from, which is what removes that machine
    from training entirely.

    Note that ``test`` afterwards contains two different populations: held-out
    machines across all time, and seen machines in the test period. They answer
    different questions. Use :meth:`MachineHoldout.unseen_only` for the first.

    Sampling is stratified on whether a machine ever fails: with roughly 8
    failures per machine across a fleet of 100, an unstratified 20% draw can
    plausibly take every failing machine or none of them.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1); got {fraction}.")

    ever_fails = frame.groupby("machine_id")[LABEL_COLUMN].max()
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for _, group in ever_fails.groupby(ever_fails.to_numpy()):
        machines = np.sort(group.index.to_numpy())
        count = max(1, round(len(machines) * fraction)) if len(machines) else 0
        chosen.extend(int(value) for value in rng.choice(machines, size=min(count, len(machines)), replace=False))

    held = set(chosen)
    LOGGER.info("Holding out %d of %d machines.", len(held), frame["machine_id"].nunique())
    out = frame.copy()
    out.loc[out["machine_id"].isin(held), SPLIT_COLUMN] = "test"
    return MachineHoldout(frame=out, held_out=tuple(sorted(held)))


def split_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The matrix carved into train/val/test, preserving row order."""
    return {
        name: frame[frame[SPLIT_COLUMN] == name].reset_index(drop=True)
        for name in SPLIT_NAMES
        if (frame[SPLIT_COLUMN] == name).any()
    }


def xy(frame: pd.DataFrame, feature_order: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Features and labels as arrays, in contract order."""
    features = frame.loc[:, list(feature_order)].to_numpy(dtype="float64", copy=False)
    labels = frame[LABEL_COLUMN].to_numpy(dtype="int8", copy=False)
    return features, labels


def format_splits(bounds: Sequence[SplitBounds]) -> str:
    frame = pd.DataFrame(
        [
            {
                "split": bound.split,
                "rows": bound.rows,
                "positives": bound.positives,
                "positive_rate": f"{bound.positive_rate:.4%}",
                "ts_min": bound.ts_min,
                "ts_max": bound.ts_max,
                "machines": bound.machines,
            }
            for bound in bounds
        ]
    )
    return frame.to_string(index=False) if not frame.empty else "(no rows)"


__all__ = [
    "ID_COLUMNS",
    "MachineHoldout",
    "SplitBounds",
    "SplitViolation",
    "describe_splits",
    "format_splits",
    "holdout_machines",
    "machine_overlap",
    "split_frames",
    "verify_time_splits",
    "xy",
]
