"""Machine-level features: age, model, and time since each component was last
serviced.

"Static" means fixed per machine (age, model) or changing only at discrete
maintenance events. The maintenance clocks are the interesting ones: a component
that was replaced three days ago is in a very different state from one running
on eleven months of wear, and that is exactly what the sensors are noisy about.

Everything here is computed with :func:`pandas.merge_asof` in ``backward``
direction, which by construction can only see the most recent event at or before
each timestamp. There is no version of this that accidentally reaches forward.
"""

from __future__ import annotations

import logging

import pandas as pd

from features.contract import COMPONENTS, MACHINE_MODELS

LOGGER = logging.getLogger(__name__)

KEY_COLUMNS = ["machine_id", "ts"]

#: Hours in a year, used only to keep the maintenance clocks on a sane scale in
#: log messages. Nothing is normalised by it.
HOURS_PER_YEAR = 8_760


def machine_features(
    grid: pd.DataFrame,
    machines: pd.DataFrame,
    *,
    machine_models: tuple[str, ...] = MACHINE_MODELS,
) -> pd.DataFrame:
    """Age and a one-hot model indicator, broadcast onto the grid.

    Models are one-hot rather than ordinal: ``model3`` is not "more" than
    ``model1``, and a tree given an ordinal encoding will happily split on an
    ordering that does not exist.
    """
    out = grid[KEY_COLUMNS].merge(machines[["machine_id", "model", "age"]], on="machine_id", how="left")

    unknown = sorted(set(out["model"].dropna().unique()) - set(machine_models))
    if unknown:
        LOGGER.warning("Dropping %d machine model(s) outside the contract: %s", len(unknown), unknown)

    out["machine_age"] = out["age"].astype("float64")
    for model in machine_models:
        out[f"model_{model}"] = (out["model"] == model).astype("float64")

    if out["machine_age"].isna().any():
        missing = int(out["machine_age"].isna().sum())
        raise ValueError(f"{missing} grid row(s) reference a machine absent from the machines table.")

    return out.drop(columns=["model", "age"])


def maintenance_clocks(
    grid: pd.DataFrame,
    maint: pd.DataFrame,
    *,
    components: tuple[str, ...] = COMPONENTS,
) -> pd.DataFrame:
    """Hours since each component was last replaced, per machine per timestamp.

    Where a component has no maintenance record before a timestamp the clock is
    right-censored at the machine's first observation rather than left null: all
    we know is "at least this long", which is both true and leakage-free.
    Filling it forward from a later record would not be.
    """
    out = grid[KEY_COLUMNS].sort_values("ts", kind="stable").reset_index(drop=True)
    first_seen = out.groupby("machine_id", sort=False)["ts"].transform("min")

    clock_columns: list[str] = []
    for component in components:
        column = f"hours_since_maint_{component}"
        clock_columns.append(column)

        events = maint[maint["component"] == component][KEY_COLUMNS] if not maint.empty else maint[KEY_COLUMNS]
        if events.empty:
            out[column] = (out["ts"] - first_seen).dt.total_seconds() / 3_600.0
            continue

        events = events.sort_values("ts", kind="stable").rename(columns={"ts": "last_maint_ts"})
        merged = pd.merge_asof(
            out[KEY_COLUMNS],
            events,
            left_on="ts",
            right_on="last_maint_ts",
            by="machine_id",
            direction="backward",
            allow_exact_matches=True,
        )
        elapsed = (merged["ts"] - merged["last_maint_ts"]).dt.total_seconds() / 3_600.0
        censored = (out["ts"] - first_seen).dt.total_seconds() / 3_600.0
        out[column] = elapsed.fillna(censored)

    # The most recently serviced component. A machine where everything is old is
    # a different animal from one where three of four parts are new.
    out["hours_since_maint_any"] = out[clock_columns].min(axis=1)
    return out.sort_values(KEY_COLUMNS, kind="stable", ignore_index=True)


def build_static_features(
    grid: pd.DataFrame,
    machines: pd.DataFrame,
    maint: pd.DataFrame,
    *,
    components: tuple[str, ...] = COMPONENTS,
    machine_models: tuple[str, ...] = MACHINE_MODELS,
) -> pd.DataFrame:
    """Machine metadata and maintenance clocks, joined on ``(machine_id, ts)``."""
    metadata = machine_features(grid, machines, machine_models=machine_models)
    clocks = maintenance_clocks(grid, maint, components=components)
    return metadata.merge(clocks, on=KEY_COLUMNS, how="left", validate="one_to_one").sort_values(
        KEY_COLUMNS, kind="stable", ignore_index=True
    )
