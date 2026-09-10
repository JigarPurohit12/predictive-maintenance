"""Column-level checks over the raw tables, run at ingest and callable on demand.

Every check returns zero or more :class:`ValidationIssue`. Nothing raises until
you ask it to, so a run can report all of the problems at once rather than
stopping at the first. Severity decides what blocks:

``ERROR``
    The data is wrong in a way that corrupts labels or features — duplicate
    keys, impossible sensor values, events for machines that do not exist.
``WARN``
    The data is unusual and worth knowing about, but modelling can proceed —
    telemetry gaps, machines that never fail.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from data_layer.schemas import SENSOR_RANGES, TELEMETRY_FREQ_HOURS

LOGGER = logging.getLogger(__name__)

Severity = Literal["ERROR", "WARN"]

#: How many offending values to carry in an issue for debugging. The point is to
#: make the problem reproducible, not to dump the frame into a log line.
SAMPLE_SIZE = 5


class ValidationIssue(BaseModel):
    """One failed check against one table."""

    model_config = ConfigDict(frozen=True)

    table: str
    check: str
    severity: Severity
    count: int
    detail: str
    sample: tuple[str, ...] = ()

    def __str__(self) -> str:
        sample = f" e.g. {', '.join(self.sample)}" if self.sample else ""
        return f"[{self.severity}] {self.table}.{self.check}: {self.detail} (n={self.count}){sample}"


class ValidationError(ValueError):
    """Raised by :meth:`ValidationReport.raise_for_errors`."""


class ValidationReport(BaseModel):
    """The result of validating one or more tables."""

    model_config = ConfigDict(frozen=True)

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "ERROR")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "WARN")

    @property
    def ok(self) -> bool:
        """True when nothing blocking was found. Warnings do not count."""
        return not self.errors

    def extend(self, issues: Iterable[ValidationIssue]) -> ValidationReport:
        return ValidationReport(issues=(*self.issues, *issues))

    def log(self, logger: logging.Logger = LOGGER) -> None:
        for issue in self.issues:
            level = logging.ERROR if issue.severity == "ERROR" else logging.WARNING
            logger.log(level, "%s", issue)
        if not self.issues:
            logger.info("Validation clean: no issues found.")

    def raise_for_errors(self) -> None:
        if self.errors:
            joined = "\n  ".join(str(issue) for issue in self.errors)
            raise ValidationError(f"{len(self.errors)} blocking validation issue(s):\n  {joined}")


def _sample(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(str(value) for value in list(values)[:SAMPLE_SIZE])


# --- Individual checks -----------------------------------------------------


def check_required_columns(frame: pd.DataFrame, table: str, columns: Sequence[str]) -> list[ValidationIssue]:
    missing = [column for column in columns if column not in frame.columns]
    if not missing:
        return []
    return [
        ValidationIssue(
            table=table,
            check="required_columns",
            severity="ERROR",
            count=len(missing),
            detail=f"missing column(s) {missing}; got {list(frame.columns)}",
        )
    ]


def check_duplicates(frame: pd.DataFrame, table: str, keys: Sequence[str]) -> list[ValidationIssue]:
    """Duplicate keys silently multiply rows in every join downstream."""
    if any(key not in frame.columns for key in keys):
        return []
    duplicated = frame.duplicated(subset=list(keys), keep=False)
    if not duplicated.any():
        return []
    offenders = frame.loc[duplicated, list(keys)].drop_duplicates()
    return [
        ValidationIssue(
            table=table,
            check="duplicates",
            severity="ERROR",
            count=int(duplicated.sum()),
            detail=f"duplicate rows on {list(keys)}",
            sample=_sample(offenders.itertuples(index=False, name=None)),
        )
    ]


def check_nulls(frame: pd.DataFrame, table: str, columns: Sequence[str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for column in columns:
        if column not in frame.columns:
            continue
        null_count = int(frame[column].isna().sum())
        if null_count:
            issues.append(
                ValidationIssue(
                    table=table,
                    check="nulls",
                    severity="ERROR",
                    count=null_count,
                    detail=f"column {column!r} contains nulls",
                )
            )
    return issues


def check_ranges(
    frame: pd.DataFrame,
    table: str,
    ranges: dict[str, tuple[float, float]] = SENSOR_RANGES,
) -> list[ValidationIssue]:
    """Sensor values outside physical plausibility."""
    issues: list[ValidationIssue] = []
    for column, (low, high) in ranges.items():
        if column not in frame.columns:
            continue
        values = frame[column]
        outside = values.notna() & ((values < low) | (values > high))
        if outside.any():
            issues.append(
                ValidationIssue(
                    table=table,
                    check="range",
                    severity="ERROR",
                    count=int(outside.sum()),
                    detail=f"{column} outside [{low}, {high}]",
                    sample=_sample(values[outside]),
                )
            )
    return issues


def check_timestamp_gaps(
    frame: pd.DataFrame,
    table: str,
    *,
    freq_hours: int = TELEMETRY_FREQ_HOURS,
    group_column: str = "machine_id",
    ts_column: str = "ts",
) -> list[ValidationIssue]:
    """Missing readings inside a machine's series.

    A gap is not fatal — the rolling windows in phase 2 are time-based and
    tolerate them — but a gap longer than a window means the features over it
    are computed from fewer observations than the name suggests, so it is worth
    seeing before you trust a metric.
    """
    if group_column not in frame.columns or ts_column not in frame.columns:
        return []
    expected = pd.Timedelta(hours=freq_hours)
    deltas = frame.sort_values([group_column, ts_column]).groupby(group_column, sort=False)[ts_column].diff()
    gaps = deltas > expected
    if not gaps.any():
        return []
    largest = deltas[gaps].max()
    return [
        ValidationIssue(
            table=table,
            check="timestamp_gaps",
            severity="WARN",
            count=int(gaps.sum()),
            detail=f"gaps longer than {freq_hours}h within a {group_column}; largest {largest}",
            sample=_sample(deltas[gaps].sort_values(ascending=False)),
        )
    ]


def check_known_machines(
    frame: pd.DataFrame,
    table: str,
    known_machine_ids: Iterable[int],
) -> list[ValidationIssue]:
    """Events referencing a machine that is not in the machines table."""
    if "machine_id" not in frame.columns:
        return []
    known = set(known_machine_ids)
    unknown = sorted(set(frame["machine_id"].unique()) - known)
    if not unknown:
        return []
    return [
        ValidationIssue(
            table=table,
            check="known_machines",
            severity="ERROR",
            count=len(unknown),
            detail="machine_id values absent from the machines table",
            sample=_sample(unknown),
        )
    ]


def check_machines_without_failures(
    failures: pd.DataFrame,
    known_machine_ids: Iterable[int],
) -> list[ValidationIssue]:
    """Machines that never fail contribute negatives only.

    That is normal and they stay in the data. It matters because a split that
    holds out whole machines can land every failing machine on one side, and
    because the per-machine positive rate is not uniform.
    """
    known = set(known_machine_ids)
    failing = set(failures["machine_id"].unique()) if "machine_id" in failures.columns else set()
    without = sorted(known - failing)
    if not without:
        return []
    return [
        ValidationIssue(
            table="failures",
            check="machines_without_failures",
            severity="WARN",
            count=len(without),
            detail=f"{len(without)} of {len(known)} machines have no failure records",
            sample=_sample(without),
        )
    ]


def check_monotonic_within_group(
    frame: pd.DataFrame,
    table: str,
    *,
    group_column: str = "machine_id",
    ts_column: str = "ts",
) -> list[ValidationIssue]:
    """Timestamps must be non-decreasing once sorted by machine — a cheap guard
    against a mis-parsed date format flipping day and month."""
    if group_column not in frame.columns or ts_column not in frame.columns:
        return []
    ordered = frame.sort_values([group_column, ts_column])
    if ordered[ts_column].isna().any():
        return [
            ValidationIssue(
                table=table,
                check="timestamp_parse",
                severity="ERROR",
                count=int(ordered[ts_column].isna().sum()),
                detail=f"{ts_column} failed to parse (NaT)",
            )
        ]
    return []


# --- Table-level entry points ----------------------------------------------

TELEMETRY_COLUMNS = ("machine_id", "ts", *SENSOR_RANGES.keys())


def validate_telemetry(frame: pd.DataFrame, known_machine_ids: Iterable[int] | None = None) -> ValidationReport:
    issues = [
        *check_required_columns(frame, "telemetry", TELEMETRY_COLUMNS),
        *check_nulls(frame, "telemetry", TELEMETRY_COLUMNS),
        *check_monotonic_within_group(frame, "telemetry"),
        *check_duplicates(frame, "telemetry", ("machine_id", "ts")),
        *check_ranges(frame, "telemetry"),
        *check_timestamp_gaps(frame, "telemetry"),
    ]
    if known_machine_ids is not None:
        issues.extend(check_known_machines(frame, "telemetry", known_machine_ids))
    return ValidationReport(issues=tuple(issues))


def validate_events(
    frame: pd.DataFrame,
    table: str,
    keys: Sequence[str],
    known_machine_ids: Iterable[int] | None = None,
) -> ValidationReport:
    """Errors, maintenance and failures share a shape: machine, timestamp, code."""
    issues = [
        *check_required_columns(frame, table, keys),
        *check_nulls(frame, table, keys),
        *check_monotonic_within_group(frame, table),
        *check_duplicates(frame, table, keys),
    ]
    if known_machine_ids is not None:
        issues.extend(check_known_machines(frame, table, known_machine_ids))
    return ValidationReport(issues=tuple(issues))


def validate_machines(frame: pd.DataFrame) -> ValidationReport:
    columns = ("machine_id", "model", "age")
    issues = [
        *check_required_columns(frame, "machines", columns),
        *check_nulls(frame, "machines", columns),
        *check_duplicates(frame, "machines", ("machine_id",)),
    ]
    return ValidationReport(issues=tuple(issues))


def validate_all(tables: dict[str, pd.DataFrame]) -> ValidationReport:
    """Validate the five raw tables together, including cross-table checks.

    ``tables`` is keyed by the canonical table names produced by
    :mod:`data_layer.ingest`: telemetry, errors, maint, failures, machines.
    """
    machines = tables.get("machines")
    known_ids = machines["machine_id"].tolist() if machines is not None else None

    report = ValidationReport()
    if machines is not None:
        report = report.extend(validate_machines(machines).issues)
    if (telemetry := tables.get("telemetry")) is not None:
        report = report.extend(validate_telemetry(telemetry, known_ids).issues)
    for table, keys in (
        ("errors", ("machine_id", "ts", "error_id")),
        ("maint", ("machine_id", "ts", "component")),
        ("failures", ("machine_id", "ts", "component")),
    ):
        if (frame := tables.get(table)) is not None:
            report = report.extend(validate_events(frame, table, keys, known_ids).issues)
    if (failures := tables.get("failures")) is not None and known_ids is not None:
        report = report.extend(check_machines_without_failures(failures, known_ids))
    return report
