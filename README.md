# Predictive Maintenance & Equipment Failure Detection

Batch scoring system answering one question per machine per run: **will this
asset fail within the next 24 hours?**

Sensor telemetry, error logs, maintenance records and failure events are joined
in SQL, turned into rolling-window features, and used to train a gradient-boosted
classifier on a severely imbalanced target. Scores above a cost-tuned threshold
become maintenance work orders.

Status: **phase 1 of 5 complete** (data and labels). See `CLAUDE.md`.

## Dataset

[Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
— 100 machines, hourly readings across a year, four sensors, plus separate
tables for error codes, maintenance records, machine metadata and failure
events. Roughly 761 failure records against ~876,000 telemetry rows.

Five tables, so the joins are real ones.

The data is **not** committed. Get it before running anything:

```bash
python -m data_layer.ingest --download
```

That needs the Kaggle CLI (`pip install kaggle`) and an API token at
`~/.kaggle/kaggle.json`. Downloading the five CSVs by hand into `data/landing/`
works identically — they must keep their original names (`PdM_telemetry.csv`,
`PdM_errors.csv`, `PdM_maint.csv`, `PdM_failures.csv`, `PdM_machines.csv`).

## Quick start

Everything below must run inside the project venv. A bare `pytest` picks up the
global interpreter, which has none of these dependencies and fails at collection
with `ModuleNotFoundError: No module named 'duckdb'`.

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements-dev.txt
cp .env.example .env                                # optional; defaults match

python -m data_layer.ingest                         # CSVs -> validated parquet
python -m features.labeling --write                 # labels + class balance report
pytest
```

## Problem definition

The four numbers everything else hangs off, all set in `config.py`:

| Setting | Value | Why |
|---|---|---|
| `PREDICTION_HORIZON_HOURS` | 24 | Long enough to dispatch a technician, short enough that the signal is still in the sensors. |
| `FEATURE_CADENCE_HOURS` | 3 | How often the fleet is scored. Also the training grid: scoring every hourly row would mostly train on near-duplicates. |
| `POST_FAILURE_EXCLUSION_HOURS` | 24 | Rows from a failure until 24h later are dropped, not labelled 0. A freshly repaired machine looks nothing like a healthy one. |
| `GAP_HOURS` | 24 | Buffer between splits, >= the horizon. Without it the last training rows carry labels that depend on validation-window data. |

A row at `ts` is positive when a failure falls in `(ts, ts + 24h]` — open on the
left, because a model scoring at the instant of failure has nothing left to warn
about.

## Layout

```
config.py                  pydantic-settings, single source of truth
sql/01_staging.sql         typed staging views over the raw tables
sql/02_failure_labels.sql  the ONLY implementation of the labelling rules
data_layer/schemas.py      TelemetryRow, FailureEvent, MaintRecord, ...
data_layer/ingest.py       landing CSVs -> typed, partitioned parquet
data_layer/validate.py     duplicates, gaps, ranges, referential integrity
data_layer/duckdb_io.py    attaches the raw relations, runs the SQL
features/labeling.py       runs the label SQL, reports the class balance
results/                   committed metric JSONs — keep these
```

Phases 2-5 add `features/`, `training/`, `serving/`, `aws/` and `monitoring/`.

## Metrics

Nothing here reports accuracy. On a ~2%-positive target, predicting "never
fails" scores 98% while being useless. Primary metric is PR-AUC, reported
alongside recall at a fixed false-alarm rate.

No numbers yet — phase 3 produces them, and `results/` holds the JSON.
