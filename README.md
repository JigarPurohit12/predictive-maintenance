# Predictive Maintenance & Equipment Failure Detection

Batch scoring system answering one question per machine per run: **will this
asset fail within the next 24 hours?**

Sensor telemetry, error logs, maintenance records and failure events are joined
in SQL, turned into rolling-window features, and used to train a gradient-boosted
classifier on a severely imbalanced target. Scores above a cost-tuned threshold
become maintenance work orders.

All five build phases are implemented. **No numbers here are real yet** — see
[Status](#status).

## Dataset

[Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
— 100 machines, hourly readings across a year, four sensors, plus separate
tables for error codes, maintenance records, machine metadata and failure
events. Roughly 761 failure records against ~876,000 telemetry rows.

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
```

Then the whole pipeline, in order:

```bash
python -m data_layer.ingest                 # CSVs -> validated parquet
python -m features.labeling --write         # labels + class balance report
python -m features.build_features           # leakage-free feature matrix
python -m training.train                    # 5 models, cost curves, MLflow
python -m training.register --best          # promote the winner
python -m serving.batch_score --local       # score the fleet, raise work orders
python -m monitoring.drift --build-reference
python -m monitoring.performance            # delayed-label evaluation
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
config.py                       pydantic-settings, single source of truth
sql/01_staging.sql              typed staging views over the raw tables
sql/02_failure_labels.sql       the ONLY implementation of the labelling rules
data_layer/schemas.py           TelemetryRow, FailureEvent, MaintRecord, ...
data_layer/ingest.py            landing CSVs -> typed, partitioned parquet
data_layer/validate.py          duplicates, gaps, ranges, referential integrity
data_layer/duckdb_io.py         attaches the raw relations, runs the SQL
features/contract.py            the ordered feature list and its hash
features/windows.py             trailing rolling stats, lags, trends, error counts
features/static.py              machine age, model, maintenance clocks
features/build_features.py      the ONE feature path — training and serving
features/labeling.py            runs the label SQL, reports the class balance
training/splits.py              time-based splits, gap checks, machine holdout
training/imbalance.py           scale_pos_weight, cost curve, recall@budget
training/train.py               5 models, threshold discipline, MLflow
training/evaluate.py            PR-AUC, cost-curve plot, SHAP
training/register.py            promote the winner to the registry
serving/inference.py            SageMaker handlers + the feature-hash assertion
serving/batch_score.py          the scoring driver
serving/alerts.py               scores -> work orders -> SNS
aws/glue_jobs/etl_telemetry.py  PySpark: raw -> curated (no feature logic)
aws/sagemaker/*.py              training and transform job launchers
aws/infra_notes.md              every resource created, and its teardown command
monitoring/drift.py             PSI per feature against the training distribution
monitoring/performance.py       delayed-label evaluation, running metrics table
results/                        committed metric JSONs — keep these
```

## Design decisions worth knowing

- **One feature module, two execution paths.** `features/build_features.py` is
  called by training and by scoring. `tests/test_parity.py` runs both over the
  same input and asserts the values are identical. A second Spark implementation
  would diverge silently, and nothing would tell you.
- **Labels are defined once, in SQL.** DuckDB locally, Athena on AWS. The tests
  register ten-row pandas frames as the source relations and assert against the
  production SQL rather than a Python copy of it.
- **The feature order is hashed.** XGBoost fed a numpy array reads column 7 as
  column 7. A reordered matrix raises nothing on its own, so the hash travels
  with the model and `serving/inference.py` refuses any batch that disagrees.
- **The threshold comes from a cost curve**, chosen on validation and applied
  unchanged to test. 0.5 is meaningless once `scale_pos_weight` is in play.
- **No real-time endpoint.** Batch transform starts, scores, stops. A test greps
  the launchers for `create_endpoint` and fails if one appears.

## Metrics

Nothing here reports accuracy. On a ~2%-positive target, predicting "never
fails" scores 98% while being useless. Primary metric is PR-AUC, reported
alongside its lift over the base rate and recall at a fixed alarm budget —
"catches N% of failures at one false alarm per machine per month".

Two separate numbers are worth keeping, and they measure different things:

1. **Model quality** (phase 3) — rules baseline PR-AUC vs tuned XGBoost PR-AUC.
   This is what feature engineering, imbalance handling and thresholding bought.
2. **Operational delta** (phase 4) — retrain-to-production time, scoring latency
   for the fleet, reproducibility. SageMaker, Glue and MLflow do not improve
   model quality; they are deployment and tracking infrastructure, and claiming
   otherwise is the fastest way to lose an interview.

Have an answer ready for "what was your base rate?".

## Status

Code complete across all five phases: 303 tests, `ruff` and `mypy` clean.

**The real dataset has never been through this pipeline.** `data/landing/` is
empty; the Kaggle CSVs were never downloaded. Everything currently in `results/`
was produced from `tests/conftest.py::synthetic_dataset` — 90 days across 6
machines with three planted failures — which exists so the code has something to
run on. Its signal is a linear vibration ramp, which is why the rules baseline
beats the trees on it, and **no metric from it means anything**.

Download the data, run phases 1–3 against it, confirm the base rate lands near
2%, and only then quote a figure from this repository.
