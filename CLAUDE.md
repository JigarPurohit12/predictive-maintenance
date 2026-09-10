# Predictive maintenance — working rules

## Ground rules

- Python 3.11. Type hints everywhere. Pydantic v2 for data contracts.
- config.py is the single source of truth for settings. Nothing else reads os.environ.
- LOCAL_MODE=true must run the entire training pipeline with zero AWS calls,
  reading parquet from ./data/ and using a local MLflow tracking directory.
- NEVER use train_test_split with shuffle=True on this data. All splits are
  time-based. If you are about to write a random split, stop and ask me.
- Any feature computed at time t may only use data at or before t. No centered
  rolling windows, no bfill, no groupby transforms across the split boundary.
- Do not report accuracy anywhere. Primary metric is PR-AUC; report recall at a
  fixed false-alarm rate alongside it.
- Feature engineering lives in ONE module used by both training and serving.
  Do not write a second Spark version of the same logic.
- Do not add Airflow, Kubeflow, Kubernetes, or a feature store. Out of scope.
- Do not create a SageMaker real-time endpoint unless I explicitly ask.
- Every module gets a test in tests/. Mock all AWS calls. Tests run offline.
- Log with the logging module, not print().
- When you finish a phase, run the tests and show me the output before moving on.

## Orientation

Two execution paths that never mix:

- **Training (offline, occasional):** raw tables → labels → features →
  time-based split → XGBoost → MLflow registry.
- **Scoring (batch, scheduled):** new telemetry → Glue ETL → features → batch
  transform → threshold → work order.

There is no real-time endpoint and there will not be one. Failure horizons are
measured in hours; a persistent endpoint bills 24/7 to answer a question nobody
asks more than once every three. `tests/test_aws_jobs.py` greps the launchers
for `create_endpoint` and fails if one ever appears.

## The data layout

```
data/landing/   downloaded CSVs, pre-ingest. The only place a CSV may live.
data/raw/       typed parquet mirror of the CSVs   <- S3_RAW_PREFIX
data/curated/   cleaned/joined tables and labels   <- S3_CURATED_PREFIX
data/features/  feature matrix + manifest.json     <- S3_FEATURES_PREFIX
data/scores/    dt=YYYY-MM-DD/ scoring output      <- S3_SCORES_PREFIX
```

The local tree mirrors the S3 prefixes on purpose: phase 4 swaps the root, not
the shape. Nothing under `data/` is committed; everything under `results/` is.

## Things that are easy to get wrong here

**Labels and SQL**

- **Labels are defined in SQL, once.** `sql/02_failure_labels.sql` is the only
  implementation. `features/labeling.py` runs it and reports on the output.
  Locally that SQL runs in DuckDB, on AWS in Athena.
- The `.sql` files never name a file path or a literal horizon. They read
  relations called `raw_*` and a one-row `params` table, both created by
  `data_layer/duckdb_io.py` from config.py. That is what lets the same text run
  over parquet, over pandas frames in tests, and over the Glue Catalog.
- Every interval comparison in SQL is `date_diff('second', a, b)` rather than
  interval arithmetic. `date_diff` has the same signature in DuckDB and Athena;
  interval arithmetic does not.
- The horizon window is **open on the left, closed on the right**: `(ts, ts+H]`.
  A failure at exactly `ts` is not a positive. The exclusion window is the other
  way round — `[failure, failure+E]` — because the moment of failure is itself
  part of the repair.
- **Post-failure rows are dropped, never labelled 0.**
- The last `PREDICTION_HORIZON_HOURS` of data cannot be labelled at all and are
  dropped, positives included.

**Features**

- `features/build_features.py` is the only feature path. Training calls it and
  so does scoring. The Glue job cleans and aggregates and calls nothing else —
  if a column ends up in `FEATURE_ORDER`, it is not computed in Spark.
- Windows are **trailing and right-closed**: `(ts - W, ts]`, including the
  reading at `ts`. `features/windows.WINDOW_CLOSED` is the one place that lives.
  The spec's prose says "closed on the right" while its pandas hint says
  `closed='left'`, which means the opposite; the prose won. See the module
  docstring.
- Windows are **time-based** (`rolling('24h')`), never row-based. Telemetry has
  gaps and a row-based window silently reaches further back whenever one occurs.
- The feature grid is anchored to the Unix epoch, not to each machine's first
  reading. `tests/test_parity.py` asserts the pandas grid and the SQL grid pick
  the same timestamps at every cadence.
- The contract is a **function**, `build_contract(settings)`, not a module-level
  `FEATURE_ORDER`: computing it at import would read settings as a side effect
  of importing the module.
- Error codes, components and machine models are **pinned constants**, not
  `SELECT DISTINCT`. A quiet Tuesday with no `error3` must not produce a
  narrower matrix and a hash mismatch.

**Modelling**

- `GAP_HOURS >= PREDICTION_HORIZON_HOURS`, enforced in config.py. The same gap
  sits between validation and test.
- The threshold is chosen on **validation** and applied unchanged to test.
  Choosing it on test is how a cost curve becomes a number nobody should believe.
- 0.5 is meaningless once `scale_pos_weight` is in play. Derive it from the cost
  curve, or pin it in config and know why.
- The rules baseline is not decoration. A GBM that barely beats it is a finding.
- SMOTE is an ablation (`--models smote`), never part of the comparison.

**Serving**

- Every score carries `model_version` (the registry version number, not the
  metric it won on) and `feature_hash`. Both are non-negotiable.
- `serving/inference.py` refuses any batch whose column hash does not match the
  model's. A *reordered* matrix is the dangerous case: nothing else in the stack
  would notice, and every score would silently be wrong.
- Batch transform runs with `SplitType: None` so the payload keeps its header —
  which is what the hash check reads.

## Working locally

- The venv is Python 3.12; the `Dockerfile` pins 3.11 as the spec requires. 3.11
  was not available on the development machine.
- **Run everything through the venv.** A bare `pytest` picks up the global
  interpreter and dies at collection with `No module named 'duckdb'`.
- MLflow is `sqlite:///mlflow.db`, not `file:./mlruns`. MLflow 3.x refuses the
  filesystem store outright and its model registry never supported it. Still
  local, still no server.
- `pytest` stays offline. `tests/conftest.py` strips every setting from the
  environment before each test, derived from `Settings.model_fields`, so a new
  setting cannot leak in from a shell. Every AWS client is a fake object.
- `ruff check .` and `mypy config.py data_layer features training serving monitoring aws`
  before finishing.
- Labelling tests register ten-row pandas frames as the `raw_*` relations and
  run the production SQL over them. Add cases there, not to a Python copy.
- `tests/conftest.py::synthetic_dataset` is 90 days across 6 machines with three
  planted failures, tuned to land the base rate near 3.5%. It exists so the
  pipeline has something to chew on. **No metric computed from it means
  anything** — the signal is a linear vibration ramp, which is why the rules
  baseline wins on it.

## Build phases

Defined in `PDM_BUILD_SPEC.md` section 8. All five are implemented.

1. Data and labels — raw CSVs → validated parquet → labelled grid. ✅
2. Features and splits — leakage-free matrix, time-based split, parity tests. ✅
3. Modelling — five models, cost curve, SHAP, MLflow registry. ✅
4. Operationalise on AWS — Glue, SageMaker batch transform, SNS work orders. ✅
5. Monitoring — PSI drift, delayed-label evaluation. ✅

**What is *not* done, and matters more than any of the above: the real dataset
has never been through this.** `data/landing/` is empty — the Kaggle CSVs were
never downloaded. Every number in `results/` was produced from the synthetic
fixture and is worth nothing. Download the data, run phases 1–3 against it, and
only then quote a figure from this repository.
