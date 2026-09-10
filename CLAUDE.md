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
asks more than once an hour.

## The data layout

```
data/landing/   downloaded CSVs, pre-ingest. The only place a CSV may live.
data/raw/       typed parquet mirror of the CSVs   <- S3_RAW_PREFIX
data/curated/   cleaned/joined tables and labels   <- S3_CURATED_PREFIX
data/features/  feature matrices                   <- S3_FEATURES_PREFIX
data/scores/    batch transform output             <- S3_SCORES_PREFIX
```

The local tree mirrors the S3 prefixes on purpose: phase 4 swaps the root, not
the shape. Nothing under `data/` is committed.

## Things that are easy to get wrong here

- **Labels are defined in SQL, once.** `sql/02_failure_labels.sql` is the only
  implementation. `features/labeling.py` runs it and reports on the output — it
  does not reimplement it, and neither should anything else. Locally that SQL
  runs in DuckDB, on AWS in Athena.
- The `.sql` files never name a file path or a literal horizon. They read
  relations called `raw_*` and a one-row `params` table, both created by
  `data_layer/duckdb_io.py` from config.py. That is what lets the same text run
  over parquet, over pandas frames in tests, and over the Glue Catalog.
- Every interval comparison in SQL is written as `date_diff('second', a, b)`
  rather than by adding an interval to a timestamp. `date_diff` has the same
  signature in DuckDB and Athena; interval arithmetic does not.
- The feature grid is anchored to the Unix epoch, not to each machine's first
  reading, so every machine lands on the same ticks and the training grid
  matches what the scoring path will produce.
- The horizon window is **open on the left, closed on the right**: `(ts, ts+H]`.
  A failure at exactly `ts` is not a positive — there is nothing left to warn
  about. The exclusion window is the other way round: `[failure, failure+E]`,
  because the moment of failure is itself part of the repair.
- **Post-failure rows are dropped, never labelled 0.** A machine that has just
  had a component replaced looks nothing like a healthy one.
- The last `PREDICTION_HORIZON_HOURS` of data cannot be labelled at all and are
  dropped, positives included: a row is only a trustworthy negative once its
  whole horizon has been observed.
- `GAP_HOURS` must be >= `PREDICTION_HORIZON_HOURS` and config.py enforces it.
  The same gap sits between validation and test, for the same reason.
- Ingest partitions telemetry by date, not by `(machine_id, date)`. The spec
  asks for the latter; 36,600 directories of 24 rows each is the small-files
  problem, and the scoring path prunes on date anyway. See the docstring in
  `data_layer/ingest.py` and flip `partition_columns` if you disagree.

## Working locally

- The venv is Python 3.12; the spec asks for 3.11 and the container will be
  3.11-slim. 3.11 was not available on the development machine.
- Nothing in phases 1-3 touches AWS. There is no `.env` to fill in yet;
  `.env.example` is committed and the defaults in config.py match it.
- `pytest` must stay offline. `tests/conftest.py` strips every setting from the
  environment before each test — the list is derived from `Settings.model_fields`,
  so a new setting cannot silently leak in from a developer's shell.
- `ruff check .` and `mypy config.py data_layer features` before finishing.
- Labelling tests register ten-row pandas frames as the `raw_*` relations and
  run the production SQL over them. Add cases there, not to a Python copy.

## Build phases

Defined in `PDM_BUILD_SPEC.md` section 8.

1. Data and labels — raw CSVs → validated parquet → labelled grid. ✅
2. Features and splits — leakage-free feature matrix, time-based split. ⬜
3. Modelling — baselines, XGBoost, cost curve, MLflow. ⬜
4. Operationalise on AWS — Glue, SageMaker batch transform, SNS. ⬜
5. Monitoring — PSI drift, delayed-label evaluation. ⬜

**Phase 1 has not been run against the real dataset yet.** The CSVs are not in
`data/landing/` — see the README. Until they are, the base rate in
`results/phase1_label_balance.json` does not exist and no number from this repo
means anything.
