-- The feature grid, the target, the exclusion windows and the split periods.
--
-- This file is the only implementation of the labelling rules. features/
-- labeling.py runs it and reports on the output; it does not reimplement it.
-- Locally that is DuckDB, on AWS it is Athena over the same curated tables.
--
-- Reads the one-row `params` table written by data_layer/duckdb_io.py from
-- config.py, so the horizon and the cadence are never hard-coded here:
--   cadence_hours, horizon_hours, exclusion_hours,
--   train_end_ts, val_start_ts, val_end_ts, test_start_ts
--
-- Portability note: date_diff(unit, start, end) has the same signature and
-- semantics in DuckDB and in Athena (Trino), which is why every interval
-- comparison below is written as a difference in seconds rather than by adding
-- an interval to a timestamp. Interval arithmetic is where the two dialects
-- part company.

-- Failure moments, collapsed across components. A machine-moment where three
-- components fail together is one event for labelling and one repair window.
CREATE OR REPLACE VIEW failure_moments AS
SELECT DISTINCT machine_id, ts
FROM stg_failures;

CREATE OR REPLACE VIEW components AS
SELECT DISTINCT component
FROM stg_failures;

CREATE OR REPLACE VIEW data_bounds AS
SELECT min(ts) AS ts_min, max(ts) AS ts_max
FROM stg_telemetry;

-- The scoring grid: one row per machine per cadence tick.
--
-- Anchored to the Unix epoch rather than to each machine's first reading, so
-- the grid is identical for every machine, reproducible across runs, and the
-- same set of timestamps the scoring path will produce. Training on all 876k
-- hourly rows would mostly be training on near-duplicates.
CREATE OR REPLACE VIEW feature_grid AS
SELECT t.machine_id, t.ts
FROM stg_telemetry t
CROSS JOIN params p
WHERE minute(t.ts) = 0
  AND second(t.ts) = 0
  AND date_diff('hour', TIMESTAMP '1970-01-01 00:00:00', t.ts) % p.cadence_hours = 0;

-- For every grid row: the next failure strictly after it, and the most recent
-- failure at or before it. One aggregate pass rather than two correlated
-- subqueries, because Athena charges for the second one.
CREATE OR REPLACE VIEW grid_failure_context AS
SELECT
    g.machine_id,
    g.ts,
    min(CASE WHEN f.ts >  g.ts THEN f.ts END) AS next_failure_ts,
    max(CASE WHEN f.ts <= g.ts THEN f.ts END) AS last_failure_ts
FROM feature_grid g
LEFT JOIN failure_moments f
       ON f.machine_id = g.machine_id
GROUP BY g.machine_id, g.ts;

-- The labelled grid, before any rows are dropped. Every flag is exposed rather
-- than applied so the phase 1 report can count what each rule removes.
CREATE OR REPLACE VIEW labels AS
SELECT
    c.machine_id,
    c.ts,
    -- Positive when a failure falls in (ts, ts + horizon]. Open on the left:
    -- a failure happening exactly at ts is not something a model scoring at ts
    -- could have warned about.
    CASE
        WHEN c.next_failure_ts IS NOT NULL
         AND date_diff('second', c.ts, c.next_failure_ts) <= p.horizon_hours * 3600
        THEN 1 ELSE 0
    END AS label,
    c.next_failure_ts,
    c.last_failure_ts,
    -- Excluded when ts falls in [failure, failure + exclusion]. Closed on the
    -- left: the moment of failure is itself part of the repair. These rows are
    -- dropped, never labelled 0 — a freshly repaired machine looks nothing like
    -- a healthy one, and keeping them teaches the model to detect repairs.
    CASE
        WHEN c.last_failure_ts IS NOT NULL
         AND date_diff('second', c.last_failure_ts, c.ts) <= p.exclusion_hours * 3600
        THEN TRUE ELSE FALSE
    END AS excluded,
    -- A row can only be trusted as a negative once its whole horizon has been
    -- observed. The last `horizon` hours of the dataset cannot be labelled at
    -- all and are dropped, positives included.
    CASE
        WHEN date_diff('second', c.ts, b.ts_max) >= p.horizon_hours * 3600
        THEN TRUE ELSE FALSE
    END AS labelable,
    -- Half-open periods. 'gap' rows sit in the buffer between partitions and
    -- belong to no split: without the buffer the last training rows carry
    -- labels that depend on data inside the validation window.
    CASE
        WHEN c.ts <  p.train_end_ts                              THEN 'train'
        WHEN c.ts >= p.val_start_ts  AND c.ts < p.val_end_ts      THEN 'val'
        WHEN c.ts >= p.test_start_ts                             THEN 'test'
        ELSE 'gap'
    END AS split
FROM grid_failure_context c
CROSS JOIN params p
CROSS JOIN data_bounds b;

-- What actually reaches phase 2.
CREATE OR REPLACE VIEW labels_training AS
SELECT machine_id, ts, label, split
FROM labels
WHERE excluded = FALSE
  AND labelable = TRUE
  AND split <> 'gap';

-- Same rows, one per component, for the class-balance report. Which component
-- is about to fail is not the modelled target — the target is any failure — but
-- a per-component rate of 0 means the exclusion or horizon logic has eaten a
-- component, and that is worth seeing before phase 2.
CREATE OR REPLACE VIEW labels_by_component AS
SELECT
    l.machine_id,
    l.ts,
    l.split,
    c.component,
    max(CASE
            WHEN f.ts IS NOT NULL
             AND f.ts > l.ts
             AND date_diff('second', l.ts, f.ts) <= p.horizon_hours * 3600
            THEN 1 ELSE 0
        END) AS label
FROM labels l
CROSS JOIN components c
CROSS JOIN params p
LEFT JOIN stg_failures f
       ON f.machine_id = l.machine_id
      AND f.component  = c.component
WHERE l.excluded = FALSE
  AND l.labelable = TRUE
  AND l.split <> 'gap'
GROUP BY l.machine_id, l.ts, l.split, c.component;
