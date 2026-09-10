-- Typed staging views over the raw tables.
--
-- Inputs are the relations raw_telemetry / raw_errors / raw_maint /
-- raw_failures / raw_machines. Locally those are DuckDB views over parquet
-- (see data_layer/duckdb_io.py); on AWS they are Glue Catalog tables. Nothing
-- in this file names a file path, so the same text runs in both places.
--
-- Two jobs only: pin the types, and drop duplicate keys. data_layer/validate.py
-- already fails the ingest on duplicates, so the row_number filters here are for
-- the case ingest cannot see — the same S3 object delivered twice into a
-- partition that Athena is reading.

CREATE OR REPLACE VIEW stg_machines AS
SELECT machine_id, model, age
FROM (
    SELECT
        CAST(machine_id AS INTEGER) AS machine_id,
        CAST(model AS VARCHAR)      AS model,
        CAST(age AS INTEGER)        AS age,
        row_number() OVER (PARTITION BY machine_id ORDER BY machine_id) AS rn
    FROM raw_machines
) m
WHERE rn = 1;

CREATE OR REPLACE VIEW stg_telemetry AS
SELECT machine_id, ts, volt, rotate, pressure, vibration
FROM (
    SELECT
        CAST(machine_id AS INTEGER) AS machine_id,
        CAST(ts AS TIMESTAMP)       AS ts,
        CAST(volt AS DOUBLE)        AS volt,
        CAST(rotate AS DOUBLE)      AS rotate,
        CAST(pressure AS DOUBLE)    AS pressure,
        CAST(vibration AS DOUBLE)   AS vibration,
        row_number() OVER (PARTITION BY machine_id, ts ORDER BY ts) AS rn
    FROM raw_telemetry
) t
WHERE rn = 1;

-- An error is not a failure. Most machines log errors continuously and never
-- fail; these are predictive input, never the target.
CREATE OR REPLACE VIEW stg_errors AS
SELECT DISTINCT
    CAST(machine_id AS INTEGER) AS machine_id,
    CAST(ts AS TIMESTAMP)       AS ts,
    CAST(error_id AS VARCHAR)   AS error_id
FROM raw_errors;

-- Component replacements, both scheduled and post-failure. A maintenance record
-- on its own does not mean anything broke.
CREATE OR REPLACE VIEW stg_maint AS
SELECT DISTINCT
    CAST(machine_id AS INTEGER) AS machine_id,
    CAST(ts AS TIMESTAMP)       AS ts,
    CAST(component AS VARCHAR)  AS component
FROM raw_maint;

-- The target events. One row per component, so a single machine-moment can
-- carry several.
CREATE OR REPLACE VIEW stg_failures AS
SELECT DISTINCT
    CAST(machine_id AS INTEGER) AS machine_id,
    CAST(ts AS TIMESTAMP)       AS ts,
    CAST(component AS VARCHAR)  AS component
FROM raw_failures;
