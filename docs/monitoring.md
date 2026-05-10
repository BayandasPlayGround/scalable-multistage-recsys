# Drift Monitoring Guide

[Back to docs hub](README.md) | [Back to main README](../README.md)

## What The Monitoring Layer Does

The monitoring subsystem lives under `src/amazon_recsys/monitoring/`.

It:

- creates a reference profile at bundle export time
- logs recommendation inference events after successful serving
- ingests delayed user outcomes
- computes scheduled batch data drift and concept drift
- persists the latest monitoring summary outside the runtime bundle
- logs monitoring runs to a separate MLflow experiment named `<experiment>-monitoring`

This is batch monitoring, not real-time alerting.

## The Short Mental Model

Monitoring only makes sense once you have all three of these:

1. a trained and active bundle
2. recommendation traffic that the app has already served
3. an outcomes file that tells the system what happened after those recommendations

If one of those is missing, monitoring will either do nothing useful or fail.

The most common mistake is:

- running `ingest-outcomes --source .\outcomes.csv` before `outcomes.csv` actually exists

For local development you do not need to hand-create that file anymore. Use the automated synthetic-outcomes path instead.

## The Three Inputs Monitoring Needs

### 1. Active Bundle

You need a real trained bundle first:

```powershell
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
```

### 2. Served Recommendation Traffic

You then need to start the app and actually generate recommendation requests:

```powershell
python -m amazon_recsys.cli.main serve
```

Then use the UI or API to request recommendations for a few users. That creates the inference log records that monitoring compares against the reference profile.

### 3. Outcomes File

Finally, you need a file that says what happened after the recommendations were served.

You can use either:

- `user_id`
  - easiest when you are creating the file yourself
- `user_key`
  - already hashed if you want to avoid raw IDs in the input file

If you provide `user_id`, the monitoring pipeline hashes it for storage automatically.

There is a sample file here:

- [`docs/examples/outcomes.example.csv`](examples/outcomes.example.csv)

## Main Surfaces

CLI:

- `python -m amazon_recsys.cli.main ingest-outcomes --source ...`
- `python -m amazon_recsys.cli.main monitor-drift --window-start ... --window-end ...`
- `python -m amazon_recsys.cli.main monitor-backfill --days N`
- `python -m amazon_recsys.cli.main diagnose-candidates --bundle-version active --split test --sample-size 500 --persist`

API:

- `GET /monitoring/drift/summary`
- `GET /monitoring/candidate-recall/summary`
- `GET /monitoring/candidate-recall/history`

UI:

- the dashboard monitoring panel shows the latest persisted drift summary for the active bundle
- the candidate recovery panel shows persisted active-bundle recall by category, history length, source, and cold-start scenario

## What Is Measured

### Data Drift

Current v1 feature set includes:

- request history length
- known-user rate
- unseen-user rate
- unseen-history-item rate
- served category mix
- served price
- served average rating
- score distribution
- candidate-source mix
- popularity-bucket mix

Metrics:

- PSI for numeric or binned numeric features
- Jensen-Shannon distance for categorical distributions

### Concept Drift

Concept drift is computed from delayed outcomes joined back to logged inference records.

Primary positive signals:

- `purchase`
- or `rating >= 4`

Current metrics:

- `hit_rate@K`
- `ndcg@K`
- `mrr@K`
- `purchase_rate@K`
- `cold_start_hit_rate@K`

### Candidate Recovery Diagnostics

Candidate recovery diagnostics run before ranker tuning. They measure whether the held-out target reaches the candidate set at two stages:

- `candidate_union`
- `ranker_candidates`

Persisted slices include target category, history-length bucket, candidate source, and cold-start user type.

## Relevant Environment Variables

- `AMAZON_RECSYS_MONITORING_ENABLED`
- `AMAZON_RECSYS_MONITORING_ROOT`
- `AMAZON_RECSYS_MONITORING_WINDOW_DAYS`
- `AMAZON_RECSYS_MONITORING_LABEL_DELAY_DAYS`
- `AMAZON_RECSYS_MONITORING_ATTRIBUTION_HORIZON_DAYS`
- `AMAZON_RECSYS_MONITORING_MIN_EVENTS_PER_WINDOW`
- `AMAZON_RECSYS_MONITORING_PSI_WARN`
- `AMAZON_RECSYS_MONITORING_PSI_ALERT`
- `AMAZON_RECSYS_MONITORING_JS_WARN`
- `AMAZON_RECSYS_MONITORING_JS_ALERT`
- `AMAZON_RECSYS_MONITORING_PERFORMANCE_DROP_WARN`
- `AMAZON_RECSYS_MONITORING_PERFORMANCE_DROP_ALERT`

## Monitoring Quick Start

This is the recommended local demo flow because it does not require creating a CSV manually.

1. Enable monitoring in `.env`:

```text
AMAZON_RECSYS_MONITORING_ENABLED=true
AMAZON_RECSYS_MONITORING_ROOT=artifacts/amazon_recsys/monitoring
AMAZON_RECSYS_MONITORING_WINDOW_DAYS=1
AMAZON_RECSYS_MONITORING_LABEL_DELAY_DAYS=2
AMAZON_RECSYS_MONITORING_ATTRIBUTION_HORIZON_DAYS=7
AMAZON_RECSYS_MONITORING_MIN_EVENTS_PER_WINDOW=500
```

2. Train and activate a real bundle:

```powershell
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
```

3. Start the app and generate recommendation traffic:

```powershell
python -m amazon_recsys.cli.main serve
```

4. In the UI or API, request recommendations for a few known users.

5. Generate synthetic delayed outcomes automatically from the logged inference events:

python -m amazon_recsys.cli.main simulate-outcomes --days 1
```

6. Run the monitoring job:

```powershell
python -m amazon_recsys.cli.main monitor-backfill --days 1 --simulate-outcomes
```

7. Review the latest summary:

- dashboard monitoring panel
- `http://127.0.0.1:8000/monitoring/drift/summary`
- MLflow experiment `<your-experiment>-monitoring` if MLflow is enabled

## Automated Local Demo Path

If you just want monitoring to work end to end on your local machine, do this exact order:

1. Train and activate a bundle:

```powershell
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
```

2. Start the app:

```powershell
python -m amazon_recsys.cli.main serve
```

3. Generate a few recommendation requests in the browser or API.

4. In a new shell, generate synthetic purchase outcomes automatically:

```powershell
python -m amazon_recsys.cli.main simulate-outcomes --days 1
```

5. Run the monitoring job:

```powershell
python -m amazon_recsys.cli.main monitor-backfill --days 1 --simulate-outcomes
```

If you skip step 3, the monitoring result will be empty because there are no logged recommendation events to replay.

## Real Outcome Ingestion Path

Use this when you have real delayed feedback from another system and want to ingest it explicitly.

Prepare an outcomes file with this schema:

```text
occurred_at,user_key|user_id,item_id,event_type,rating,value
```

Minimum required columns:

- `occurred_at`
- `item_id`
- `event_type`
- either `user_key` or `user_id`

If you want a starter file, copy the sample:

```powershell
Copy-Item .\docs\examples\outcomes.example.csv .\outcomes.csv
```

Then ingest it:

```powershell
python -m amazon_recsys.cli.main ingest-outcomes --source .\outcomes.csv
```

## Local Monitoring Exact Commands

```powershell
pip install -e .[dev]
Copy-Item .env.example .env -Force
$env:AMAZON_RECSYS_MONITORING_ENABLED="true"
$env:AMAZON_RECSYS_MLFLOW_ENABLED="true"
mlflow ui --backend-store-uri ".\mlflow_runs" --port 5000
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
python -m amazon_recsys.cli.main serve
python -m amazon_recsys.cli.main simulate-outcomes --days 1
python -m amazon_recsys.cli.main monitor-backfill --days 1 --simulate-outcomes
```

## Production Monitoring Exact Commands

```powershell
$env:AMAZON_RECSYS_ENVIRONMENT="production"
$env:AMAZON_RECSYS_USE_MOCK_BUNDLE_IF_MISSING="false"
$env:AMAZON_RECSYS_MONITORING_ENABLED="true"
$env:AMAZON_RECSYS_MLFLOW_ENABLED="true"
$env:AMAZON_RECSYS_MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"
$env:AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME="amazon-recsys-prod"
python -m amazon_recsys.cli.main export-bundle --run-name prod-local --run-profile quality --activate
python -m amazon_recsys.cli.main ingest-outcomes --source .\prod-outcomes.csv
python -m amazon_recsys.cli.main monitor-drift --window-start 2026-04-14T00:00:00Z --window-end 2026-04-15T00:00:00Z
python -m amazon_recsys.cli.main serve
```

## Operational Notes

- training still happens during `export-bundle`
- monitoring does not retrain the model
- monitoring reads the active bundle reference profile, logged inference events, and delayed outcomes
- drift summaries are stored outside the runtime bundle
- if MLflow is enabled, monitoring runs are logged to a separate experiment ending in `-monitoring`
- `simulate-outcomes` is for local testing and demos, not production truth data

If you customize:

- `AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME=enaex-recsys-dev`

then monitoring runs appear under:

- `enaex-recsys-dev-monitoring`
