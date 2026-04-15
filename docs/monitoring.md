# Drift Monitoring Guide

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

## Main Surfaces

CLI:

- `python -m amazon_recsys.cli.main ingest-outcomes --source ...`
- `python -m amazon_recsys.cli.main monitor-drift --window-start ... --window-end ...`
- `python -m amazon_recsys.cli.main monitor-backfill --days N`

API:

- `GET /monitoring/drift/summary`

UI:

- the dashboard monitoring panel shows the latest persisted drift summary for the active bundle

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

4. Prepare an outcomes file with this schema:

```text
occurred_at,user_key|user_id,item_id,event_type,rating,value
```

Minimum required columns:

- `occurred_at`
- `item_id`
- `event_type`
- either `user_key` or `user_id`

5. Ingest outcomes:

```powershell
python -m amazon_recsys.cli.main ingest-outcomes --source .\outcomes.csv
```

6. Run a monitoring window:

```powershell
python -m amazon_recsys.cli.main monitor-drift --window-start 2026-04-14T00:00:00Z --window-end 2026-04-15T00:00:00Z
```

7. Review the latest summary:

- dashboard monitoring panel
- `http://127.0.0.1:8000/monitoring/drift/summary`
- MLflow experiment `<your-experiment>-monitoring` if MLflow is enabled

## Local Monitoring Exact Commands

```powershell
pip install -e .[dev]
Copy-Item .env.example .env -Force
$env:AMAZON_RECSYS_MONITORING_ENABLED="true"
$env:AMAZON_RECSYS_MLFLOW_ENABLED="true"
mlflow ui --backend-store-uri ".\mlflow_runs" --port 5000
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
python -m amazon_recsys.cli.main serve
python -m amazon_recsys.cli.main ingest-outcomes --source .\outcomes.csv
python -m amazon_recsys.cli.main monitor-drift --window-start 2026-04-14T00:00:00Z --window-end 2026-04-15T00:00:00Z
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
