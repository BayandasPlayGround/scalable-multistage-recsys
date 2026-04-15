# MLflow Guide

## What Is Logged

When MLflow is enabled, the package-first training workflow logs:

- resolved runtime configuration
- dataset and split counts
- evaluation metrics from offline metric CSV files
- evaluation artifacts
- exported bundle files and bundle lineage metadata

One MLflow run can therefore tell you:

- which settings produced a bundle
- which evaluation metrics were recorded
- which bundle version was exported from that run

## Relevant Environment Variables

- `AMAZON_RECSYS_MLFLOW_ENABLED`
- `AMAZON_RECSYS_MLFLOW_TRACKING_URI`
- `AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME`
- `AMAZON_RECSYS_MLFLOW_BACKEND_ROOT`
- `AMAZON_RECSYS_MLFLOW_RUN_NAME_PREFIX`

## How To Customize The Experiment Name

The experiment name is controlled by:

- `AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME`

You can set it in two main ways.

### Option 1: Set It In `.env`

This is the normal persistent approach.

Example:

```text
AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME=enaex-recsys-dev
```

Then run your normal training command:

```powershell
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
```

That run will be logged under the MLflow experiment:

- `enaex-recsys-dev`

### Option 2: Override It For One Shell Session

This is useful when you want a temporary name without editing `.env`.

PowerShell example:

```powershell
$env:AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME="enaex-recsys-qa"
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
```

That only affects commands run in the current shell session.

### Important Monitoring Detail

The monitoring subsystem uses the same base experiment name and appends `-monitoring`.

Example:

- if `AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME=enaex-recsys-dev`
- training runs go to `enaex-recsys-dev`
- monitoring runs go to `enaex-recsys-dev-monitoring`

### Related But Different Setting: Run Names

Do not confuse the experiment name with the run name prefix.

- `AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME`
  - controls which experiment bucket the run appears under
- `AMAZON_RECSYS_MLFLOW_RUN_NAME_PREFIX`
  - changes the run name inside that experiment

Example:

```powershell
$env:AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME="enaex-recsys-prod"
$env:AMAZON_RECSYS_MLFLOW_RUN_NAME_PREFIX="prod"
python -m amazon_recsys.cli.main export-bundle --run-name prod-local --run-profile quality --activate
```

That gives you:

- experiment: `enaex-recsys-prod`
- run name: something like `prod-train-prod-local-quality`

## MLflow Quick Start

If you want MLflow locally with a file-backed store in this repo:

1. Install the package:

```powershell
pip install -e .[dev]
```

2. Create your local `.env`:

```powershell
Copy-Item .env.example .env -Force
```

3. Enable MLflow in `.env`:

```text
AMAZON_RECSYS_MLFLOW_ENABLED=true
AMAZON_RECSYS_MLFLOW_TRACKING_URI=
AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME=amazon-recsys-local
AMAZON_RECSYS_MLFLOW_BACKEND_ROOT=mlflow_runs
```

4. Start the MLflow UI:

```powershell
mlflow ui --backend-store-uri ".\mlflow_runs" --port 5000
```

5. Train and export a bundle:

```powershell
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
```

6. Start the app:

```powershell
python -m amazon_recsys.cli.main serve
```

Open:

- `http://127.0.0.1:5000/` for MLflow
- `http://127.0.0.1:8000/` for the recommender app

## Local MLflow Exact Commands

```powershell
pip install -e .[dev]
Copy-Item .env.example .env -Force
mlflow ui --backend-store-uri ".\mlflow_runs" --port 5000
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
python -m amazon_recsys.cli.main serve
```

## Production MLflow Exact Commands

```powershell
$env:AMAZON_RECSYS_ENVIRONMENT="production"
$env:AMAZON_RECSYS_USE_MOCK_BUNDLE_IF_MISSING="false"
$env:AMAZON_RECSYS_MLFLOW_ENABLED="true"
$env:AMAZON_RECSYS_MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"
$env:AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME="amazon-recsys-prod"
$env:AMAZON_RECSYS_MLFLOW_RUN_NAME_PREFIX="prod"
python -m amazon_recsys.cli.main export-bundle --run-name prod-local --run-profile quality --activate
python -m amazon_recsys.cli.main serve
```

## Local MLflow Mode

Use local MLflow mode when you want training history stored directly inside the repo workspace.

Recommended `.env` values:

- `AMAZON_RECSYS_MLFLOW_ENABLED=true`
- `AMAZON_RECSYS_MLFLOW_TRACKING_URI=`
- `AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME=amazon-recsys-local`
- `AMAZON_RECSYS_MLFLOW_BACKEND_ROOT=mlflow_runs`

If `AMAZON_RECSYS_MLFLOW_TRACKING_URI` is blank, the app uses the local file-backed store under `mlflow_runs`.

## Production MLflow Mode

Use production MLflow mode when you want the same training and bundle lineage sent to a remote tracking service.

Recommended values:

- `AMAZON_RECSYS_MLFLOW_ENABLED=true`
- `AMAZON_RECSYS_MLFLOW_TRACKING_URI=http://your-mlflow-server:5000`
- `AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME=amazon-recsys-prod`
- `AMAZON_RECSYS_MLFLOW_RUN_NAME_PREFIX=prod`

In production mode:

- the API still serves only the active exported bundle
- MLflow tracks training and bundle lineage
- the web app does not need MLflow to be reachable at serving time

## Where To Look In MLflow

Open the experiment you configured, usually:

- `amazon-recsys-local`
- or `amazon-recsys-prod`

Inside a run, check:

- `Parameters`
  - resolved config values
- `Metrics`
  - dataset counts and offline evaluation metrics
- `Artifacts`
  - `training/`
  - `evaluation/`
  - `bundle/`
- `Tags`
  - phase, backend, retriever variants, bundle version

Important note:

- standard training runs show up under experiment runs
- the `Traces` tab stays empty unless you explicitly instrument tracing
