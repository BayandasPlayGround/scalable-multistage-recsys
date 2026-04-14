# Amazon Reviews Recommender System

This repository now supports two valid ways of working:

- a **package-first production workflow** under `src/amazon_recsys/`
- a **notebook workflow** that consumes the package-owned ML core

If you are specifically looking for experiment tracking setup, jump to:

- [MLflow Tracking](#mlflow-tracking)
- [MLflow Quick Start](#mlflow-quick-start)

The recommender engine now lives in:

- `src/amazon_recsys/ml/core.py`

The notebook-facing entrypoint:

- `notebooks/amazon_recsys_pipeline.py`

is now a compatibility layer that re-exports the package implementation.

## Table of Contents

- [Moving from the research environment into the production environment](#moving-from-the-research-environment-into-the-production-environment)
- [The Two Working Modes](#the-two-working-modes)
- [Notebook / research mode](#notebook--research-mode)
- [Production scaffold mode](#production-scaffold-mode)
- [Current ML Architecture](#current-ml-architecture)
- [Retrieval / candidate generation](#retrieval--candidate-generation)
- [Ranking](#ranking)
- [Data framing](#data-framing)
- [Current Software Architecture](#current-software-architecture)
- [Repository Map](#repository-map)
- [`template.py`](#templatepy)
- [FastAPI + Jinja App Layer](#fastapi--jinja-app-layer)
- [API direction](#api-direction)
- [Frontend direction](#frontend-direction)
- [Azure-First Structure](#azure-first-structure)
- [Configuration Story](#configuration-story)
- [Core ML configuration](#core-ml-configuration)
- [Application/runtime configuration](#applicationruntime-configuration)
- [MLflow Tracking](#mlflow-tracking)
- [MLflow Quick Start](#mlflow-quick-start)
- [Local MLflow exact commands](#local-mlflow-exact-commands)
- [Production MLflow exact commands](#production-mlflow-exact-commands)
- [Local MLflow mode](#local-mlflow-mode)
- [Production MLflow mode](#production-mlflow-mode)
- [How To Work With The Repo](#how-to-work-with-the-repo)
- [If you are doing ML experimentation](#if-you-are-doing-ml-experimentation)
- [If you are doing application work](#if-you-are-doing-application-work)
- [If you are doing infrastructure work](#if-you-are-doing-infrastructure-work)
- [Quick Start](#quick-start)
- [Install runtime dependencies](#install-runtime-dependencies)
- [Install the package in editable mode](#install-the-package-in-editable-mode)
- [Install with dev extras](#install-with-dev-extras)
- [Run the notebook workflow](#run-the-notebook-workflow)
- [Run the scaffolded app](#run-the-scaffolded-app)
- [Run tests](#run-tests)
- [Export and activate a local bundle](#export-and-activate-a-local-bundle)
- [Beginner Command Walkthrough](#beginner-command-walkthrough)
- [Step 1: Install the project in editable dev mode](#step-1-install-the-project-in-editable-dev-mode)
- [Step 2: Create your local `.env` file](#step-2-create-your-local-env-file)
- [Step 3: Train, export, and activate a serving bundle](#step-3-train-export-and-activate-a-serving-bundle)
- [Step 4: Start the API and web app](#step-4-start-the-api-and-web-app)
- [Migration Status](#migration-status)
- [Already real](#already-real)
- [Still evolving](#still-evolving)
- [Recommended Mental Model](#recommended-mental-model)
- [Local Dev Vs Production-Like Runbook](#local-dev-vs-production-like-runbook)
- [Local dev mode](#local-dev-mode)
- [Production-like local mode](#production-like-local-mode)
- [Verification Matrix](#verification-matrix)

## Moving from the research environment into the production environment

The earlier research environment was mostly:

- `RecSys.ipynb`
- `amazon_recsys_pipeline.py`
- research files in `Research/`

The current production/development environment includes:

- a modular package under `src/amazon_recsys/`
- a FastAPI + Jinja scaffold for serving and demoing recommendations
- tests
- Docker and compose files
- Azure-first infrastructure placeholders
- `template.py` for regenerating the scaffold structure

The shortest mental model is:

- **use `src/amazon_recsys/` as the source of truth**
- **use the notebook as a client of that code**

## The Two Working Modes

### Notebook / research mode

This is now the notebook-facing experience over the package ML core.

Primary files:

- `notebooks/RecSys.ipynb`
- `notebooks/amazon_recsys_pipeline.py`
- `Research/`

Use this mode for:

- EDA
- debugging the corpus
- recommender experimentation
- retrieval and ranking evaluation
- qualitative recommendation review

### Production scaffold mode

This is the application-oriented source of truth.

Primary files and folders:

- `src/amazon_recsys/`
- `app.py`
- `template.py`
- `tests/`
- `infra/azure/`
- `Dockerfile`
- `docker-compose.yml`
- `.env.example`

Use this mode for:

- the authoritative ML implementation
- service interfaces
- API development
- bundle export and serving
- frontend prototyping
- containerization
- Azure deployment preparation

## Current ML Architecture

The recommender is still a two-stage system.

### Retrieval / candidate generation

The current direction is hybrid and recall-first:

- popularity backfill
- item-item cooccurrence / KNN
- latent collaborative filtering via sparse SVD
- content-based retrieval with TF-IDF + SVD text vectors
- optional neural two-tower retrieval
- hybrid candidate union

### Ranking

The current default precision stage is:

- **XGBoost**

The experimental path remains:

- **DLRM-style neural ranking**

### Data framing

The notebook pipeline currently uses:

- `parent_asin` as the canonical item ID
- positives from `rating >= 4`
- neutral interactions from `rating == 3`
- hard negatives from `rating <= 2`
- Amazon metadata enrichment
- leave-last-out temporal evaluation

## Current Software Architecture

The new codebase is being shaped as a **modular monolith**.

The package root is:

- `src/amazon_recsys/`

Its responsibilities are split like this:

- `config/`
  - settings
  - environment loading
  - dependency wiring
- `domain/`
  - entities
  - value objects
  - protocols and interfaces
- `application/`
  - use cases
  - orchestration services
- `infrastructure/`
  - repositories
  - artifact loading
  - retriever and ranker adapters
  - storage integrations
- `ml/`
  - feature logic
  - training and evaluation flows
  - inference bundle loading
- `api/`
  - FastAPI app
  - routers
  - request and response models
- `web/`
  - Jinja templates
  - static assets
  - UI routes
- `observability/`
  - logging and monitoring hooks
- `cli/`
  - train, evaluate, serve, and demo commands

## Repository Map

```text
Recommender Systems/
|-- README.md
|-- Requirements.txt
|-- pyproject.toml
|-- template.py
|-- app.py
|-- Dockerfile
|-- docker-compose.yml
|-- .env.example
|-- notebooks/
|   |-- RecSys.ipynb
|   `-- amazon_recsys_pipeline.py
|-- src/
|   `-- amazon_recsys/
|-- tests/
|-- infra/
|   `-- azure/
|-- Research/
|-- artifacts/
|-- amazon_review_data/
```

How to interpret that:

- `src/amazon_recsys/` is the production codebase and current source of truth
- `notebooks/amazon_recsys_pipeline.py` is the notebook compatibility import layer
- `artifacts/` stores caches, trained artifacts, bundles, and evaluation outputs
- `amazon_review_data/` stores local source data and metadata

## `template.py`

`template.py` is the scaffold generator for the upgraded structure.

It is designed to:

- create the agreed folder layout
- create missing placeholder files
- skip non-empty files
- log what it created versus what already existed

That means you can rerun it safely while the project evolves.

## FastAPI + Jinja App Layer

The scaffold now includes a lightweight serving and demo layer.

### API direction

The app shape includes routes such as:

- `/health`
- `/ready`
- `/config`
- `/users`
- `/recommend`
- `/users/{user_id}/history`
- `/models/active`
- `/evaluate/summary`

### Frontend direction

The Jinja-based UI is intended to support:

- user lookup
- searchable trained-user selection
- prior-order inspection
- recommendation display
- candidate provenance display
- active model and bundle visibility

The goal is a useful demo and testing surface, not a polished product UI yet.

## Azure-First Structure

The repository now includes Azure-oriented deployment scaffolding under:

- `infra/azure/bicep/`
- `infra/azure/aml/`
- `infra/azure/aks/`

Intended responsibilities:

- `bicep/`
  - infrastructure-as-code
- `aml/`
  - Azure ML environments and jobs
- `aks/`
  - serving deployment manifests

This is still scaffold-level for deployment, but the actual training and inference logic now lives in `src/amazon_recsys/ml/core.py`.

## Configuration Story

There are now effectively two configuration layers.

### Core ML configuration

The recommender workflow is controlled mainly by `PipelineConfig` inside `src/amazon_recsys/ml/core.py`.

That covers:

- ingestion and sampling
- filtering
- training caps
- retriever and ranker choices
- evaluation controls
- artifact locations

### Application/runtime configuration

The application runtime is controlled by typed settings under `src/amazon_recsys/config/`.

The intended split is:

- `DataConfig`
- `TrainingConfig`
- `RetrievalConfig`
- `RankingConfig`
- `ServingConfig`
- `MLflowConfig`
- `AzureConfig`

Environment-driven values are documented in:

- `.env.example`

## MLflow Tracking

MLflow is now integrated into the package-first training workflow.

If you only want the shortest usable commands, go straight to:

- [MLflow Quick Start](#mlflow-quick-start)

When MLflow is enabled, the training pipeline logs:

- the resolved runtime configuration
- dataset and split counts
- evaluation metrics from the offline metric CSV files
- the generated evaluation artifacts
- the exported bundle files and bundle lineage metadata

This means one MLflow run can now tell you:

- which settings produced a bundle
- which evaluation metrics were recorded
- which bundle version was exported from that run

The relevant environment variables are:

- `AMAZON_RECSYS_MLFLOW_ENABLED`
- `AMAZON_RECSYS_MLFLOW_TRACKING_URI`
- `AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME`
- `AMAZON_RECSYS_MLFLOW_BACKEND_ROOT`
- `AMAZON_RECSYS_MLFLOW_RUN_NAME_PREFIX`

### MLflow Quick Start

This is the shortest version.

If you want MLflow locally with a file-backed store in this repo:

1. Install the package:

```powershell
pip install -e .[dev]
```

2. Create your local environment file:

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

7. Open:

- `http://127.0.0.1:5000/` for MLflow
- `http://127.0.0.1:8000/` for the recommender app

What this gives you:

- the recommender still serves the exported active bundle
- MLflow stores the training run, metrics, and bundle lineage
- the default local MLflow store lives in `mlflow_runs/`

### Local MLflow exact commands

```powershell
pip install -e .[dev]
Copy-Item .env.example .env -Force
mlflow ui --backend-store-uri ".\mlflow_runs" --port 5000
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
python -m amazon_recsys.cli.main serve
```

### Production MLflow exact commands

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

### Local MLflow mode

Use local MLflow mode when you want the training history stored directly inside the repo workspace.

Recommended `.env` values:

- `AMAZON_RECSYS_MLFLOW_ENABLED=true`
- `AMAZON_RECSYS_MLFLOW_TRACKING_URI=`
- `AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME=amazon-recsys-local`
- `AMAZON_RECSYS_MLFLOW_BACKEND_ROOT=mlflow_runs`

Important detail:

- if `AMAZON_RECSYS_MLFLOW_TRACKING_URI` is left blank, the app uses the local file-backed store under `mlflow_runs`

Typical local MLflow workflow:

```powershell
pip install -e .[dev]
Copy-Item .env.example .env -Force
mlflow ui --backend-store-uri ".\mlflow_runs" --port 5000
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
python -m amazon_recsys.cli.main serve
```

Open:

- `http://127.0.0.1:5000/` for MLflow
- `http://127.0.0.1:8000/` for the app

What to expect:

- the `export-bundle` command trains the recommender
- the training run appears in MLflow
- the evaluation CSV artifacts appear in the run
- the final bundle files are attached to that same run under the bundle artifact path

### Production MLflow mode

Use production MLflow mode when you want the same training and bundle lineage sent to a remote tracking service.

Recommended environment values:

- `AMAZON_RECSYS_MLFLOW_ENABLED=true`
- `AMAZON_RECSYS_MLFLOW_TRACKING_URI=http://your-mlflow-server:5000`
- `AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME=amazon-recsys-prod`
- `AMAZON_RECSYS_MLFLOW_RUN_NAME_PREFIX=prod`

Typical production-style workflow:

```powershell
$env:AMAZON_RECSYS_ENVIRONMENT="production"
$env:AMAZON_RECSYS_USE_MOCK_BUNDLE_IF_MISSING="false"
$env:AMAZON_RECSYS_MLFLOW_ENABLED="true"
$env:AMAZON_RECSYS_MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"
$env:AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME="amazon-recsys-prod"
python -m amazon_recsys.cli.main export-bundle --run-name prod-local --run-profile quality --activate
python -m amazon_recsys.cli.main serve
```

In production mode:

- the API still serves only the active exported bundle
- MLflow tracks training and bundle lineage
- the web app does not need MLflow to be reachable at serving time unless you choose to build a direct MLflow UI integration later

## How To Work With The Repo

### If you are doing ML experimentation

Start with:

- `notebooks/RecSys.ipynb`
- `src/amazon_recsys/ml/core.py`

That is the correct place for:

- model debugging
- offline metrics
- candidate diagnostics
- ranker comparisons
- demo recommendations

### If you are doing application work

Start with:

- `src/amazon_recsys/`
- `app.py`
- `tests/`

That is the correct place for:

- changing the production implementation directly
- building service interfaces
- serving recommendations
- adding UI and API behavior

### If you are doing infrastructure work

Start with:

- `infra/azure/`
- `Dockerfile`
- `docker-compose.yml`
- `.github/workflows/`

That is the correct place for:

- deployment scaffolding
- containerization
- CI/CD
- Azure setup

## Quick Start

### Install runtime dependencies

```bash
pip install -r Requirements.txt
```

### Install the package in editable mode

```bash
pip install -e .
```

### Install with dev extras

```bash
pip install -e .[dev]
```

### Run the notebook workflow

Open:

- `notebooks/RecSys.ipynb`

### Run the scaffolded app

```bash
uvicorn app:app --reload
```

### Run tests

```bash
pytest
```

### Export and activate a local bundle

```bash
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
```

## Beginner Command Walkthrough

If you want the shortest real workflow, these are the main commands:

```powershell
pip install -e .[dev]
Copy-Item .env.example .env -Force
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
python -m amazon_recsys.cli.main serve
```

This is what each one means in plain English.

### Step 1: Install the project in editable dev mode

```powershell
pip install -e .[dev]
```

What this command means:

- `pip install` tells Python to install the project.
- `.` means "install the project in the current folder".
- `-e` means editable mode.
- editable mode means when you change the code in this repo, Python uses those live changes without you reinstalling every time.
- `[dev]` means "also install the development extras", such as test and tooling dependencies.

Why you need it:

- without this, the `amazon_recsys` package may not be importable from the command line
- the CLI command `python -m amazon_recsys.cli.main ...` depends on the package being installed
- the tests also depend on the dev extras

What to expect:

- Python downloads or resolves dependencies
- the package becomes available in your shell
- after this, the CLI commands in this README should work

### Step 2: Create your local `.env` file

```powershell
Copy-Item .env.example .env -Force
```

What this command means:

- it copies the template environment file into a real local `.env` file
- `.env.example` is the safe template kept in source control
- `.env` is your local runtime configuration file
- `-Force` means overwrite an existing `.env` file with the template copy

Why you need it:

- the app loads configuration from `.env`
- this is where host, port, data paths, artifact paths, and training defaults come from
- it gives you a working starting configuration without hand-writing settings

Important default values from `.env.example`:

- `AMAZON_RECSYS_ENVIRONMENT=local`
- `AMAZON_RECSYS_RUN_PROFILE=debug`
- `AMAZON_RECSYS_USE_MOCK_BUNDLE_IF_MISSING=true`
- `AMAZON_RECSYS_RANKER_BACKEND=xgboost`
- `AMAZON_RECSYS_MLFLOW_ENABLED=false`
- `AMAZON_RECSYS_MLFLOW_BACKEND_ROOT=mlflow_runs`

What to expect:

- a new `.env` file appears in the repo root
- the app and CLI will read from that file on the next command

### Step 3: Train, export, and activate a serving bundle

```powershell
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
```

Yes: training happens here.

This command does not only export a bundle. It first runs the training pipeline, then saves the result as a serving bundle, then optionally activates it.

What each part means:

- `python -m amazon_recsys.cli.main` runs the package CLI module
- `export-bundle` tells the CLI to create a versioned serving bundle
- `--run-name debug-local` gives this training run a readable name inside `artifacts/`
- `--run-profile debug` uses the smaller debug configuration rather than a larger quality or full run
- `--activate` marks the new bundle as the active one for online serving

What actually happens under the hood:

- the CLI loads settings from `.env`
- it builds the dependency container
- it runs the training pipeline
- the training pipeline prepares the corpus
- it builds train, validation, and test splits
- it trains the retrievers
- it trains the ranker
- it saves a bundle under `artifacts/amazon_recsys/bundles/`
- it writes or updates `artifacts/production/active_bundle.json`
- if MLflow is enabled, it logs the run configuration, evaluation artifacts, metrics, and bundle lineage to the configured tracking store

Why this is the most important command:

- it is the command that turns your raw data and configuration into something the API can actually serve
- if you skip this and you do not allow mock mode, the app will have nothing real to load

What files and folders it depends on:

- your dataset should exist in either `amazon_review_data/` or `notebooks/amazon_review_data/`
- your runtime settings come from `.env`

What to expect:

- the command may take a while depending on data size
- you should see JSON output describing the created bundle
- after it finishes, there should be bundle files under `artifacts/amazon_recsys/bundles/`
- the app is now able to serve a real active bundle

If you want only training without bundle export:

```powershell
python -m amazon_recsys.cli.main train --run-name debug-local --run-profile debug
```

That trains the model artifacts, but it does not package and activate a serving bundle.

### Step 4: Start the API and web app

```powershell
python -m amazon_recsys.cli.main serve
```

What this command means:

- it starts the FastAPI application
- it loads the app settings
- it builds the service container
- it serves the API and Jinja web pages on the configured host and port

What it tries to load:

- first choice: the active real bundle from `artifacts/production/active_bundle.json`
- fallback in local mode: a mock bundle, if `AMAZON_RECSYS_USE_MOCK_BUNDLE_IF_MISSING=true`

Why bundle activation matters:

- if you ran `export-bundle ... --activate`, the app should load the real trained bundle
- if you did not activate a bundle and mock mode is disabled, readiness should fail

What to expect:

- the server starts on the host and port from `.env`
- by default that is `http://127.0.0.1:8000/` or `http://localhost:8000/`
- the API docs are at `/docs`
- the health endpoint is at `/health`
- the readiness endpoint is at `/ready`

Typical first-run sequence:

```powershell
pip install -e .[dev]
Copy-Item .env.example .env -Force
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
python -m amazon_recsys.cli.main serve
```

Open these pages after the server starts:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/ready`

## Migration Status

### Already real

- package-owned ML core in `src/amazon_recsys/ml/core.py`
- notebook compatibility import layer
- hybrid retrieval experimentation
- XGBoost-first ranking direction
- artifact generation, evaluation outputs, and bundle activation
- FastAPI + Jinja scaffold
- Docker and compose files
- Azure folder structure
- local-dev and production-like tests

### Still evolving

- further splitting `ml/core.py` into smaller package modules over time
- hardening bundle export beyond the default classical + XGBoost path
- wiring real Azure ML training jobs
- validating Azure deployment against a live subscription

## Recommended Mental Model

If the upgrade feels large, keep this short map in mind:

- `src/amazon_recsys/ml/core.py` is the recommender engine
- `notebooks/amazon_recsys_pipeline.py` is the notebook compatibility layer
- `src/amazon_recsys/application/services.py` serves active bundles
- `template.py` recreates the scaffold
- `infra/azure/` prepares deployment
- `tests/` protect both local-dev and production-like behavior

That is the simplest way to make sense of the upgrade.

## Local Dev Vs Production-Like Runbook

### Local dev mode

Use this when you want quick feedback and safe startup behavior.

Recommended settings:

- `AMAZON_RECSYS_ENVIRONMENT=local`
- `AMAZON_RECSYS_USE_MOCK_BUNDLE_IF_MISSING=true`
- `AMAZON_RECSYS_RUN_PROFILE=debug`
- `AMAZON_RECSYS_ENABLE_NEURAL_RETRIEVER=false`

Typical commands:

```bash
pip install -e .[dev]
uvicorn app:app --reload
pytest -m "foundation or config or serving"
```

If you want a real local bundle:

```bash
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
python -m amazon_recsys.cli.main serve
```

If you also want local MLflow tracking:

```bash
mlflow ui --backend-store-uri "./mlflow_runs" --port 5000
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
```

### Production-like local mode

Use this when you want readiness and serving semantics closer to production.

Recommended settings:

- `AMAZON_RECSYS_ENVIRONMENT=production`
- `AMAZON_RECSYS_DEBUG=false`
- `AMAZON_RECSYS_RELOAD=false`
- `AMAZON_RECSYS_USE_MOCK_BUNDLE_IF_MISSING=false`

Typical commands:

```bash
python -m amazon_recsys.cli.main export-bundle --run-name prod-local --run-profile debug --activate
python -m amazon_recsys.cli.main serve
```

If you also want production-style remote MLflow tracking:

```bash
export AMAZON_RECSYS_MLFLOW_ENABLED=true
export AMAZON_RECSYS_MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
python -m amazon_recsys.cli.main export-bundle --run-name prod-local --run-profile quality --activate
```

Expected behavior:

- `/ready` returns `503` before a real bundle is activated
- `/ready` returns `200` after activation
- `/models/active` reports the active bundle metadata

## Verification Matrix

Run the full suite:

```bash
pytest
```

Fast local-dev checks:

```bash
pytest -m "foundation or config or serving"
```

Real-bundle training and serving checks:

```bash
pytest -m "data or retrieval or ranking or serving"
```

Environment mode checks:

```bash
pytest tests/test_runtime_modes.py
```

Notebook compatibility check:

```bash
pytest tests/test_notebook_compatibility.py
```
