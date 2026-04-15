# Amazon RecSys

Amazon RecSys is an Azure-first, end-to-end recommender systems project that trains a multi-stage recommendation pipeline on Amazon review data, packages the trained artifacts into versioned serving bundles, tracks experiments with MLflow, and serves recommendations through a FastAPI web application with an analytics-focused UI.

The current source of truth is the package under `src/amazon_recsys/`. The notebook-facing file `notebooks/amazon_recsys_pipeline.py` is a compatibility layer over the package-owned ML core.

## Table of Contents

- [What This Repo Does](#what-this-repo-does)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Repository At A Glance](#repository-at-a-glance)
- [Current Status](#current-status)

## What This Repo Does

- Trains a hybrid recommender with classical retrieval plus XGBoost ranking.
- Exports versioned serving bundles for online inference.
- Serves recommendations through FastAPI + Jinja.
- Tracks training and monitoring runs in MLflow.
- Computes scheduled batch data drift and concept drift from logged inference events and delayed outcomes.
- Keeps Azure deployment as the primary target without hard-coding business logic to Azure SDKs.

## Quick Start

```powershell
pip install -e .[dev]
Copy-Item .env.example .env -Force
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
python -m amazon_recsys.cli.main serve
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

The command that actually trains the model is:

```powershell
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
```

If you want the full local/prod runbook, MLflow setup, or monitoring workflow, use the docs below.

## Documentation

Detailed docs now live under [`docs/`](docs/README.md).

- [Docs Hub](docs/README.md)
- [Architecture Guide](docs/architecture.md)
- [Running The App](docs/running-the-app.md)
- [MLflow Guide](docs/mlflow.md)
- [Drift Monitoring Guide](docs/monitoring.md)

## Repository At A Glance

```text
.
|-- README.md
|-- docs/
|   |-- README.md
|   |-- architecture.md
|   |-- running-the-app.md
|   |-- mlflow.md
|   `-- monitoring.md
|-- src/
|   `-- amazon_recsys/
|       |-- api/
|       |-- application/
|       |-- cli/
|       |-- config/
|       |-- domain/
|       |-- infrastructure/
|       |-- ml/
|       |-- monitoring/
|       |-- observability/
|       `-- web/
|-- tests/
|   |-- conftest.py
|   |-- test_api_app.py
|   |-- test_container.py
|   |-- test_mlflow_integration.py
|   |-- test_monitoring_integration.py
|   |-- test_monitoring_metrics.py
|   |-- test_notebook_compatibility.py
|   |-- test_pipeline_integration.py
|   |-- test_runtime_modes.py
|   `-- test_template.py
|-- notebooks/
|   |-- README.md
|   `-- amazon_recsys_pipeline.py
|-- infra/
|   `-- azure/
|       |-- aks/
|       |-- aml/
|       `-- bicep/
|-- .github/
|   `-- workflows/
|-- app.py
|-- template.py
|-- Dockerfile
|-- docker-compose.yml
|-- pyproject.toml
|-- Requirements.txt
`-- .env.example
```

Key paths:

- `docs/`: detailed project guides split out from the main README
- `src/amazon_recsys/ml/core.py`: current recommender engine
- `src/amazon_recsys/application/services.py`: bundle-backed serving layer
- `src/amazon_recsys/monitoring/`: drift monitoring subsystem
- `src/amazon_recsys/cli/main.py`: train/evaluate/export/serve/monitor CLI
- `notebooks/amazon_recsys_pipeline.py`: notebook compatibility import surface

## Current Status

Already in place:

- package-owned ML core
- versioned serving bundles
- FastAPI + Jinja app
- MLflow training tracking
- scheduled batch drift monitoring
- local and production-like tests
- Azure deployment scaffolding

Still evolving:

- deeper decomposition of `ml/core.py`
- broader bundle export support beyond the current default path
- live Azure validation and scheduled cloud monitoring jobs

Use `src/amazon_recsys/` as the source of truth and treat the notebooks as consumers of that package.
