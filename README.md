# Amazon RecSys

Amazon RecSys is a package-owned multi-stage recommender built on Amazon review data. It trains hybrid retrieval plus ranking pipelines, exports versioned runtime bundles, serves the active bundle through FastAPI and Jinja, and records MLflow and monitoring artifacts around that flow.

`src/amazon_recsys/` is the source of truth. `notebooks/amazon_recsys_pipeline.py` and `notebooks/RecSys.ipynb` remain as notebook-facing compatibility layers over the same package-owned ML core.

## What This Repo Does

- trains a hybrid recommender with popularity, cooccurrence, latent CF, content-based retrieval, and an optional two-tower retriever
- ranks candidate sets with XGBoost by default, with `dlrm` kept as an experimental backend
- exports and activates versioned serving bundles for online inference
- serves recommendation, history, model, evaluation, and monitoring endpoints through FastAPI plus a Jinja UI
- logs training and monitoring runs to MLflow when enabled
- computes batch feature drift and concept drift from served inference logs and delayed outcomes

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

The training and export entry point is:

```powershell
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
```

## Documentation

Detailed guides live under [`docs/`](docs/README.md).

- [Docs Hub](docs/README.md)
- [Architecture Guide](docs/architecture.md)
- [Running The App](docs/running-the-app.md)
- [MLflow Guide](docs/mlflow.md)
- [Drift Monitoring Guide](docs/monitoring.md)

## Key Paths

- `src/amazon_recsys/ml/core.py`: recommender training, retrieval, ranking, and bundle-facing artifacts
- `src/amazon_recsys/application/services.py`: active-bundle recommendation service and serving fallback behavior
- `src/amazon_recsys/api/`: FastAPI routers for health, models, recommendations, and monitoring
- `src/amazon_recsys/monitoring/`: reference profiles, inference and outcome logging, and drift computation
- `src/amazon_recsys/cli/main.py`: train, evaluate, export, activate, serve, and monitor commands
- `notebooks/amazon_recsys_pipeline.py`: notebook compatibility import surface

## Current Role Of The Repo

Primary runtime:

- package-owned recommender logic
- bundle export and activation
- FastAPI plus Jinja serving
- MLflow integration
- batch monitoring based on served traffic and delayed outcomes

Secondary support:

- notebook compatibility for existing exploration workflows
- `template.py` for bootstrap/template generation
- `infra/azure/` and workflow files for deployment-oriented support

Use `src/amazon_recsys/` as the source of truth and treat notebooks and deployment/template assets as consumers or support layers around that package.
