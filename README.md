# Amazon RecSys

Amazon RecSys is a package-owned multi-stage recommender built on Amazon review data. It trains hybrid retrieval plus ranking pipelines, exports versioned runtime bundles, serves the active bundle through FastAPI and Jinja, and records MLflow and monitoring artifacts around that flow.

`src/amazon_recsys/` is the source of truth. `notebooks/amazon_recsys_pipeline.py` and `notebooks/RecSys.ipynb` remain as notebook-facing compatibility layers over the same package-owned ML core.

## What This Repo Does

- trains a hybrid recommender with popularity, cooccurrence, latent CF, content-based retrieval, and an optional two-tower retriever
- ranks candidate sets with XGBoost by default, with `dlrm` kept as an experimental backend
- exports and activates versioned ONNX-backed serving bundles for online inference
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

## Training Configuration

Training is configured from `.env`, temporary `AMAZON_RECSYS_*` environment variables, and a few CLI flags. The CLI flags are intentionally small:

- `--run-name`: names the artifact folder under `artifacts/amazon_recsys/<run-name>/`
- `--run-profile`: chooses `debug`, `quality`, or `full`
- `--force-rebuild`: rebuilds cached corpus artifacts when you need a clean preprocessing run
- `--version`: sets the exported bundle version for `export-bundle`
- `--activate`: makes the exported bundle the serving bundle

The most important data-volume controls are:

- `AMAZON_RECSYS_RUN_PROFILE`: broad preset. `debug` reads up to 100k review rows per category and keeps small training caps; `quality` uses the full available category files with moderate caps; `full` uses the full files, enables the neural retriever, and removes the retriever training cap.
- `AMAZON_RECSYS_CATEGORIES`: JSON list of categories to train on, for example `["All_Beauty"]` for a smaller single-category run or `["All_Beauty","Automotive","Industrial_and_Scientific"]` for the default three-category run.
- `AMAZON_RECSYS_DEV_MODE` and `AMAZON_RECSYS_DEV_FRACTION`: deterministic sampling before preprocessing. This is useful for quick experiments against a fraction of each configured category.
- `AMAZON_RECSYS_K_CORE`: minimum positive interactions required per user and item. Lower values keep more sparse data; higher values produce a denser but smaller training graph.
- `AMAZON_RECSYS_RANKER_TRAIN_EXAMPLE_CAP`, `AMAZON_RECSYS_RANKER_VAL_EXAMPLE_CAP`, and `AMAZON_RECSYS_EVAL_USER_CAP`: cap ranker and evaluation workload without changing the raw data scan.
- `AMAZON_RECSYS_ENABLE_NEURAL_RETRIEVER`: controls whether the TensorFlow two-tower retriever is trained. Keep this off for fast local and CPU runs; enable it for production-quality experiments when compute allows.

For a quick small run:

```powershell
$env:AMAZON_RECSYS_CATEGORIES='["All_Beauty"]'
$env:AMAZON_RECSYS_DEV_MODE="true"
$env:AMAZON_RECSYS_DEV_FRACTION="0.1"
python -m amazon_recsys.cli.main export-bundle --run-name beauty-dev --run-profile debug --activate
```

For a production-style training run:

```powershell
$env:AMAZON_RECSYS_ENVIRONMENT="production"
$env:AMAZON_RECSYS_USE_MOCK_BUNDLE_IF_MISSING="false"
$env:AMAZON_RECSYS_MLFLOW_ENABLED="true"
$env:AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME="amazon-recsys-prod"
$env:AMAZON_RECSYS_CATEGORIES='["All_Beauty","Automotive","Industrial_and_Scientific"]'
python -m amazon_recsys.cli.main export-bundle --run-name prod-2026-04-28 --run-profile quality --version prod-2026-04-28
```

Review the metrics and exported artifacts before activation:

```powershell
python -m amazon_recsys.cli.main activate-bundle prod-2026-04-28
```

Use [`docs/running-the-app.md`](docs/running-the-app.md) for the detailed training configuration guide, including how profiles change the amount of data and compute used.

## Practical Azure Implementation

In practice, this system would usually sit behind a web or mobile product as an online recommendation service. A common Azure shape is:

- train and export a bundle from CI, Azure ML, or a scheduled batch job
- store the exported ONNX bundle under durable storage and activate the current version
- deploy the FastAPI service to Azure App Service, Azure Container Apps, or AKS
- expose `/recommend` or a product-specific endpoint behind Azure Front Door or API Management
- let the website, app, or backend call that endpoint whenever a user lands on a page, opens the app, views a product, or refreshes a personalized feed

The request flow is typically:

1. A user opens the site or app.
2. The frontend or backend sends the known `user_id`, or a short interaction/history payload for a cold-start session, to the recommender service.
3. The service loads the active bundle, generates candidates, ranks them, and returns the top items for that placement.
4. The client renders those recommendations on the homepage, product detail page, cart page, email widget, or in-app feed.
5. The serving layer logs the inference event so monitoring can later compare live traffic against the reference bundle.
6. When outcomes arrive later, such as clicks, purchases, ratings, or add-to-cart events, those can be ingested into the monitoring flow to measure drift and online performance.

On Azure, the clean separation is usually:

- website or mobile app: owns page rendering and user session context
- application backend: decides when to request recommendations and what business rules apply
- recommender service: owns ranking logic and active model bundle selection
- storage and MLflow: keep bundle artifacts, metrics, and lineage
- monitoring jobs: compute drift and online quality from served traffic plus delayed outcomes

That means this repo can be used either as:

- a direct recommendation microservice called synchronously during page load
- an internal backend service called by another API before the final page payload is assembled
- a batch or near-real-time generator for precomputed recommendation slots that are later cached at the edge

The simplest production version is usually a backend call on page load: when a user lands on the homepage or opens the app, the product backend calls this service, gets the top recommendations for that user, injects them into the response payload, and returns the final page or screen data to the client.

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
