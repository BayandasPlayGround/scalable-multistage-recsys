# Amazon Reviews Recommender System

This repository now supports two valid ways of working:

- a **package-first production workflow** under `src/amazon_recsys/`
- a **notebook workflow** that consumes the package-owned ML core

The recommender engine now lives in:

- `src/amazon_recsys/ml/core.py`

The notebook-facing entrypoint:

- `notebooks/amazon_recsys_pipeline.py`

is now a compatibility layer that re-exports the package implementation.

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
- `/recommend`
- `/users/{user_id}/history`
- `/models/active`
- `/evaluate/summary`

### Frontend direction

The Jinja-based UI is intended to support:

- user lookup
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
- `AzureConfig`

Environment-driven values are documented in:

- `.env.example`

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
