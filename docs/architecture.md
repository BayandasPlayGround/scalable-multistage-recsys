# Architecture Guide

## From Research To Production

The earlier research environment was mostly:

- `notebooks/RecSys.ipynb`
- `notebooks/amazon_recsys_pipeline.py`
- research files in `Research/`

The current production/development environment includes:

- a modular package under `src/amazon_recsys/`
- a FastAPI + Jinja scaffold for serving and demoing recommendations
- tests
- Docker and compose files
- Azure-first infrastructure placeholders
- `template.py` for regenerating the scaffold structure

The short mental model is:

- use `src/amazon_recsys/` as the source of truth
- use the notebook as a client of that code

## Two Working Modes

### Notebook / Research Mode

Primary files:

- `notebooks/RecSys.ipynb`
- `notebooks/amazon_recsys_pipeline.py`
- `Research/`

Use this mode for:

- EDA
- corpus debugging
- recommender experimentation
- retrieval and ranking evaluation
- qualitative recommendation review

### Production Scaffold Mode

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

### Retrieval / Candidate Generation

Current direction:

- popularity backfill
- item-item cooccurrence / KNN
- latent collaborative filtering via sparse SVD
- content-based retrieval with TF-IDF + SVD text vectors
- optional neural two-tower retrieval
- hybrid candidate union

### Ranking

Default production path:

- `xgboost`

Experimental path:

- `dlrm`

### Data Framing

The pipeline currently uses:

- `parent_asin` as the canonical item ID
- positives from `rating >= 4`
- neutral interactions from `rating == 3`
- hard negatives from `rating <= 2`
- Amazon metadata enrichment
- leave-last-out temporal evaluation

## Software Architecture

The codebase is shaped as a modular monolith under `src/amazon_recsys/`.

- `config/`
  - settings, environment loading, dependency wiring
- `domain/`
  - entities, protocols, shared types
- `application/`
  - orchestration services and use cases
- `infrastructure/`
  - artifact loading, adapters, storage integrations
- `ml/`
  - core training, evaluation, bundle building
- `monitoring/`
  - reference profiles, inference/outcome logs, drift computation
- `api/`
  - FastAPI app, routers, request/response models
- `web/`
  - Jinja templates, static assets, UI routes
- `observability/`
  - logging and MLflow hooks
- `cli/`
  - train, evaluate, export, serve, monitor commands

## Repository Map

```text
.
|-- README.md
|-- docs/
|-- src/amazon_recsys/
|-- tests/
|-- notebooks/
|-- infra/azure/
|-- artifacts/
|-- amazon_review_data/
|-- app.py
|-- template.py
|-- Dockerfile
|-- docker-compose.yml
`-- .env.example
```

How to read that:

- `src/amazon_recsys/` is the production codebase and source of truth
- `notebooks/amazon_recsys_pipeline.py` is the notebook compatibility import layer
- `artifacts/` stores caches, trained artifacts, bundles, and monitoring outputs
- `amazon_review_data/` stores the local source data and metadata

## FastAPI + Jinja App Layer

The serving scaffold includes API routes such as:

- `/health`
- `/ready`
- `/config`
- `/users`
- `/recommend`
- `/users/{user_id}/history`
- `/models/active`
- `/evaluate/summary`
- `/monitoring/drift/summary`

The Jinja UI is intended to support:

- trained user lookup
- prior-order inspection
- recommendation review
- candidate provenance
- active model visibility
- latest monitoring summary visibility

## Azure-First Structure

Azure-oriented scaffolding lives under:

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

The deployment target is Azure-first, but the core recommender logic stays package-local and cloud-agnostic.

## Configuration Story

There are two main configuration layers.

### Core ML Configuration

The recommender workflow is controlled mainly by `PipelineConfig` inside `src/amazon_recsys/ml/core.py`.

That covers:

- ingestion and sampling
- filtering
- training caps
- retriever and ranker choices
- evaluation controls
- artifact locations

### Application / Runtime Configuration

The application runtime is controlled by typed settings under `src/amazon_recsys/config/`.

Main typed groups:

- `DataConfig`
- `TrainingConfig`
- `RetrievalConfig`
- `RankingConfig`
- `ServingConfig`
- `MLflowConfig`
- `MonitoringConfig`
- `AzureConfig`

Environment-driven values are documented in `.env.example`.

## Migration Status

Already real:

- package-owned ML core in `src/amazon_recsys/ml/core.py`
- notebook compatibility import layer
- hybrid retrieval experimentation
- XGBoost-first ranking direction
- artifact generation, evaluation outputs, bundle activation, and monitoring summaries
- FastAPI + Jinja scaffold
- Docker and compose files
- Azure folder structure

Still evolving:

- deeper decomposition of `ml/core.py`
- broader bundle export support beyond the current classical + XGBoost path
- real Azure ML job execution and live infra validation

## Recommended Mental Model

If the upgrade feels large, keep this short map in mind:

- `src/amazon_recsys/ml/core.py` is the recommender engine
- `notebooks/amazon_recsys_pipeline.py` is the notebook compatibility layer
- `src/amazon_recsys/application/services.py` serves active bundles
- `src/amazon_recsys/monitoring/` owns batch drift monitoring
- `template.py` recreates the scaffold
- `tests/` protect local and production-like behavior
