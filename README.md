# Amazon Reviews Recommender System

This repository now supports two valid ways of working:

- a **notebook-first recommender workflow** that still contains the main working ML logic
- a **new production scaffold** for modular code, APIs, frontend demoing, testing, and Azure deployment

The project can grow from experimentation into an application without losing the notebook path that already works.

## Moving from the research environment into the production environment

The research environment was mostly:

- `RecSys.ipynb`
- `amazon_recsys_pipeline.py`
- research files in `Research/`

Now the production/development environment also includes:

- a modular package under `src/amazon_recsys/`
- a FastAPI + Jinja scaffold for serving and demoing recommendations
- tests
- Docker and compose files
- Azure-first infrastructure placeholders
- `template.py` for regenerating the scaffold structure

The shortest mental model is:

- **use the notebook to prove the model**
- **use `src/amazon_recsys/` to productionize the project**

## The Two Working Modes

### Notebook / research mode

This is still the main end-to-end implementation.

Primary files:

- `RecSys.ipynb`
- `amazon_recsys_pipeline.py`
- `DAT.ipynb`
- `Research/`

Use this mode for:

- EDA
- debugging the corpus
- recommender experimentation
- retrieval and ranking evaluation
- qualitative recommendation review

### Production scaffold mode

This is the new application-oriented codebase.

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

- modular refactoring
- service interfaces
- API development
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
|-- RecSys.ipynb
|-- DAT.ipynb
|-- amazon_recsys_pipeline.py
|-- Dockerfile
|-- docker-compose.yml
|-- .env.example
|-- src/
|   `-- amazon_recsys/
|-- tests/
|-- infra/
|   `-- azure/
|-- notebooks/
|-- Research/
|-- artifacts/
|-- amazon_review_data/
```

How to interpret that:

- `RecSys.ipynb` and `amazon_recsys_pipeline.py` are still the main working implementation
- `src/amazon_recsys/` is the target production codebase
- `artifacts/` stores caches, trained artifacts, and evaluation outputs
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

This is currently scaffold-level. The actual training logic still lives in the notebook path until the migration into `src/amazon_recsys/` is complete.

## Configuration Story

There are now effectively two configuration layers.

### Notebook configuration

The notebook workflow is still controlled mainly by `PipelineConfig` inside `amazon_recsys_pipeline.py`.

That covers:

- ingestion and sampling
- filtering
- training caps
- retriever and ranker choices
- evaluation controls
- artifact locations

### Application configuration

The production scaffold is moving toward typed settings under `src/amazon_recsys/config/`.

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

- `RecSys.ipynb`
- `amazon_recsys_pipeline.py`

That is still the correct place for:

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

- extracting reusable logic from the notebook path
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

- `RecSys.ipynb`

### Run the scaffolded app

```bash
uvicorn app:app --reload
```

### Run tests

```bash
pytest
```

## Migration Status

### Already real

- notebook recommender flow
- hybrid retrieval experimentation
- XGBoost-first ranking direction
- artifact generation and evaluation outputs
- package scaffold
- FastAPI + Jinja scaffold
- Docker and compose files
- Azure folder structure
- test skeletons

### Still in migration

- moving working ML logic out of `amazon_recsys_pipeline.py`
- hardening bundle loading and online inference
- wiring real Azure ML training jobs
- replacing placeholder adapters with migrated production logic

So the repository is better structured now, but still **mid-migration** rather than fully productionized.

## Recommended Mental Model

If the upgrade feels large, keep this short map in mind:

- `RecSys.ipynb` proves the recommender
- `amazon_recsys_pipeline.py` is the current working engine
- `src/amazon_recsys/` is the future production codebase
- `template.py` recreates the scaffold
- `infra/azure/` prepares deployment
- `tests/` protect the architecture while migration continues

That is the simplest way to make sense of the upgrade.
