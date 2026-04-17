# Architecture Guide

## System Shape

The repo is centered on the package under `src/amazon_recsys/`. That package owns the recommender, serving, monitoring, configuration, and CLI flows. Notebook files, template assets, and deployment folders still exist, but they are support layers around the package rather than the main implementation.

Keep this mental model:

- `src/amazon_recsys/` is the source of truth
- notebooks consume the package through a compatibility import surface
- `template.py` and `infra/azure/` are secondary bootstrap and deployment assets

## Recommender

The current recommender is a multi-stage system built in `src/amazon_recsys/ml/core.py`.

Core data framing:

- `parent_asin` is the canonical item ID
- positives come from `rating >= 4`
- neutral interactions come from `rating == 3`
- hard negatives come from `rating <= 2`
- evaluation uses leave-last-out temporal splits

Retrieval stack:

- popularity backfill
- item-item cooccurrence
- latent collaborative filtering via sparse SVD
- content-based retrieval with TF-IDF plus SVD text features
- optional neural two-tower retrieval
- hybrid candidate union across sources

Ranking:

- default backend: `xgboost`
- experimental backend: `dlrm`

Training outputs are written into run-scoped artifacts and can be packaged into versioned serving bundles.

## Serving

Serving is bundle-backed.

- `python -m amazon_recsys.cli.main export-bundle ... --activate` trains, exports, and activates a bundle
- `python -m amazon_recsys.cli.main serve` starts the FastAPI app and Jinja UI
- `src/amazon_recsys/application/services.py` loads the active bundle and refreshes when activation changes

Main runtime surfaces:

- `/health`
- `/ready`
- `/config`
- `/users`
- `/recommend`
- `/users/{user_id}/history`
- `/users/{user_id}/profile`
- `/models/active`
- `/evaluate/summary`
- `/monitoring/drift/summary`
- `/monitoring/drift/history`

If no active bundle exists, the service can return a mock bundle only when `use_mock_bundle_if_missing` is enabled. The real serving path is always the active exported bundle.

## Monitoring

Monitoring is batch and bundle-scoped.

- export builds a reference profile for the bundle
- serving records inference events after successful recommendations
- delayed outcomes are ingested or simulated
- monitoring compares the active bundle reference profile against served traffic and outcomes

Operational entry points:

- `ingest-outcomes`
- `simulate-outcomes`
- `monitor-drift`
- `monitor-backfill`

The monitoring package persists latest summaries plus history, computes feature drift and concept drift, and logs monitoring runs to MLflow when enabled.

## Compatibility

Notebook compatibility is still deliberate.

- `notebooks/amazon_recsys_pipeline.py` is the notebook import layer over the package ML core
- `notebooks/RecSys.ipynb` remains the notebook front end
- `src/amazon_recsys/config/settings.py` preserves legacy workspace and artifact path resolution for notebook-era layouts

This keeps older notebook workflows usable without moving the source of truth back out of the package.

## Secondary Support Assets

These pieces still matter, but they are not the primary system:

- `template.py` copies the repo template structure into a target root
- `infra/azure/` holds deployment-oriented Azure assets
- Docker, Compose, and workflow files support packaging and deployment work

Treat them as bootstrap and deployment support around the current recommender platform, not as the core implementation.
