# Hybrid Amazon Reviews Recommender System

[Back to main README](../README.md)

The notebook workflow in this repo is a client of the package-owned recommender, not a separate implementation.

- notebook front end: [RecSys.ipynb](./RecSys.ipynb)
- compatibility import layer: [amazon_recsys_pipeline.py](./amazon_recsys_pipeline.py)
- package source of truth: `src/amazon_recsys/ml/core.py`

## What The Notebook Uses

The underlying package currently provides:

- a hybrid retrieval stack with popularity, cooccurrence, latent CF, content-based retrieval, and optional two-tower retrieval
- candidate union across retrieval sources
- XGBoost ranking by default, with `dlrm` kept as an experimental path
- leave-last-out evaluation, cached artifacts, and run profiles for local experimentation
- versioned bundle export for serving and monitoring

## How To Read The Notebook Layer

Use the notebook for:

- EDA and corpus inspection
- retrieval and ranking experiments
- qualitative recommendation review
- local debugging of package behavior

Do not treat the notebook files as the primary implementation boundary. The package under `src/amazon_recsys/` owns the recommender logic, serving artifacts, and runtime integration.

## Key Files

- [RecSys.ipynb](./RecSys.ipynb): notebook front end
- [amazon_recsys_pipeline.py](./amazon_recsys_pipeline.py): compatibility imports and notebook-facing helpers
- [Requirements.txt](../Requirements.txt): notebook dependency list
- [`amazon_review_data/`](../amazon_review_data): local review data and metadata
- [`artifacts/`](../artifacts): caches, trained models, evaluation outputs, bundles, and monitoring artifacts
- [`Research/`](../Research): notes and reference material

## Practical Starting Point

For the current default path:

1. install dependencies
2. start from the `quality` or `debug` profile
3. keep `ranker_backend="xgboost"`
4. leave the neural retriever disabled unless you are running an explicit ablation

If you need the deployed or API-backed runtime, switch back to the package CLI rather than extending the notebook layer.
