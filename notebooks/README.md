# Hybrid Amazon Reviews Recommender System

This project builds an end-to-end recommender system on top of the Amazon Reviews 2023 dataset samples in `amazon_review_data/`.

The current implementation is intentionally structured as a scalable two-stage recommendation system:

1. high-recall candidate generation
2. high-precision ranking

The notebook front-end is [RecSys.ipynb](./RecSys.ipynb).

The notebook import layer lives in [amazon_recsys_pipeline.py](./amazon_recsys_pipeline.py), but the package-owned ML implementation now lives in:

- `src/amazon_recsys/ml/core.py`

## What This Project Does

The pipeline:

- loads review events from:
  - `All_Beauty.jsonl`
  - `Automotive.jsonl`
  - `Industrial_and_Scientific.jsonl`
- optionally downloads matching Amazon Reviews 2023 metadata
- builds a filtered implicit-feedback recommendation corpus
- creates leave-last-out train, validation, and test splits
- trains a hybrid retrieval stack:
  - popularity by category
  - item-item cooccurrence
  - latent collaborative filtering via sparse SVD
  - content-based retrieval using TF-IDF + SVD text vectors
  - optional neural two-tower retrieval
- unions candidates from multiple sources for higher recall
- trains a ranker:
  - default: XGBoost LambdaMART-style ranking
  - optional: DLRM-like neural ranker
- produces final recommendations for:
  - warm users
  - sparse users
  - cold-start history-only scenarios

## Repository Layout

- [RecSys.ipynb](./RecSys.ipynb): main notebook
- [amazon_recsys_pipeline.py](./amazon_recsys_pipeline.py): notebook compatibility import layer over the package ML core
- [_generate_recsys_notebook.py](./_generate_recsys_notebook.py): notebook generator
- [Requirements.txt](./Requirements.txt): Python dependencies
- [`amazon_review_data/`](./amazon_review_data): raw review data and metadata
- [`artifacts/`](./artifacts): cached datasets, trained models, evaluation outputs
- [`Research/`](./Research): papers, notes, and reference material

## Architecture Overview

### 1. Data Ingestion and Corpus Preparation

The pipeline reads raw review JSONL files, extracts interaction signals, and converts them into a recommendation corpus.

Key design choices:

- canonical item ID: `parent_asin`
- positive interactions: `rating >= 4`
- neutral interactions: `rating == 3`
- hard negatives: `rating <= 2`
- item enrichment: Amazon metadata such as title, price, categories, description/features, average rating, rating count
- graph filtering: iterative `k-core` filtering to keep a learnable user-item graph

### 2. Split Strategy

The pipeline uses chronological leave-last-out splitting per user:

- train: all but the latest two positive interactions
- validation: penultimate positive interaction
- test: last positive interaction

This keeps the evaluation causal and avoids future leakage into training.

### 3. Retrieval Stage

The system is hybrid by design. Retrieval is not delegated to one model alone.

Implemented retrieval sources:

- `popularity_by_category`
  - backfill baseline
  - cheap and stable
- `item_item_cooccurrence`
  - strong collaborative baseline
  - useful for high recall
- `latent_cf`
  - sparse user-item matrix factorization via `TruncatedSVD`
  - ANN retrieval over latent item vectors
- `content_based`
  - TF-IDF over item text
  - reduced to dense vectors via `TruncatedSVD`
  - user query vector from recent item-history text profile
- `two_tower`
  - optional neural retriever
  - disabled by default in local quality/debug profiles because native Windows TensorFlow is CPU-only in this setup

### 4. Candidate Union

The hybrid candidate generator combines multiple retrieval sources into one candidate pool.

Default source budgets before deduplication:

- cooccurrence: `100`
- latent CF: `150`
- content-based: `100`
- neural two-tower: `150`
- popularity backfill: `50`

The union then:

- merges duplicate items across sources
- tracks provenance with `from_*` flags
- computes a union score from source-specific reciprocal-rank contributions
- keeps the top `candidate_union_top_k`

This design prioritizes recall first. The goal is to avoid missing good items before ranking begins.

### 5. Ranking Stage

The default precision stage is XGBoost ranking.

Implemented backends:

- `xgboost`
  - default
  - grouped ranking by `example_id`
  - best choice for current local CPU workflow
- `dlrm`
  - optional neural ablation
  - available through the pipeline, but not the default production path

The ranker uses features such as:

- retrieval score
- embedding similarity
- user history length
- user interaction count
- user mean rating
- user verified-purchase rate
- recency features
- category preference distribution
- per-source candidate provenance and ranks
- item metadata features
- item text similarity to user history
- price and rating deltas versus history
- hard-negative history flag

### 6. Final Recommendation Flow

At inference time:

1. collect a user's recent order history or supplied history items
2. generate candidates from multiple retrievers
3. remove already seen items
4. rerank candidates
5. return top-K recommendations with metadata and source provenance

## Current Implementation Philosophy

This project is optimized for:

- local experimentation on a CPU-first Windows laptop
- artifact reuse through caching
- transparent notebook diagnostics
- incremental scaling from fast smoke tests to larger quality runs

It is not optimized for:

- native Windows GPU TensorFlow training
- fully distributed training
- production online serving infrastructure

## Data and Feature Pipeline

### Review Signals

The review corpus is transformed into:

- positive interaction table
- hard-negative table
- item feature table
- text feature matrix

### Item Features

Item features combine:

- title
- source category
- price
- average rating
- rating count
- verified purchase rate
- helpful vote aggregates
- recency/popularity statistics
- dense text embedding from:
  - `TfidfVectorizer`
  - `TruncatedSVD`

### User Features

User features are derived from history and split examples:

- recent item history
- interaction count
- mean rating
- verified-purchase rate
- days since last interaction
- average days between interactions
- category-preference shares such as:
  - `pref_All_Beauty`
  - `pref_Automotive`
  - `pref_Industrial_and_Scientific`

## Main Pipeline API

The notebook is a front-end over these core functions:

- `prepare_corpus(config)`
  - raw-data processing
  - item/text feature generation
  - artifact caching
- `make_splits(prepared)`
  - temporal split creation
  - user/item indexing
  - cooccurrence map construction
- `train_retrievers(prepared, splits)`
  - trains the hybrid retrieval stack
- `generate_candidate_union(prepared, splits, retrievers, examples, ...)`
  - merges candidate sources into a single retrieval table
- `train_ranker(prepared, splits, retrievers, backend="xgboost")`
  - trains the final ranker
- `recommend(prepared, splits, retriever_or_retrievers, ranker=..., user_id=..., history_items=...)`
  - produces final recommendation tables
- `get_user_order_history(prepared, splits, user_id, ...)`
  - returns human-readable prior orders for evaluation/demo

## End-to-End Workflow

Typical notebook flow:

```python
prepared = prepare_corpus(CONFIG)
splits = make_splits(prepared)
retrievers = train_retrievers(prepared, splits)
ranker = train_ranker(prepared, splits, retrievers, backend=CONFIG.ranker_backend)

recommendations = recommend(
    prepared,
    splits,
    retrievers,
    ranker=ranker,
    user_id="some_user_id",
    top_k=10,
)
```

## Configuration Guide

All configuration is centralized in `PipelineConfig`.

The best way to work is:

1. create a `PipelineConfig`
2. apply a run profile with `apply_run_profile(config)`
3. override only the fields you explicitly care about

Example:

```python
CONFIG = apply_run_profile(
    PipelineConfig(
        base_dir=Path.cwd(),
        run_name="hybrid_quality_dev",
        run_profile="quality",
        dev_mode=True,
        ranker_backend="xgboost",
    )
)
```

### Run Profiles

The project supports three run profiles.

| Profile | Purpose | Key Behavior |
|---|---|---|
| `debug` | fastest smoke tests | smaller caps, neural retriever off, smaller candidate pools |
| `quality` | serious local experimentation | richer evaluation, neural retriever still off by default |
| `full` | largest local run | uncapped or much larger caps, neural retriever enabled |

Current profile defaults applied in code:

| Setting | `debug` | `quality` | `full` |
|---|---:|---:|---:|
| `retriever_train_example_cap` | 40,000 | 100,000 | `None` |
| `retriever_quality_min_history` | 2 | 3 | 3 |
| `enable_neural_retriever` | `False` | `False` | `True` |
| `eval_user_cap` | 1,000 | 2,000 | `None` |
| `candidate_union_top_k` | 200 | 200 | 300 |
| `candidate_union_batch_size` | 300 | 500 | 1,000 |
| `ranker_candidate_top_k` | 75 | 100 | 200 |
| `ranker_train_example_cap` | 2,000 | 5,000 | 50,000 |
| `ranker_val_example_cap` | 500 | 1,000 | 5,000 |

### Configuration Categories

#### 1. Paths and Run Identity

| Field | Default | Purpose |
|---|---|---|
| `base_dir` | `Path.cwd()` | repo root / working directory |
| `run_name` | `"default"` | artifact namespace under `artifacts/amazon_recsys/` |
| `cache_version` | `5` | manual cache invalidation when preprocessing logic changes |
| `run_profile` | `"quality"` | profile used by `apply_run_profile()` |

Artifacts are written to:

- `artifact_root = artifacts/amazon_recsys/<run_name>/`
- `cache_dir = artifact_root/cache/`
- `model_dir = artifact_root/models/`
- `eval_dir = artifact_root/evaluation/`

#### 2. Corpus and Sampling Controls

| Field | Default | Purpose |
|---|---:|---|
| `categories` | 3 categories | source review files to include |
| `k_core` | `5` | iterative graph filtering strength |
| `dev_mode` | `False` | enables smaller sampled development runs |
| `dev_fraction` | `0.05` | base keep fraction in dev mode |
| `dev_sampling_strategy` | `"stratified_user"` | dev sampler variant |
| `dev_hard_negative_multiplier` | `2.5` | hard-negative upweighting in dev mode |
| `dev_neutral_multiplier` | `1.5` | neutral-rating upweighting in dev mode |
| `max_rows_per_category` | `None` | hard cap for raw rows per category |
| `train_positive_cap` | `2_000_000` | cap after preprocessing for training corpus size |
| `review_chunk_rows` | `250_000` | raw ingestion chunk size |

Supported dev sampling strategies:

- `user`
  - deterministic user-level sampling
- `stratified_user`
  - user-level sampling with rating-aware upweighting
- `category_balanced_user`
  - user-level sampling plus rough category balancing

#### 3. Text and Embedding Controls

| Field | Default | Purpose |
|---|---:|---|
| `history_len` | `10` | max recent-history length used in examples |
| `text_max_features` | `12_000` | TF-IDF vocabulary size |
| `text_svd_dim` | `64` | dense text vector dimension |
| `retriever_embedding_dim` | `64` | neural retriever embedding width |
| `latent_cf_components` | `64` | latent CF dimension upper bound |
| `memory_map_item_text` | `True` | memory-map text matrix from disk when possible |

#### 4. Neural Retriever Controls

| Field | Default | Purpose |
|---|---:|---|
| `enable_neural_retriever` | `False` | include two-tower retriever in hybrid stack |
| `retriever_hidden_dims` | `(256, 128, 64)` | two-tower MLP widths |
| `retriever_batch_size` | `256` | training batch size |
| `retriever_epochs` | `3` | training epochs |
| `retriever_train_example_cap` | `50_000` | cap on sampled base retriever examples |
| `retriever_shuffle_buffer` | `20_000` | TensorFlow shuffle buffer |
| `negatives_per_positive` | `3` | train negatives per positive |
| `retriever_validation_negatives_per_positive` | `10` | validation negatives per positive |
| `retriever_quality_min_history` | `2` | minimum usable history length |
| `retriever_logit_scale` | `8.0` | scaled-cosine logit factor |
| `persist_encoder_models` | `False` | save `.keras` encoder models if serializable |
| `in_batch_weight` | `0.15` | weight for in-batch retrieval loss component |
| `dat_mimic_weight` | `0.10` | DAT-lite auxiliary loss weight |
| `dat_category_alignment_weight` | `0.05` | DAT-lite category regularizer weight |

Recommended use:

- keep `enable_neural_retriever=False` for fast local CPU work
- enable it only for ablation or larger quality/full runs

#### 5. Candidate Generation Controls

| Field | Default | Purpose |
|---|---:|---|
| `ann_trees` | `50` | Annoy index tree count |
| `retrieval_top_k` | `100` | top-K for single-retriever evaluation |
| `cooccurrence_candidate_k` | `100` | cooccurrence contribution budget |
| `latent_cf_candidate_k` | `150` | latent CF contribution budget |
| `content_candidate_k` | `100` | content retrieval contribution budget |
| `neural_candidate_k` | `150` | neural contribution budget |
| `popularity_backfill_k` | `50` | popularity fallback budget |
| `candidate_union_top_k` | `300` | final candidate union size before ranker pruning |
| `candidate_union_batch_size` | `500` | batch size for chunked union generation |

#### 6. Ranker Controls

| Field | Default | Purpose |
|---|---:|---|
| `ranker_backend` | `"xgboost"` | `xgboost` or `dlrm` |
| `ranker_candidate_top_k` | `200` | per-example candidates passed to ranker |
| `ranker_train_example_cap` | `2_000` | max train examples for ranker training |
| `ranker_val_example_cap` | `1_000` | max validation examples for ranker evaluation |
| `ranker_negatives_per_positive` | `10` | negative downsampling during ranker training |
| `ranker_batch_size` | `512` | neural ranker batch size |
| `ranker_epochs` | `3` | neural ranker epochs |
| `ranker_embedding_dim` | `16` | neural ranker ID-embedding width |
| `ranker_hidden_dims` | `(128, 64, 32)` | neural ranker MLP widths |

XGBoost-specific settings:

| Field | Default | Purpose |
|---|---:|---|
| `xgb_learning_rate` | `0.05` | boosting step size |
| `xgb_n_estimators` | `300` | number of trees |
| `xgb_max_depth` | `6` | max tree depth |
| `xgb_subsample` | `0.8` | row subsampling |
| `xgb_colsample_bytree` | `0.8` | feature subsampling |

#### 7. Runtime and Convenience Controls

| Field | Default | Purpose |
|---|---:|---|
| `eval_user_cap` | `1_000` | evaluation-user cap for notebook safety |
| `show_progress` | `True` | progress bars for preprocessing |
| `metadata_download_if_missing` | `True` | auto-download metadata files if needed |
| `training_verbose` | `2` | Keras verbosity |
| `tf_prefetch_batches` | `1` | TensorFlow dataset prefetching |
| `seed` | `42` | global reproducibility seed |

## Practical Configuration Recipes

### Fast Smoke Test

Use this when you want to check that the notebook runs from top to bottom.

```python
CONFIG = apply_run_profile(
    PipelineConfig(
        base_dir=Path.cwd(),
        run_name="debug_smoke",
        run_profile="debug",
        dev_mode=True,
        dev_fraction=0.10,
        ranker_backend="xgboost",
    )
)
```

Optional additional reductions:

```python
CONFIG.retriever_epochs = 3
CONFIG.xgb_n_estimators = 100
```

### Balanced Local Quality Run

This is the current recommended local setting.

```python
CONFIG = apply_run_profile(
    PipelineConfig(
        base_dir=Path.cwd(),
        run_name="hybrid_quality_dev",
        run_profile="quality",
        dev_mode=True,
        dev_fraction=0.20,
        dev_sampling_strategy="category_balanced_user",
        ranker_backend="xgboost",
    )
)
```

### Full Local Run

Use this when you want the richest offline evaluation and can tolerate much higher runtime.

```python
CONFIG = apply_run_profile(
    PipelineConfig(
        base_dir=Path.cwd(),
        run_name="hybrid_full",
        run_profile="full",
        dev_mode=False,
        ranker_backend="xgboost",
    )
)
```

### Neural Retriever Ablation

Use this only when you explicitly want to test the two-tower retriever.

```python
CONFIG.enable_neural_retriever = True
CONFIG.retriever_epochs = 10
```

## Evaluation Outputs

The notebook writes evaluation artifacts to:

- `artifacts/amazon_recsys/<run_name>/evaluation/`

Common outputs include:

- `baseline_metrics.csv`
- `content_based_retriever_metrics.csv`
- `latent_cf_retriever_metrics.csv`
- `two_tower_retriever_metrics.csv` when enabled
- `hybrid_union_metrics.csv`
- ranker metrics CSVs
- candidate tables for validation and test
- retriever pair summaries
- retriever sanity checks

## Demo Recommendation Output

The final demo is designed to be human-readable.

For each user scenario, the notebook shows:

- prior order history
- category and rating context
- final recommendations
- provenance such as:
  - `cooccurrence`
  - `latent_cf`
  - `content_based`
  - `popularity`
  - `two_tower` when enabled

This makes qualitative inspection easier when offline metrics alone are not enough.

## Caching and Artifact Behavior

The pipeline caches aggressively because raw data processing is expensive.

Key behavior:

- config-sensitive artifacts are rebuilt when cache-sensitive settings change
- processed data lives in `cache_dir`
- trained retriever/ranker artifacts live in `model_dir`
- evaluation outputs live in `eval_dir`

If a cached parquet file is locked by Jupyter or another process, the pipeline is designed to fail clearly rather than silently corrupting the cache.

## Environment Notes

### Native Windows TensorFlow

On this machine and this codebase, TensorFlow is CPU-bound on native Windows.

Implication:

- the neural retriever can be much slower than the classical retrievers
- the default notebook profile keeps `enable_neural_retriever=False`

If you later want hardware acceleration, the cleaner path is usually:

- WSL2/Linux for TensorFlow
- or a separate PyTorch XPU path for Intel Arc

### Memory Management

The pipeline already includes notebook-safety measures such as:

- training caps
- evaluation caps
- candidate generation batching
- optional memory-mapped text matrix loading
- explicit garbage collection in heavy sections

If a run is still too large:

- lower `ranker_train_example_cap`
- lower `ranker_val_example_cap`
- lower `candidate_union_top_k`
- lower `candidate_union_batch_size`
- use `run_profile="debug"`

## Known Limitations

- the neural retriever is still an ablation path, not the default production winner
- native Windows TensorFlow training is slow relative to a Linux/GPU setup
- recommendation quality is sensitive to graph density after filtering
- multi-category recommendation is harder than single-domain toy notebooks
- offline metrics should be interpreted together with candidate diagnostics and qualitative demos

## Suggested Starting Point

If you are new to the repo, start with:

1. install dependencies from [Requirements.txt](./Requirements.txt)
2. open [RecSys.ipynb](./RecSys.ipynb)
3. run the notebook with the default `quality` dev configuration
4. inspect:
   - raw audit
   - split diagnostics
   - baseline metrics
   - hybrid union metrics
   - ranker metrics
   - final demo recommendations

## Installation

Create an environment and install:

```bash
pip install -r Requirements.txt
```

Minimum practical dependencies used by the current default path:

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `pyarrow`
- `annoy`
- `tensorflow`
- `xgboost`
- `tqdm`
- `nbformat`

Optional but supported:

- `tensorflow-recommenders`
- `torch`
- `transformers`
- `sentence_transformers`

## Summary

This repository implements a practical hybrid recommender system with:

- Amazon Reviews 2023 interaction data
- content and collaborative retrieval
- hybrid candidate union
- XGBoost ranking
- notebook-safe configuration profiles
- reusable cached artifacts
- qualitative demo tooling for model inspection

If you only remember one configuration principle, use this one:

- start with `run_profile="quality"`, `dev_mode=True`, `ranker_backend="xgboost"`, and keep `enable_neural_retriever=False` until the classical retrieval stack is stable.
