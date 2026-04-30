# Running The App

[Back to docs hub](README.md) | [Back to main README](../README.md)

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
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/ready`

## Beginner Command Walkthrough

These are the main commands:

```powershell
pip install -e .[dev]
Copy-Item .env.example .env -Force
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
python -m amazon_recsys.cli.main serve
```

### Step 1: Install The Project

```powershell
pip install -e .[dev]
```

Meaning:

- installs the package in editable mode
- installs dev dependencies such as pytest
- makes `amazon_recsys` importable from the CLI

### Step 2: Create Your Local `.env`

```powershell
Copy-Item .env.example .env -Force
```

Meaning:

- copies the safe template to a real local runtime config
- gives you a working default setup

Important defaults:

- `AMAZON_RECSYS_ENVIRONMENT=local`
- `AMAZON_RECSYS_RUN_PROFILE=debug`
- `AMAZON_RECSYS_USE_MOCK_BUNDLE_IF_MISSING=true`
- `AMAZON_RECSYS_RANKER_BACKEND=xgboost`

### Step 3: Train, Export, And Activate A Bundle

```powershell
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
```

Training happens here.

This command:

- loads settings from `.env`
- builds the dependency container
- prepares the corpus
- builds train, validation, and test splits
- trains the retrievers
- trains the ranker
- exports a versioned serving bundle
- activates that bundle for online serving

Outputs land under:

- `artifacts/amazon_recsys/bundles/`
- `artifacts/production/active_bundle.json`

New bundle versions are portable ONNX bundle directories. The deployable ranker lives at `models/ranker.onnx`, while `runtime_bundle.json` points to the JSON, Parquet, NumPy, and Annoy artifacts needed by retrieval, serving, monitoring, and evaluation summary views.

If you want training without bundle export:

```powershell
python -m amazon_recsys.cli.main train --run-name debug-local --run-profile debug
```

## How Training Configuration Works

The training entry point is `src/amazon_recsys/cli/main.py`. It loads `AppSettings` from `.env` and from the current shell environment, then builds a package-owned `PipelineConfig` in `src/amazon_recsys/ml/pipelines.py`.

Configuration precedence is:

1. Values already in the shell, such as `$env:AMAZON_RECSYS_RUN_PROFILE="quality"`.
2. Values in `.env`.
3. Defaults in `src/amazon_recsys/config/settings.py`.
4. Profile defaults applied by `src/amazon_recsys/ml/core.py`.
5. CLI overrides for `--run-name`, `--run-profile`, `--workspace-root`, `--host`, and `--port`.

The normal training commands are:

```powershell
python -m amazon_recsys.cli.main train --run-name debug-local --run-profile debug
python -m amazon_recsys.cli.main evaluate --run-name debug-local --run-profile debug
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
```

Use `train` when you only want metrics and cached model artifacts. Use `evaluate` when you want an evaluation summary and the metric CSV files already exist, or when you want the command to train first if metrics are missing. Use `export-bundle` for the deployable path because it trains, exports a versioned runtime bundle, and can optionally activate that bundle.

### Run Profiles

`AMAZON_RECSYS_RUN_PROFILE` and `--run-profile` accept four values:

| Profile | Best for | Data and compute behavior |
| --- | --- | --- |
| `debug` | Local smoke tests and quick iteration | Reads up to `100000` review rows per category, disables the neural retriever, caps retriever training at `40000` examples, caps ranker training at `2000` examples, and evaluates up to `1000` users. |
| `quality` | Stronger offline experiments on CPU | Uses the full configured review files, disables the neural retriever by default, caps retriever training at `100000` examples, caps ranker training at `5000` examples, and evaluates up to `2000` users. |
| `quality-neural` | Optional neural retrieval experiment | Uses the quality-sized data and caps, but enables the TensorFlow two-tower retriever for candidate-recovery experiments. |
| `full` | Heavier production-style experiments | Uses the full configured review files, enables the neural retriever, removes the retriever training cap, caps ranker training at `50000` examples, and removes the evaluation user cap. |

Profile defaults are applied only when the corresponding internal `PipelineConfig` value is still at its package default. Settings exposed through `.env`, such as ranker caps and candidate sizes, can still override many of the runtime limits.

### Data Files

The code expects Amazon review files under:

```text
amazon_review_data/
  All_Beauty.jsonl
  Automotive.jsonl
  Industrial_and_Scientific.jsonl
  metadata/
    meta_All_Beauty.jsonl.gz
    meta_Automotive.jsonl.gz
    meta_Industrial_and_Scientific.jsonl.gz
```

The default review categories are:

```text
All_Beauty
Automotive
Industrial_and_Scientific
```

Metadata can be downloaded automatically for the built-in categories when `AMAZON_RECSYS_METADATA_DOWNLOAD_IF_MISSING=true`. Review files are not downloaded by the training command; they must already exist in the configured data directory.

To use a different data directory:

```powershell
$env:AMAZON_RECSYS_DATA_DIR="D:\datasets\amazon_review_data"
python -m amazon_recsys.cli.main export-bundle --run-name external-data --run-profile debug --activate
```

### Changing The Volume Of Data Trained On

The practical controls are:

| Setting | Effect |
| --- | --- |
| `AMAZON_RECSYS_RUN_PROFILE` | Broadest volume and compute control. `debug` is smallest, `quality` is medium, `full` is largest. |
| `AMAZON_RECSYS_CATEGORIES` | Controls how many category files are included. This must be a JSON list, for example `["All_Beauty"]`. |
| `AMAZON_RECSYS_DEV_MODE` | Enables deterministic sampling before preprocessing. |
| `AMAZON_RECSYS_DEV_FRACTION` | Fraction of configured category rows to keep when dev mode is enabled. Must be greater than `0` and less than or equal to `1`. |
| `AMAZON_RECSYS_K_CORE` | Minimum positive interactions per user and item. Lower values keep more sparse users/items; higher values reduce data but improve density. |
| `AMAZON_RECSYS_TRAIN_POSITIVE_CAP` | Caps training split examples while they are being generated. This is the main memory control for Stage 2 split construction. |
| `AMAZON_RECSYS_SPLIT_EVAL_EXAMPLE_CAP` | Caps validation and test split examples while they are being generated. This keeps evaluation/ranker validation sets from holding millions of users in memory. |
| `AMAZON_RECSYS_RANKER_TRAIN_EXAMPLE_CAP` | Caps examples used to train the ranker after train/validation/test examples are built. |
| `AMAZON_RECSYS_RANKER_VAL_EXAMPLE_CAP` | Caps validation examples used by the ranker. |
| `AMAZON_RECSYS_EVAL_USER_CAP` | Caps users evaluated in offline retrieval/ranking metrics. |
| `AMAZON_RECSYS_ENABLE_NEURAL_RETRIEVER` | Adds the TensorFlow neural retriever. This increases training time and memory use. |

Important implementation detail: the internal pipeline has a `max_rows_per_category` field. Today it is controlled by run profile defaults and is not exposed directly as an `AMAZON_RECSYS_*` environment variable. Use `debug` for the built-in 100k-per-category cap, or use `dev_mode` plus `dev_fraction` to sample smaller volumes.

Training commands emit stage-level `INFO` logs by default, including corpus preparation, split creation, retriever training, ranker training, bundle export, and activation. For row-level and chunk-level progress bars during long local runs, also enable:

```powershell
$env:AMAZON_RECSYS_SHOW_PROGRESS="true"
```

Small single-category experiment:

```powershell
$env:AMAZON_RECSYS_CATEGORIES='["All_Beauty"]'
$env:AMAZON_RECSYS_DEV_MODE="true"
$env:AMAZON_RECSYS_DEV_FRACTION="0.1"
$env:AMAZON_RECSYS_K_CORE="2"
$env:AMAZON_RECSYS_TRAIN_POSITIVE_CAP="50000"
$env:AMAZON_RECSYS_SPLIT_EVAL_EXAMPLE_CAP="1000"
$env:AMAZON_RECSYS_RANKER_TRAIN_EXAMPLE_CAP="1000"
$env:AMAZON_RECSYS_RANKER_VAL_EXAMPLE_CAP="250"
python -m amazon_recsys.cli.main export-bundle --run-name beauty-small --run-profile debug --activate
```

Medium multi-category experiment:

```powershell
$env:AMAZON_RECSYS_CATEGORIES='["All_Beauty","Automotive","Industrial_and_Scientific"]'
$env:AMAZON_RECSYS_DEV_MODE="false"
$env:AMAZON_RECSYS_K_CORE="3"
$env:AMAZON_RECSYS_ENABLE_NEURAL_RETRIEVER="false"
$env:AMAZON_RECSYS_TRAIN_POSITIVE_CAP="500000"
$env:AMAZON_RECSYS_SPLIT_EVAL_EXAMPLE_CAP="5000"
$env:AMAZON_RECSYS_RANKER_TRAIN_EXAMPLE_CAP="5000"
$env:AMAZON_RECSYS_EVAL_USER_CAP="2000"
python -m amazon_recsys.cli.main export-bundle --run-name quality-local --run-profile quality --activate
```

Heavy production-style experiment:

```powershell
$env:AMAZON_RECSYS_CATEGORIES='["All_Beauty","Automotive","Industrial_and_Scientific"]'
$env:AMAZON_RECSYS_DEV_MODE="false"
$env:AMAZON_RECSYS_K_CORE="5"
$env:AMAZON_RECSYS_ENABLE_NEURAL_RETRIEVER="true"
$env:AMAZON_RECSYS_TRAIN_POSITIVE_CAP="2000000"
$env:AMAZON_RECSYS_SPLIT_EVAL_EXAMPLE_CAP="10000"
$env:AMAZON_RECSYS_RANKER_TRAIN_EXAMPLE_CAP="50000"
$env:AMAZON_RECSYS_RANKER_VAL_EXAMPLE_CAP="5000"
python -m amazon_recsys.cli.main export-bundle --run-name prod-candidate --run-profile full --version prod-candidate
```

If a sampled run fails with an empty split, increase `AMAZON_RECSYS_DEV_FRACTION`, add more categories, or lower `AMAZON_RECSYS_K_CORE`. Split generation needs enough users with at least three positive interactions after filtering because the latest two interactions become validation and test targets.

### Candidate And Ranking Controls

These settings do not directly change how many raw rows are read, but they materially affect training cost and recommendation quality:

| Setting | Meaning |
| --- | --- |
| `AMAZON_RECSYS_CANDIDATE_UNION_TOP_K` | Maximum candidate union size before ranking. Higher values give the ranker more choices and cost more. |
| `AMAZON_RECSYS_CANDIDATE_UNION_BATCH_SIZE` | Batch size for candidate generation. Increase carefully when memory is available. |
| `AMAZON_RECSYS_COOCCURRENCE_CANDIDATE_K` | Candidate count from cooccurrence retrieval. |
| `AMAZON_RECSYS_LATENT_CF_CANDIDATE_K` | Candidate count from latent collaborative filtering. |
| `AMAZON_RECSYS_CONTENT_CANDIDATE_K` | Candidate count from content retrieval. |
| `AMAZON_RECSYS_NEURAL_CANDIDATE_K` | Candidate count from the neural retriever when enabled. |
| `AMAZON_RECSYS_RANKER_CANDIDATE_TOP_K` | Number of candidates considered by the ranker per example. |
| `AMAZON_RECSYS_RANKER_NEGATIVES_PER_POSITIVE` | Negative sampling ratio for ranker training. Higher values can improve discrimination but increase training cost. |
| `AMAZON_RECSYS_RANKER_BACKEND` | `xgboost` by default. `dlrm` is available as an experimental TensorFlow backend. |

Candidate recovery can be inspected on an exported bundle without retraining:

```powershell
python -m amazon_recsys.cli.main diagnose-candidates --bundle-version active --split test --sample-size 500
```

The XGBoost ranker is controlled by:

```text
AMAZON_RECSYS_XGB_LEARNING_RATE
AMAZON_RECSYS_XGB_N_ESTIMATORS
AMAZON_RECSYS_XGB_MAX_DEPTH
AMAZON_RECSYS_XGB_SUBSAMPLE
AMAZON_RECSYS_XGB_COLSAMPLE_BYTREE
```

### Cache Rebuilds

Corpus preprocessing is cached under:

```text
artifacts/amazon_recsys/<run-name>/cache/
```

The pipeline automatically rebuilds cached corpus artifacts when cache-sensitive config changes are detected. Use `--force-rebuild` when you want to rebuild everything for the current run name:

```powershell
python -m amazon_recsys.cli.main export-bundle --run-name quality-local --run-profile quality --force-rebuild --activate
```

Changing `--run-name` is often cleaner for experiments because it creates a separate artifact root and keeps old metrics and bundles available for comparison.

### Outputs From A Training Run

A successful `export-bundle` run writes:

```text
artifacts/amazon_recsys/<run-name>/
  cache/
  evaluation/
  models/
  config.json

artifacts/amazon_recsys/bundles/<bundle-version>/
  runtime_bundle.json
  models/ranker.onnx
  ...

artifacts/production/active_bundle.json
```

`config.json` is the resolved internal pipeline config for the run. The evaluation directory contains metric CSV files. The bundle directory is the portable serving unit used by the API.

### Step 4: Start The API And Web App

```powershell
python -m amazon_recsys.cli.main serve
```

This starts FastAPI and the Jinja UI. The app tries to load the active real bundle first, and only falls back to a mock bundle if mock mode is enabled.

## Local Dev Vs Production-Like Mode

### Local Dev Mode

Use this when you want fast feedback and safe startup behavior.

Recommended settings:

- `AMAZON_RECSYS_ENVIRONMENT=local`
- `AMAZON_RECSYS_USE_MOCK_BUNDLE_IF_MISSING=true`
- `AMAZON_RECSYS_RUN_PROFILE=debug`
- `AMAZON_RECSYS_ENABLE_NEURAL_RETRIEVER=false`

Typical commands:

```powershell
pip install -e .[dev]
uvicorn app:app --reload
pytest -m "foundation or config or serving" -p no:cacheprovider
```

If you want a real local bundle:

```powershell
python -m amazon_recsys.cli.main export-bundle --run-name debug-local --run-profile debug --activate
python -m amazon_recsys.cli.main serve
```

### Production-Like Local Mode

Use this when you want readiness behavior closer to deployment.

Recommended settings:

- `AMAZON_RECSYS_ENVIRONMENT=production`
- `AMAZON_RECSYS_DEBUG=false`
- `AMAZON_RECSYS_RELOAD=false`
- `AMAZON_RECSYS_USE_MOCK_BUNDLE_IF_MISSING=false`

Typical commands:

```powershell
python -m amazon_recsys.cli.main export-bundle --run-name prod-local --run-profile debug --activate
python -m amazon_recsys.cli.main serve
```

Expected behavior:

- `/ready` returns `503` before a real bundle is activated
- `/ready` returns `200` after activation
- `/models/active` reports active bundle metadata

## Best Practices For Production Training Runs

Use immutable run and bundle names. Prefer names that include the date, data snapshot, git SHA, or pipeline version, such as `prod-2026-04-28-main-a1b2c3d`. Avoid overwriting a production candidate with the same `--run-name`.

Train and export first, then activate after review:

```powershell
python -m amazon_recsys.cli.main export-bundle --run-name prod-2026-04-28 --run-profile quality --version prod-2026-04-28
python -m amazon_recsys.cli.main activate-bundle prod-2026-04-28
```

Keep MLflow enabled for production candidates:

```powershell
$env:AMAZON_RECSYS_MLFLOW_ENABLED="true"
$env:AMAZON_RECSYS_MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"
$env:AMAZON_RECSYS_MLFLOW_EXPERIMENT_NAME="amazon-recsys-prod"
$env:AMAZON_RECSYS_MLFLOW_RUN_NAME_PREFIX="prod"
```

Review these before activation:

- `artifacts/amazon_recsys/<run-name>/config.json`
- metric CSV files under `artifacts/amazon_recsys/<run-name>/evaluation/`
- MLflow parameters, metrics, artifacts, and tags
- bundle contents under `artifacts/amazon_recsys/bundles/<version>/`
- `/models/active` after activation
- `/ready` after the service refreshes the active bundle

Recommended production defaults:

- use `quality` as the normal scheduled training profile
- reserve `full` for planned heavier experiments with enough CPU, memory, and time budget
- keep `AMAZON_RECSYS_DEV_MODE=false`
- keep `AMAZON_RECSYS_USE_MOCK_BUNDLE_IF_MISSING=false`
- use `AMAZON_RECSYS_ENVIRONMENT=production`
- use a remote or durable MLflow backend
- store exported bundles in durable storage before deployment
- activate only a reviewed bundle version
- run monitoring after serving traffic and delayed outcomes are available

For Azure ML, the scaffold under `infra/azure/aml/` runs the same package CLI:

```text
python -m amazon_recsys.cli.main export-bundle --run-name aml-run --run-profile debug --version aml-${{name}} --activate
```

For a real production Azure ML job, change the profile and run naming in `infra/azure/aml/train-job.yml`, point data and artifact paths at durable mounted storage, and publish the exported bundle to your registry or storage account before activating it in the serving environment.

## Verification Matrix

Run the full suite:

```powershell
pytest -p no:cacheprovider
```

Fast local checks:

```powershell
pytest -m "foundation or config or serving" -p no:cacheprovider
```

Training and serving checks:

```powershell
pytest -m "data or retrieval or ranking or serving" -p no:cacheprovider
```

Environment mode checks:

```powershell
pytest tests/test_runtime_modes.py -p no:cacheprovider
```

Notebook compatibility check:

```powershell
pytest tests/test_notebook_compatibility.py -p no:cacheprovider
```
