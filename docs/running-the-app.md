# Running The App

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

If you want training without bundle export:

```powershell
python -m amazon_recsys.cli.main train --run-name debug-local --run-profile debug
```

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
