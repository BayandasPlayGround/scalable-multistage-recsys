from __future__ import annotations

import argparse
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


SCAFFOLD_FILES = [
    ".env",
    "pyproject.toml",
    "Dockerfile",
    "docker-compose.yml",
    "app.py",
    "README.md",
    "docs/README.md",
    "docs/architecture.md",
    "docs/running-the-app.md",
    "docs/mlflow.md",
    "docs/monitoring.md",
    "docs/examples/outcomes.example.csv",
    "Requirements.txt",
    "notebooks/README.md",
    "Research/README.md",
    ".github/workflows/ci.yml",
    ".github/workflows/deploy-azure.yml",
    "infra/azure/bicep/main.bicep",
    "infra/azure/aml/README.md",
    "infra/azure/aml/environment.yml",
    "infra/azure/aml/train-job.yml",
    "infra/azure/aks/deployment.yaml",
    "src/amazon_recsys/__init__.py",
    "src/amazon_recsys/config/__init__.py",
    "src/amazon_recsys/config/settings.py",
    "src/amazon_recsys/config/container.py",
    "src/amazon_recsys/domain/__init__.py",
    "src/amazon_recsys/domain/entities.py",
    "src/amazon_recsys/domain/protocols.py",
    "src/amazon_recsys/application/__init__.py",
    "src/amazon_recsys/application/services.py",
    "src/amazon_recsys/infrastructure/__init__.py",
    "src/amazon_recsys/infrastructure/artifacts.py",
    "src/amazon_recsys/infrastructure/repositories.py",
    "src/amazon_recsys/infrastructure/retrievers.py",
    "src/amazon_recsys/infrastructure/rankers.py",
    "src/amazon_recsys/infrastructure/azure.py",
    "src/amazon_recsys/ml/__init__.py",
    "src/amazon_recsys/ml/core.py",
    "src/amazon_recsys/ml/legacy.py",
    "src/amazon_recsys/ml/bundles.py",
    "src/amazon_recsys/ml/feature_builders.py",
    "src/amazon_recsys/ml/pipelines.py",
    "src/amazon_recsys/monitoring/__init__.py",
    "src/amazon_recsys/monitoring/utils.py",
    "src/amazon_recsys/monitoring/metrics.py",
    "src/amazon_recsys/monitoring/reference.py",
    "src/amazon_recsys/monitoring/store.py",
    "src/amazon_recsys/monitoring/service.py",
    "src/amazon_recsys/api/__init__.py",
    "src/amazon_recsys/api/app.py",
    "src/amazon_recsys/api/dependencies.py",
    "src/amazon_recsys/api/schemas.py",
    "src/amazon_recsys/api/routers/__init__.py",
    "src/amazon_recsys/api/routers/health.py",
    "src/amazon_recsys/api/routers/models.py",
    "src/amazon_recsys/api/routers/monitoring.py",
    "src/amazon_recsys/api/routers/recommendations.py",
    "src/amazon_recsys/web/__init__.py",
    "src/amazon_recsys/web/router.py",
    "src/amazon_recsys/web/templates/base.html",
    "src/amazon_recsys/web/templates/dashboard.html",
    "src/amazon_recsys/web/static/app.js",
    "src/amazon_recsys/web/static/favicon.svg",
    "src/amazon_recsys/web/static/style.css",
    "src/amazon_recsys/observability/__init__.py",
    "src/amazon_recsys/observability/logging.py",
    "src/amazon_recsys/observability/mlflow.py",
    "src/amazon_recsys/cli/__init__.py",
    "src/amazon_recsys/cli/main.py",
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/test_template.py",
    "tests/test_api_app.py",
    "tests/test_container.py",
    "tests/test_mlflow_integration.py",
    "tests/test_monitoring_metrics.py",
    "tests/test_monitoring_integration.py",
    "tests/test_monitoring_store.py",
    "tests/test_pipeline_integration.py",
    "tests/test_notebook_compatibility.py",
    "tests/test_runtime_modes.py",
]

WORKFLOW_TEMPLATES = {
    ".github/workflows/ci.yml": """name: CI

on:
  workflow_dispatch:
  push:
    branches: ["main"]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install package
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e .[dev]
      - name: Run tests
        run: pytest -p no:cacheprovider
""",
    ".github/workflows/deploy-azure.yml": """name: Deploy Azure

on:
  workflow_dispatch:

jobs:
  validate-scaffold:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate Azure scaffold presence
        run: |
          test -f infra/azure/bicep/main.bicep
          test -f infra/azure/aml/train-job.yml
          test -f infra/azure/aks/deployment.yaml
""",
}


def create_project_template(target_root: Path, source_root: Path | None = None) -> list[Path]:
    source_root = source_root or Path(__file__).resolve().parent
    target_root = Path(target_root).resolve()
    source_root = Path(source_root).resolve()
    created_files: list[Path] = []

    for relative_path in SCAFFOLD_FILES:
        relative = Path(relative_path)
        source_path = source_root / relative
        target_path = target_root / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists() and target_path.stat().st_size > 0:
            logging.info("Skipping existing file: %s", target_path)
            continue

        if relative_path in WORKFLOW_TEMPLATES:
            target_path.write_text(WORKFLOW_TEMPLATES[relative_path], encoding="utf-8")
            logging.info("Created workflow template: %s", target_path)
            created_files.append(target_path)
            continue

        if source_path.exists() and source_path.resolve() != target_path.resolve():
            target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
            logging.info("Created file from template: %s", target_path)
        else:
            target_path.touch(exist_ok=True)
            logging.info("Created empty file: %s", target_path)
        created_files.append(target_path)

    return created_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Amazon RecSys production scaffold.")
    parser.add_argument(
        "--root",
        default=".",
        help="Target directory where the scaffold should be created. Defaults to the current directory.",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Optional source directory that contains the scaffold templates. Defaults to this script's directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    created = create_project_template(
        target_root=Path(args.root),
        source_root=Path(args.source_root).resolve() if args.source_root else None,
    )
    logging.info("Template creation complete. Files created: %s", len(created))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
