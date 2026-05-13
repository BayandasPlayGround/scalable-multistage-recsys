from __future__ import annotations

import shutil
from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")
from mlflow.tracking import MlflowClient

from amazon_recsys.config.container import build_container


@pytest.mark.slow
@pytest.mark.config
@pytest.mark.data
@pytest.mark.retrieval
@pytest.mark.ranking
def test_training_run_and_bundle_export_are_logged_to_mlflow(test_settings, train_export_activate) -> None:
    tracking_dir = Path.cwd() / f"mlflow-pytest-{test_settings.workspace_root.name}"
    if tracking_dir.exists():
        shutil.rmtree(tracking_dir, ignore_errors=True)
    settings = test_settings.model_copy(
        update={
            "mlflow_enabled": True,
            "mlflow_experiment_name": "amazon-recsys-pytest",
            "mlflow_backend_root": tracking_dir,
            "mlflow_run_name_prefix": "pytest",
        }
    )
    try:
        settings.ensure_runtime_directories()
        container = build_container(settings)

        bundle = train_export_activate(container, version="mlflow-fixture", activate=False)

        assert bundle.session.mlflow_run_id is not None
        assert bundle.session.mlflow_tracking_uri == settings.mlflow.tracking_uri
        assert bundle.manifest.notes["mlflow_tracking_enabled"] is True
        assert bundle.manifest.notes["mlflow_run_id"] == bundle.session.mlflow_run_id

        client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
        run = client.get_run(bundle.session.mlflow_run_id)
        root_artifacts = {item.path for item in client.list_artifacts(bundle.session.mlflow_run_id)}
        bundle_artifacts = {item.path for item in client.list_artifacts(bundle.session.mlflow_run_id, "bundle")}

        assert run.data.tags["phase"] == "train"
        assert run.data.tags["bundle.version"] == "mlflow-fixture"
        assert run.data.params["training.run_name"] == "pytest"
        assert run.data.metrics["dataset.interactions"] > 0
        assert {"bundle", "evaluation", "training"}.issubset(root_artifacts)
        assert "bundle/manifest.json" in bundle_artifacts
        assert "bundle/evaluation_summary.json" in bundle_artifacts
        assert "bundle/runtime_bundle.json" not in bundle_artifacts
    finally:
        shutil.rmtree(tracking_dir, ignore_errors=True)
