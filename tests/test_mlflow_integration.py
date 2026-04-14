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
def test_training_run_and_bundle_export_are_logged_to_mlflow(test_settings) -> None:
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

        session = container.training_pipeline.run(force_rebuild=True)
        manifest = container.artifact_store.save_bundle(session, version="mlflow-fixture")

        assert session.mlflow_run_id is not None
        assert session.mlflow_tracking_uri == settings.mlflow.tracking_uri
        assert manifest.notes["mlflow_tracking_enabled"] is True
        assert manifest.notes["mlflow_run_id"] == session.mlflow_run_id

        client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
        run = client.get_run(session.mlflow_run_id)
        root_artifacts = {item.path for item in client.list_artifacts(session.mlflow_run_id)}

        assert run.data.tags["phase"] == "train"
        assert run.data.tags["bundle.version"] == "mlflow-fixture"
        assert run.data.params["training.run_name"] == "pytest"
        assert run.data.metrics["dataset.interactions"] > 0
        assert {"bundle", "evaluation", "training"}.issubset(root_artifacts)
    finally:
        shutil.rmtree(tracking_dir, ignore_errors=True)
