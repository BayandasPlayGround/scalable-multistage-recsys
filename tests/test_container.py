from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from amazon_recsys.config.container import build_container
from amazon_recsys.config.settings import AppSettings
from amazon_recsys.domain.entities import BundleManifest, EvaluationSummary, RuntimeBundle
from amazon_recsys.infrastructure.artifacts import LocalArtifactStore


@pytest.mark.config
def test_settings_use_notebook_data_fallback(workspace_dir: Path) -> None:
    notebook_data_dir = workspace_dir / "notebooks" / "amazon_review_data"
    notebook_data_dir.mkdir(parents=True)

    settings = AppSettings(workspace_root=workspace_dir, data_dir=Path("amazon_review_data"))

    assert settings.legacy_workspace_root == (workspace_dir / "notebooks").resolve()
    assert settings.resolved_data_dir == notebook_data_dir.resolve()


@pytest.mark.config
def test_artifact_store_can_activate_manifest(workspace_dir: Path) -> None:
    settings = AppSettings(workspace_root=workspace_dir, use_mock_bundle_if_missing=False)
    store = LocalArtifactStore(settings)
    bundle_dir = settings.resolved_bundle_root / "v1"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = BundleManifest(
        version="v1",
        created_at="2026-01-01T00:00:00+00:00",
        manifest_path=str(bundle_dir / "manifest.json"),
        bundle_dir=str(bundle_dir),
        runtime_bundle_path=str(bundle_dir / "runtime_bundle.pkl"),
        evaluation_summary_path=str(bundle_dir / "evaluation_summary.json"),
        run_name="pytest",
        run_profile="debug",
        model_backend="xgboost",
        retriever_variants=["content_based"],
    )
    with open(bundle_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2)

    pointer = store.activate_bundle("v1")
    active_manifest = store.read_active_manifest()

    assert pointer.version == "v1"
    assert active_manifest is not None
    assert active_manifest.version == "v1"


@pytest.mark.config
def test_artifact_store_can_load_legacy_pickle_bundle(workspace_dir: Path) -> None:
    settings = AppSettings(workspace_root=workspace_dir, use_mock_bundle_if_missing=False)
    store = LocalArtifactStore(settings)
    bundle_dir = settings.resolved_bundle_root / "legacy"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / "runtime_bundle.pkl"
    manifest_payload = {
        "version": "legacy",
        "created_at": "2026-01-01T00:00:00+00:00",
        "manifest_path": str(bundle_dir / "manifest.json"),
        "bundle_dir": str(bundle_dir),
        "runtime_bundle_path": str(bundle_path),
        "evaluation_summary_path": str(bundle_dir / "evaluation_summary.json"),
        "run_name": "pytest",
        "run_profile": "debug",
        "model_backend": "xgboost",
        "retriever_variants": ["content_based"],
    }
    with open(bundle_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)
    with open(bundle_path, "wb") as handle:
        pickle.dump(
            RuntimeBundle(
                manifest=BundleManifest.from_dict(manifest_payload),
                evaluation_summary=EvaluationSummary(source="legacy"),
            ),
            handle,
        )

    manifest = store.load_manifest("legacy")
    loaded = store.load_bundle(manifest)

    assert manifest.bundle_format == "pickle"
    assert loaded.manifest.version == "legacy"
    assert loaded.evaluation_summary.source == "legacy"


@pytest.mark.config
def test_settings_resolve_local_mlflow_defaults(workspace_dir: Path) -> None:
    settings = AppSettings(workspace_root=workspace_dir, mlflow_enabled=True)

    assert settings.mlflow.enabled is True
    assert settings.mlflow.backend_root == (workspace_dir / "mlflow_runs").resolve()
    assert settings.mlflow.tracking_uri.endswith("mlflow_runs")
    assert settings.safe_config()["mlflow"]["enabled"] is True


@pytest.mark.foundation
def test_build_container_wires_services(mock_settings: AppSettings) -> None:
    container = build_container(mock_settings)

    assert container.settings.app_name
    assert container.artifact_store is not None
    assert container.training_pipeline is not None
    assert container.bundle_export_service is not None
    assert container.recommendation_service.readiness().ready is True
