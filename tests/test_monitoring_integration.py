from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from amazon_recsys.api.app import create_app
from amazon_recsys.config.container import build_container


def _monitoring_settings(test_settings, **updates):
    return test_settings.model_copy(
        update={
            "monitoring_enabled": True,
            "monitoring_min_events_per_window": 1,
            "monitoring_psi_warn": 10.0,
            "monitoring_psi_alert": 20.0,
            "monitoring_js_warn": 10.0,
            "monitoring_js_alert": 20.0,
            **updates,
        }
    )


@pytest.mark.config
@pytest.mark.data
@pytest.mark.serving
def test_monitoring_summary_is_persisted_and_exposed_via_api(test_settings) -> None:
    settings = _monitoring_settings(test_settings)
    settings.ensure_runtime_directories()
    container = build_container(settings)

    session = container.training_pipeline.run(force_rebuild=True)
    manifest = container.artifact_store.save_bundle(session, version="monitoring-fixture")
    container.artifact_store.activate_bundle(manifest.version)
    container.recommendation_service.refresh()

    for user_id in ("u1", "u2"):
        items = container.recommendation_service.recommend(user_id=user_id, top_k=5)
        assert items

    inference_frame = container.monitoring_store.load_inference_frame(bundle_version=manifest.version)
    assert not inference_frame.empty
    requested_at = pd.to_datetime(inference_frame["requested_at"], utc=True)
    summary = container.monitoring_service.run_monitoring(
        window_start=(requested_at.min() - pd.Timedelta(minutes=1)).isoformat(),
        window_end=(requested_at.max() + pd.Timedelta(minutes=2)).isoformat(),
        bundle_version=manifest.version,
        simulate_outcomes=True,
    )

    assert summary.bundle_version == manifest.version
    assert summary.inference_count == int(inference_frame["request_id"].nunique())
    assert summary.outcome_count == int(inference_frame["request_id"].nunique())
    assert summary.concept_drift.sample_size == int(inference_frame["request_id"].nunique())
    assert Path(manifest.notes["reference_profile_path"]).exists()

    latest = container.monitoring_service.latest_summary(manifest.version)
    assert latest is not None
    assert latest.window_end == summary.window_end

    client = TestClient(create_app(settings))
    response = client.get("/monitoring/drift/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["bundle_version"] == manifest.version
    assert payload["summary"]["window_end"] == summary.window_end

    dashboard_response = client.get("/")
    assert dashboard_response.status_code == 200
    assert "Latest monitoring trajectory" in dashboard_response.text
    assert "sparkline-chart" in dashboard_response.text


@pytest.mark.config
@pytest.mark.data
@pytest.mark.serving
def test_monitor_backfill_can_auto_simulate_recent_outcomes(test_settings) -> None:
    settings = _monitoring_settings(test_settings)
    settings.ensure_runtime_directories()
    container = build_container(settings)

    session = container.training_pipeline.run(force_rebuild=True)
    manifest = container.artifact_store.save_bundle(session, version="monitoring-backfill-fixture")
    container.artifact_store.activate_bundle(manifest.version)
    container.recommendation_service.refresh()

    container.recommendation_service.recommend(user_id="u1", top_k=5)
    summaries = container.monitoring_service.monitor_backfill(
        days=1,
        bundle_version=manifest.version,
        simulate_outcomes=True,
    )

    assert summaries
    latest = summaries[-1]
    assert latest.bundle_version == manifest.version
    assert latest.inference_count >= 1
    assert latest.outcome_count >= 1


@pytest.mark.slow
@pytest.mark.config
@pytest.mark.data
@pytest.mark.serving
def test_monitoring_run_is_logged_to_mlflow(test_settings) -> None:
    mlflow = pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    tracking_dir = Path.cwd() / f"mlflow-monitoring-{uuid4().hex}"
    settings = _monitoring_settings(
        test_settings,
        mlflow_enabled=True,
        mlflow_experiment_name="amazon-recsys-monitoring-pytest",
        mlflow_backend_root=tracking_dir,
    )
    try:
        settings.ensure_runtime_directories()
        container = build_container(settings)

        session = container.training_pipeline.run(force_rebuild=True)
        manifest = container.artifact_store.save_bundle(session, version="monitoring-mlflow-fixture")
        container.artifact_store.activate_bundle(manifest.version)
        container.recommendation_service.refresh()
        container.recommendation_service.recommend(user_id="u1", top_k=5)

        inference_frame = container.monitoring_store.load_inference_frame(bundle_version=manifest.version)
        requested_at = pd.to_datetime(inference_frame["requested_at"], utc=True)
        simulate_payload = container.monitoring_service.simulate_outcomes(
            bundle_version=manifest.version,
            window_start=(requested_at.min() - pd.Timedelta(minutes=1)).isoformat(),
            window_end=(requested_at.max() + pd.Timedelta(minutes=2)).isoformat(),
        )
        assert simulate_payload["ingested"] == 1

        summary = container.monitoring_service.run_monitoring(
            window_start=(requested_at.min() - pd.Timedelta(minutes=1)).isoformat(),
            window_end=(requested_at.max() + pd.Timedelta(minutes=2)).isoformat(),
            bundle_version=manifest.version,
        )

        assert summary.mlflow["experiment_name"] == "amazon-recsys-monitoring-pytest-monitoring"

        client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
        run = client.get_run(summary.mlflow["run_id"])
        root_artifacts = {item.path for item in client.list_artifacts(summary.mlflow["run_id"])}

        assert run.data.tags["phase"] == "monitoring"
        assert run.data.tags["bundle_version"] == manifest.version
        assert "monitoring" in root_artifacts
    finally:
        shutil.rmtree(tracking_dir, ignore_errors=True)
