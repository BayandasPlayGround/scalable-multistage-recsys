from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from amazon_recsys.api.app import create_app
from amazon_recsys.config.container import build_container


@pytest.mark.serving
@pytest.mark.config
def test_local_dev_mode_uses_mock_bundle(mock_settings) -> None:
    client = TestClient(create_app(mock_settings))

    ready_response = client.get("/ready")
    recommend_response = client.post("/recommend", json={"history_items": ["A1", "A2"], "top_k": 2})

    assert ready_response.status_code == 200
    assert ready_response.json()["source"] == "mock"
    assert recommend_response.status_code == 200
    assert recommend_response.json()["source"] == "mock"


@pytest.mark.slow
@pytest.mark.serving
@pytest.mark.config
@pytest.mark.ranking
def test_production_mode_requires_real_bundle(production_settings) -> None:
    pre_client = TestClient(create_app(production_settings))
    pre_ready = pre_client.get("/ready")
    pre_shutdown = pre_client.post("/local/shutdown")

    assert pre_ready.status_code == 503
    assert pre_ready.json()["ready"] is False
    assert pre_shutdown.status_code == 403

    container = build_container(production_settings)
    session = container.training_pipeline.run(force_rebuild=True)
    manifest = container.artifact_store.save_bundle(session, version="prod-bundle")
    container.artifact_store.activate_bundle(manifest.version)

    post_client = TestClient(create_app(production_settings))
    post_ready = post_client.get("/ready")
    post_model = post_client.get("/models/active")
    post_recommend = post_client.post("/recommend", json={"user_id": "u1", "top_k": 2})

    assert post_ready.status_code == 200
    assert post_ready.json()["version"] == "prod-bundle"
    assert post_model.status_code == 200
    assert post_model.json()["source"] == "bundle"
    assert post_recommend.status_code == 200
