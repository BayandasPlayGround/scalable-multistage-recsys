from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from amazon_recsys.api.app import create_app


@pytest.mark.slow
@pytest.mark.data
@pytest.mark.retrieval
@pytest.mark.ranking
@pytest.mark.serving
def test_training_bundle_round_trip(test_container) -> None:
    session = test_container.training_pipeline.run(force_rebuild=True)
    manifest = test_container.artifact_store.save_bundle(session, version="fixture-bundle")
    pointer = test_container.artifact_store.activate_bundle(manifest.version)
    test_container.recommendation_service.refresh()

    recommendations = test_container.recommendation_service.recommend(user_id="u1", top_k=3)
    history = test_container.recommendation_service.get_user_history("u1")
    model = test_container.recommendation_service.get_active_model()
    summary = test_container.recommendation_service.get_evaluation_summary()

    assert pointer.version == "fixture-bundle"
    assert session.pipeline_config.__class__.__module__ == "amazon_recsys.ml.core"
    assert session.prepared.__class__.__module__ == "amazon_recsys.ml.core"
    assert 1 <= len(recommendations) <= 3
    assert history
    assert model["version"] == "fixture-bundle"
    assert summary["metric_files"]

    client = TestClient(create_app(test_container.settings))
    api_response = client.post("/recommend", json={"user_id": "u1", "top_k": 2})
    active_response = client.get("/models/active")

    assert api_response.status_code == 200
    assert active_response.status_code == 200
    assert active_response.json()["version"] == "fixture-bundle"
