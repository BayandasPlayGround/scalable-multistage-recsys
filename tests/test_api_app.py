from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from amazon_recsys.api.app import create_app


@pytest.mark.foundation
@pytest.mark.serving
def test_api_endpoints_work_with_mock_bundle(mock_settings) -> None:
    client = TestClient(create_app(mock_settings))

    health_response = client.get("/health")
    ready_response = client.get("/ready")
    config_response = client.get("/config")
    model_response = client.get("/models/active")
    recommend_response = client.post("/recommend", json={"history_items": ["A1", "A2"], "top_k": 3})
    page_response = client.get("/")

    assert health_response.status_code == 200
    assert ready_response.status_code == 200
    assert config_response.status_code == 200
    assert model_response.status_code == 200
    assert recommend_response.status_code == 200
    assert page_response.status_code == 200

    payload = recommend_response.json()
    assert payload["source"] == "mock"
    assert len(payload["items"]) == 3
    assert "Amazon RecSys" in page_response.text
