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
    users_response = client.get("/users")
    monitoring_response = client.get("/monitoring/drift/summary")
    favicon_response = client.get("/favicon.ico")
    recommend_response = client.post("/recommend", json={"history_items": ["A1", "A2"], "top_k": 3})
    page_response = client.get("/")

    assert health_response.status_code == 200
    assert ready_response.status_code == 200
    assert config_response.status_code == 200
    assert model_response.status_code == 200
    assert users_response.status_code == 200
    assert monitoring_response.status_code == 503
    assert favicon_response.status_code == 200
    assert recommend_response.status_code == 200
    assert page_response.status_code == 200

    payload = recommend_response.json()
    assert payload["source"] == "mock"
    assert len(payload["items"]) == 3
    assert users_response.json()["items"] == []
    assert "Amazon RecSys" in page_response.text
    assert "available-users" in page_response.text
    assert "data-local-shutdown" in page_response.text
    assert "/static/favicon.svg?v=" in page_response.text
    assert "/static/style.css?v=" in page_response.text
    assert "/static/app.js?v=" in page_response.text
    assert "data-enhanced-form" in page_response.text
    assert "Deep-dive workspace" in page_response.text
    assert 'data-tab-group="insights"' in page_response.text
    assert 'data-tab-target="monitoring"' in page_response.text
