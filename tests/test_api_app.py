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
    query_page_response = client.get("/?history_items=A1,A2&top_k=3#qa-workspace")

    assert health_response.status_code == 200
    assert ready_response.status_code == 200
    assert config_response.status_code == 200
    assert model_response.status_code == 200
    assert users_response.status_code == 200
    assert monitoring_response.status_code == 503
    assert favicon_response.status_code == 200
    assert recommend_response.status_code == 200
    assert page_response.status_code == 200
    assert query_page_response.status_code == 200

    payload = recommend_response.json()
    assert payload["source"] == "mock"
    assert len(payload["items"]) == 3
    assert users_response.json()["items"] == []
    assert "Amazon RecSys Analyst Console" in page_response.text
    assert "Mock bundle is active." in page_response.text
    assert "available-users" in page_response.text
    assert "data-local-shutdown" in page_response.text
    assert 'aria-current="page"' in page_response.text
    assert 'data-sort-key="rank"' in query_page_response.text
    assert 'data-resizable-table="qa-results"' in query_page_response.text
    assert "R 10.00" in query_page_response.text
    assert "/static/favicon.svg?v=" in page_response.text
    assert "/static/style.css?v=" in page_response.text
    assert "/static/app.js?v=" in page_response.text
    assert "2026-05-03-viewport-toggle-1" in page_response.text
    assert 'name="color-scheme" content="light dark"' in page_response.text
    assert "amazon-recsys-theme" in page_response.text
    assert "data-theme-toggle" in page_response.text
    assert "amazon-recsys-viewport-mode" in page_response.text
    assert "data-viewport-mode-switch" in page_response.text
    assert 'data-viewport-mode-option="desktop"' in page_response.text
    assert 'data-viewport-mode-option="mobile"' in page_response.text
    assert "data-enhanced-form" in page_response.text
    assert "Analyst Workflow Modes" in page_response.text
    assert 'data-tab-group="workspace"' in page_response.text
    assert 'data-tab-target="monitoring-workspace"' in page_response.text
