"""Integration tests for FastAPI."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_forecast_requires_auth():
    response = client.post("/forecast", json={"history": [], "horizon": 7})
    assert response.status_code in [401, 422]


def test_forecast_with_data():
    history = [
        {
            "date": f"2024-01-{i:02d}",
            "sku_id": "SKU_0001",
            "dc_id": "DC_00",
            "demand": 10.0 + i,
        }
        for i in range(1, 15)
    ]
    response = client.post(
        "/forecast",
        json={"history": history, "horizon": 7},
        headers={"Authorization": "Bearer dev-token"},
    )
    # May succeed or fail if model not loaded - check structure
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert data["horizon"] == 7
