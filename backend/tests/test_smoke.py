"""
Smoke tests for the PIDEM platform FastAPI backend.
Run from backend/ directory: pytest tests/test_smoke.py
"""

import sys
from pathlib import Path

# Ensure backend/ is on path when running from project root or backend/
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ─────────────────── Health ───────────────────

def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


# ─────────────────── M00–M18 ───────────────────

def test_m00_endpoint(client: TestClient):
    response = client.post(
        "/api/m00/train",
        json={"degree": 3, "train_split": 0.8},
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data
    assert "metrics" in data


def test_m01_endpoint(client: TestClient):
    response = client.post(
        "/api/m01/train",
        json={
            "features": ["price_gap"],
            "station_type": "all",
            "regularization": 1.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data
    assert "metrics" in data


def test_m02_endpoint(client: TestClient):
    response = client.post(
        "/api/m02/train",
        json={"model_type": "xgboost", "threshold": 0.15, "tree_depth": 4},
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data
    assert "metrics" in data


@pytest.mark.slow
def test_m03_endpoint(client: TestClient):
    response = client.post(
        "/api/m03/train",
        json={"n_estimators": 50, "learning_rate": 0.1, "shap_sample": 50},
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data
    assert "metrics" in data


def test_m04_endpoint(client: TestClient):
    response = client.post(
        "/api/m04/train",
        json={"k": 4},
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data
    assert "metrics" in data


def test_m05_endpoint(client: TestClient):
    response = client.post(
        "/api/m05/train",
        json={"contamination": 0.05, "window_size": 14},
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data
    assert "metrics" in data


@pytest.mark.slow
def test_m06_endpoint(client: TestClient):
    response = client.post(
        "/api/m06/train",
        json={
            "station_id": "STN_001",
            "method": "arima",
            "p": 2,
            "d": 1,
            "q": 2,
            "horizon": 14,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data


def test_m07_endpoint(client: TestClient):
    response = client.post(
        "/api/m07/train",
        json={"station_id": "STN_001", "lags": 7},
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data
    assert "metrics" in data


def test_m08_endpoint(client: TestClient):
    response = client.post(
        "/api/m08/train",
        json={"station_id": "STN_001"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data


@pytest.mark.slow
def test_m09_endpoint(client: TestClient):
    response = client.post(
        "/api/m09/solve",
        json={
            "price_band": [1.40, 1.80],
            "volume_floor": 0.8,
            "solver_mode": "lp",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data
    assert "metrics" in data


@pytest.mark.slow
def test_m10_endpoint(client: TestClient):
    response = client.post(
        "/api/m10/run",
        json={"algorithm": "thompson_sampling", "horizon": 100},
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data
    assert "metrics" in data


@pytest.mark.slow
def test_m11_endpoint(client: TestClient):
    response = client.post(
        "/api/m11/train",
        json={"gamma": 0.95, "episodes": 50},
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data
    assert "metrics" in data


@pytest.mark.slow
def test_m12_endpoint(client: TestClient):
    response = client.post(
        "/api/m12/train",
        json={"hidden_layers": 2, "units": 32, "replay_size": 500},
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data


@pytest.mark.slow
def test_m13_endpoint(client: TestClient):
    response = client.post(
        "/api/m13/train",
        json={"n_layers": 2, "units": 32},
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data


@pytest.mark.slow
def test_m14_endpoint(client: TestClient):
    response = client.post(
        "/api/m14/train",
        json={"n_heads": 2, "n_layers": 1},
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data


def test_m15_endpoint(client: TestClient):
    response = client.get("/api/m15/architectures")
    assert response.status_code == 200
    data = response.json()
    assert "architectures" in data


def test_m16_endpoint(client: TestClient):
    response = client.post(
        "/api/m16/analyze",
        json={"text": "diesel price 1.65"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data or "data" in data


def test_m17_endpoint(client: TestClient):
    response = client.post(
        "/api/m17/train",
        json={"chunk_size": 500, "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert "figures" in data


def test_m18_endpoint(client: TestClient):
    response = client.get("/api/m18/overview")
    assert response.status_code == 200
    data = response.json()
    assert "architecture" in data
