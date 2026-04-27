"""
Integration tests for the FastAPI application.
Covers health, predict, batch, feedback loop, drift, retrain auth.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from api.main import app, _state, _load_artifacts, RETRAIN_API_KEY


@pytest.fixture(scope="module")
def client():
    """Load artifacts once for all tests in this module."""
    if _state["model"] is None:
        _load_artifacts()
    return TestClient(app)


@pytest.fixture
def sample_reading():
    return {
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100,
        "product_type": "L",
    }


# ── System endpoints   
class TestHealthEndpoints:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "scaler_loaded" in data
        assert "uptime_seconds" in data

    def test_ready_with_model(self, client):
        if _state["model"] is None:
            pytest.skip("Model not loaded")
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_ready_without_model(self, client):
        original = _state["model"]
        _state["model"] = None
        resp = client.get("/ready")
        _state["model"] = original
        assert resp.status_code == 503

    def test_model_info(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_loaded" in data
        assert "model_version" in data


# ── Prediction endpoints 
class TestPredictEndpoint:
    def test_predict_valid_input(self, client, sample_reading):
        if _state["model"] is None:
            pytest.skip("Model not loaded")
        resp = client.post("/predict", json=sample_reading)
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["failure_probability"] <= 1.0
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert "prediction_id" in data
        assert isinstance(data["prediction_id"], int)

    def test_predict_invalid_product_type(self, client, sample_reading):
        sample_reading["product_type"] = "X"
        resp = client.post("/predict", json=sample_reading)
        assert resp.status_code == 422

    def test_predict_missing_fields(self, client):
        resp = client.post("/predict", json={"air_temperature": 300.0})
        assert resp.status_code == 422

    def test_predict_out_of_range_temp(self, client, sample_reading):
        sample_reading["air_temperature"] = 999.0
        resp = client.post("/predict", json=sample_reading)
        assert resp.status_code == 422


class TestBatchEndpoint:
    def test_batch_predict(self, client, sample_reading):
        if _state["model"] is None:
            pytest.skip("Model not loaded")
        payload = {"readings": [sample_reading, sample_reading]}
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2
        assert isinstance(data["failures_detected"], int)

    def test_batch_empty(self, client):
        resp = client.post("/predict/batch", json={"readings": []})
        assert resp.status_code == 422


# ── Feedback loop endpoints 
class TestFeedbackEndpoints:
    def test_feedback_submit_valid(self, client, sample_reading):
        if _state["model"] is None:
            pytest.skip("Model not loaded")
        # First make a prediction to get a prediction_id
        pred_resp = client.post("/predict", json=sample_reading)
        assert pred_resp.status_code == 200
        pid = pred_resp.json()["prediction_id"]

        # Submit feedback
        fb_resp = client.post("/feedback", json={
            "prediction_id": pid,
            "actual_label": 0,
        })
        assert fb_resp.status_code == 200
        data = fb_resp.json()
        assert data["status"] == "recorded"
        assert data["prediction_id"] == pid
        assert isinstance(data["correct"], bool)

    def test_feedback_unknown_prediction_id(self, client):
        resp = client.post("/feedback", json={
            "prediction_id": 999999,
            "actual_label": 1,
        })
        assert resp.status_code == 404

    def test_feedback_stats(self, client):
        resp = client.get("/feedback/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_feedback" in data
        assert "window" in data
        assert isinstance(data["total_feedback"], int)


# ── Retrain endpoint (auth) 
class TestRetrainEndpoint:
    def test_retrain_without_api_key(self, client):
        resp = client.post("/retrain", params={"reason": "manual"})
        assert resp.status_code == 401

    def test_retrain_with_wrong_key(self, client):
        resp = client.post(
            "/retrain",
            params={"reason": "manual"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_retrain_with_valid_key(self, client):
        resp = client.post(
            "/retrain",
            params={"reason": "manual"},
            headers={"X-API-Key": RETRAIN_API_KEY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("triggered", "partial"), (
            f"Unexpected /retrain status: {data['status']} "
            f"(expected 'triggered' or 'partial'); full response: {data}"
        )
        assert data["triggered_by"] == "manual"


# ── Metrics endpoint 
class TestMetricsEndpoint:
    def test_prometheus_metrics(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.text
        assert "prediction_requests_total" in body
        assert "prediction_latency_seconds" in body

    def test_drift_check_empty(self, client):
        resp = client.post("/drift/check", json={"readings": []})
        assert resp.status_code == 422
