"""
Shared utilities for the Predictive Maintenance Streamlit frontend.

Centralizes:
  - Environment/config reading
  - API client with consistent error handling
  - Reusable UI components (status pills, metric cards)
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Optional
import requests
import streamlit as st


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    api_url: str
    mlflow_url: str
    airflow_url: str
    grafana_url: str
    prometheus_url: str
    request_timeout: float = 10.0


@st.cache_resource
def get_config() -> AppConfig:
    return AppConfig(
        api_url=os.environ.get("API_URL", "http://localhost:8000"),
        mlflow_url=os.environ.get("MLFLOW_URL", "http://localhost:5000"),
        airflow_url=os.environ.get("AIRFLOW_URL", "http://localhost:8080"),
        grafana_url=os.environ.get("GRAFANA_URL", "http://localhost:3000"),
        prometheus_url=os.environ.get("PROMETHEUS_URL", "http://localhost:9090"),
    )


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
class APIError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class APIClient:
    """Thin wrapper around the FastAPI backend with consistent error handling."""

    def __init__(self, cfg: AppConfig):
        self._cfg = cfg
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self._cfg.api_url.rstrip('/')}{path}"

    def _request(self, method: str, path: str, **kwargs) -> Any:
        kwargs.setdefault("timeout", self._cfg.request_timeout)
        try:
            resp = self._session.request(method, self._url(path), **kwargs)
        except requests.ConnectionError:
            raise APIError(f"Cannot reach API at {self._cfg.api_url}")
        except requests.Timeout:
            raise APIError(f"Request to {path} timed out")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise APIError(f"{resp.status_code}: {detail}", resp.status_code)
        return resp.json() if resp.content else None

    def health(self) -> dict:
        return self._request("GET", "/health")

    def model_info(self) -> dict:
        return self._request("GET", "/model/info")

    def predict(self, reading: dict) -> dict:
        return self._request("POST", "/predict", json=reading)

    def predict_batch(self, readings: list[dict]) -> dict:
        return self._request("POST", "/predict/batch", json={"readings": readings}, timeout=120)

    def drift_check(self, readings: list[dict]) -> dict:
        return self._request("POST", "/drift/check", json={"readings": readings}, timeout=60)

    def submit_feedback(self, prediction_id: int, actual_label: int) -> dict:
        return self._request("POST", "/feedback",
                             json={"prediction_id": prediction_id, "actual_label": actual_label})

    def feedback_stats(self) -> dict:
        return self._request("GET", "/feedback/stats")

    def trigger_retrain(self, reason: str, api_key: str) -> dict:
        return self._request("POST", "/retrain",
                             params={"reason": reason}, headers={"X-API-Key": api_key})


@st.cache_resource
def get_client() -> APIClient:
    return APIClient(get_config())


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
_CSS = """
<style>
.main { padding-top: 1rem; }
.block-container { padding-top: 1.2rem; max-width: 1400px; }
.pm-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #0ea5e9 100%);
    color: white; padding: 1.4rem 1.6rem; border-radius: 12px; margin-bottom: 1.4rem;
}
.pm-header h1 { color: white; margin: 0; font-size: 1.7rem; }
.pm-header p  { color: #e0f2fe; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
"""


def setup_page(title: str, icon: str = "🏭") -> None:
    st.set_page_config(
        page_title=f"{title} — Predictive Maintenance",
        page_icon=icon, layout="wide", initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="pm-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    cfg = get_config()
    client = get_client()
    with st.sidebar:
        st.markdown("### 🏭 Predictive Maintenance")
        st.caption("Machine failure prediction & monitoring")
        st.divider()
        try:
            h = client.health()
            text = "🟢 API online" if h.get("model_loaded") else "🟡 Model not loaded"
            st.markdown(text)
        except APIError:
            st.markdown("🔴 API offline")
        st.divider()
        st.markdown("**External tools**")
        st.markdown(f"- [🔬 MLflow]({cfg.mlflow_url})")
        st.markdown(f"- [🌀 Airflow]({cfg.airflow_url})")
        st.markdown(f"- [📊 Grafana]({cfg.grafana_url})")
        st.markdown(f"- [🔢 Prometheus]({cfg.prometheus_url})")
        st.markdown(f"- [📘 API docs]({cfg.api_url}/docs)")
