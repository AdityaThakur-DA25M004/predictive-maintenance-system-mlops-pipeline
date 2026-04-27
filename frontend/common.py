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
    # ── In-container HTTP target for the API client ──────────────────────
    api_url: str

    # ── Browser-facing URLs (used in <a href> links and st.link_button) ──
    # These must resolve from the user's BROWSER, not from inside the
    # frontend container. So in Docker they point to the host's exposed
    # ports (http://localhost:PORT), NOT to Docker DNS service names.
    api_browser_url: str
    mlflow_url: str
    airflow_url: str
    grafana_url: str
    prometheus_url: str

    # ── Internal URLs used ONLY by _is_reachable() health checks ─────────
    # _is_reachable runs inside the frontend container, where Docker DNS
    # service names (mlflow:5000, grafana:3000, ...) work but localhost
    # would point at the frontend itself.
    mlflow_internal_url: str
    airflow_internal_url: str
    grafana_internal_url: str
    prometheus_internal_url: str

    request_timeout: float = 10.0


@st.cache_resource
def get_config() -> AppConfig:
    # Helper: fall back to the public URL if no internal override is provided
    # (works for local non-Docker dev where browser and process see the same hosts).
    def _internal(name: str, default: str) -> str:
        return os.environ.get(f"{name}_INTERNAL_URL",
                              os.environ.get(f"{name}_URL", default))

    return AppConfig(
        # In-container API calls must keep using the Docker service hostname
        api_url=os.environ.get("API_URL", "http://localhost:8000"),
        # Browser link to API docs — separate var so the container can keep
        # API_URL=http://api:8000 for in-container HTTP calls.
        api_browser_url=os.environ.get(
            "API_BROWSER_URL",
            os.environ.get("API_URL", "http://localhost:8000"),
        ),
        # Browser-facing URLs
        mlflow_url=os.environ.get("MLFLOW_URL", "http://localhost:5000"),
        airflow_url=os.environ.get("AIRFLOW_URL", "http://localhost:8080"),
        grafana_url=os.environ.get("GRAFANA_URL", "http://localhost:3000"),
        prometheus_url=os.environ.get("PROMETHEUS_URL", "http://localhost:9090"),
        # Internal URLs (Docker DNS) for reachability probes
        mlflow_internal_url=_internal("MLFLOW", "http://localhost:5000"),
        airflow_internal_url=_internal("AIRFLOW", "http://localhost:8080"),
        grafana_internal_url=_internal("GRAFANA", "http://localhost:3000"),
        prometheus_internal_url=_internal("PROMETHEUS", "http://localhost:9090"),
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
        # Only set Content-Type for JSON requests; let requests handle multipart
        headers = kwargs.pop("headers", {})
        try:
            resp = self._session.request(method, self._url(path), headers=headers, **kwargs)
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

    def feature_importance(self) -> dict:
        return self._request("GET", "/model/feature-importance")

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

    def retrain_with_upload(self, file_bytes: bytes, filename: str,
                            reason: str, api_key: str) -> dict:
        return self._request(
            "POST", "/retrain/upload",
            files={"file": (filename, file_bytes, "text/csv")},
            data={"reason": reason},
            headers={"X-API-Key": api_key},
            timeout=60,
        )

    def list_uploads(self) -> dict:
        return self._request("GET", "/retrain/uploads")


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


def _is_reachable(url: str, timeout: float = 1.5) -> bool:
    """Quick TCP check — returns True if the host:port is listening."""
    import socket
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        host = p.hostname or "localhost"
        port = p.port or (443 if p.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


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

        # External tools — show availability status
        # NOTE: reachability is checked via the *_internal_url (Docker DNS),
        # but the link href uses the *_url (browser-facing localhost:PORT).
        st.markdown("**External tools**")
        tools = [
            # (label, browser_url, internal_url_for_reachability_check)
            ("🔬 MLflow",     cfg.mlflow_url,                cfg.mlflow_internal_url),
            ("🌀 Airflow",    cfg.airflow_url,               cfg.airflow_internal_url),
            ("📊 Grafana",    cfg.grafana_url,               cfg.grafana_internal_url),
            ("🔢 Prometheus", cfg.prometheus_url,            cfg.prometheus_internal_url),
            ("📘 API docs",   f"{cfg.api_browser_url}/docs", f"{cfg.api_url}/docs"),
        ]
        for label, browser_url, internal_url in tools:
            reachable = _is_reachable(internal_url)
            if reachable:
                st.markdown(f"- [{label}]({browser_url})")
            else:
                st.markdown(
                    f"- {label} — "
                    f"<span style='color:#f97316;font-size:0.8em;'>not running locally</span>",
                    unsafe_allow_html=True,
                )