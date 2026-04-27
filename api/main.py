"""
FastAPI Backend for the Predictive Maintenance System.

Provides REST API endpoints for:
  - Health & readiness checks
  - Single and batch predictions
  - Data drift detection
  - Model retraining triggers (authenticated)
  - Ground-truth feedback logging (feedback loop for real-world performance decay)
  - Prometheus metrics export
"""

import os
import sys
import io
import time
import sqlite3
import secrets
import base64
import uuid
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
import joblib
import requests

from fastapi import FastAPI, HTTPException, Depends, Header, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (
    SensorInput, PredictionResponse, BatchInput, BatchResponse,
    HealthResponse, DriftReport, RetrainResponse,
    FeedbackInput, FeedbackResponse, FeedbackStats,
    UploadRetrainResponse, UploadListResponse, UploadedFile,
    RollbackRequest, RollbackResponse,
)
from api.metrics import (
    PREDICTION_COUNT, PREDICTION_LATENCY, ERROR_COUNT,
    FAILURE_PREDICTIONS, FAILURE_PROBABILITY, MODEL_ACCURACY,
    MODEL_F1_SCORE, DRIFT_DETECTED, DRIFT_FEATURES_COUNT,
    DRIFT_CHECK_COUNT, MODEL_INFO, RETRAIN_TRIGGERS, ACTIVE_REQUESTS,
    FEEDBACK_COUNT, FEEDBACK_ROLLING_ACCURACY,
    UPLOAD_COUNT, UPLOAD_ROWS, UPLOAD_FILE_SIZE_BYTES,
    RETRAIN_DATA_SOURCE, LAST_TRAINING_TIMESTAMP,
    MODEL_VERSION_NUMERIC, ROLLBACK_TRIGGERS,
)
from src.data_preprocessing import engineer_features, get_feature_columns
from src.drift_detection import detect_drift
from src.utils import setup_logger, load_json

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RETRAIN_API_KEY = os.environ.get("RETRAIN_API_KEY") or secrets.token_urlsafe(32)
FEEDBACK_DB_PATH = os.environ.get(
    "FEEDBACK_DB_PATH",
    str(PROJECT_ROOT / "data" / "feedback" / "feedback.db"),
)
FEEDBACK_WINDOW = int(os.environ.get("FEEDBACK_WINDOW", "100"))
CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:8501,http://localhost:3000,http://frontend:8501",
).split(",")
UPLOADS_DIR = os.environ.get(
    "UPLOADS_DIR",
    str(PROJECT_ROOT / "data" / "feedback" / "uploads"),
)

# ---------------------------------------------------------------------------
# Airflow REST API config (used by /retrain to actually start the DAG run
# instead of only emitting an alert + Prometheus counter)
# ---------------------------------------------------------------------------
AIRFLOW_API_URL = os.environ.get("AIRFLOW_API_URL", "http://airflow-webserver:8080")
AIRFLOW_USERNAME = os.environ.get("AIRFLOW_USERNAME", "")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "")
AIRFLOW_DAG_ID = os.environ.get("AIRFLOW_DAG_ID", "predictive_maintenance_pipeline")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_state = {
    "model": None, "scaler": None, "baselines": None,
    "ref_samples": None, "baselines_loaded_at": None,
    "start_time": None, "model_version": "unknown",
    "model_source": "unloaded", "prediction_buffer": [],
    # Track recent upload content hashes so duplicate uploads of the same CSV
    # don't re-fire the retrain email. Keyed by md5 of CSV bytes, value is
    # unix timestamp of when we last alerted for that content.
    "recent_upload_alerts": {},
    # Track last-sent timestamp per `reason` for the manual /retrain endpoint
    # so accidental double-clicks don't double-email.
    "last_manual_retrain_alert": {},
}

# How long to suppress duplicate retrain alerts for the same uploaded CSV.
# 1 hour absorbs accidental re-uploads (double-click, browser retry, user
# verifying the file landed). Long-tail re-uploads after this window are
# presumed intentional and DO get a fresh alert.
RECENT_UPLOAD_TTL = 3600  # seconds

# Manual /retrain throttle: don't email twice for the same reason within this
# window. Different reasons reset the timer (so "manual" → "drift_detected"
# would still send both alerts).
RETRAIN_ALERT_THROTTLE_SECONDS = 300  # 5 minutes


def _is_duplicate_upload(csv_bytes: bytes) -> bool:
    """
    Return True if this exact CSV content was uploaded within the last
    RECENT_UPLOAD_TTL seconds. Used to suppress duplicate retrain emails.

    Side effects:
      - Garbage-collects entries older than the TTL
      - Records the current upload's hash with the current timestamp
    """
    digest = hashlib.md5(csv_bytes).hexdigest()
    now = time.time()
    recent = _state["recent_upload_alerts"]

    # GC old entries so the dict doesn't grow unbounded
    expired = [h for h, ts in recent.items() if now - ts > RECENT_UPLOAD_TTL]
    for h in expired:
        del recent[h]

    if digest in recent:
        return True  # Duplicate within TTL

    recent[digest] = now
    return False


# ---------------------------------------------------------------------------
# Feedback store (SQLite)
# ---------------------------------------------------------------------------
def _init_feedback_db() -> None:
    Path(FEEDBACK_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                air_temperature REAL, process_temperature REAL,
                rotational_speed INTEGER, torque REAL, tool_wear INTEGER,
                product_type TEXT, predicted_label INTEGER,
                predicted_proba REAL, model_version TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                actual_label INTEGER NOT NULL,
                correct INTEGER NOT NULL,
                FOREIGN KEY(prediction_id) REFERENCES predictions(prediction_id)
            )
        """)
        conn.commit()
    logger.info(f"Feedback DB initialized at {FEEDBACK_DB_PATH}")


def _log_prediction(reading: SensorInput, pred: int, proba: float) -> int:
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (timestamp, air_temperature, process_temperature, rotational_speed,
                torque, tool_wear, product_type, predicted_label, predicted_proba, model_version)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), reading.air_temperature, reading.process_temperature,
             reading.rotational_speed, reading.torque, reading.tool_wear,
             reading.product_type, pred, proba, _state["model_version"]),
        )
        conn.commit()
        return cur.lastrowid


def _update_feedback_accuracy() -> Optional[float]:
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        row = conn.execute(
            "SELECT AVG(correct) FROM (SELECT correct FROM feedback ORDER BY feedback_id DESC LIMIT ?)",
            (FEEDBACK_WINDOW,),
        ).fetchone()
    if row is None or row[0] is None:
        return None
    acc = float(row[0])
    FEEDBACK_ROLLING_ACCURACY.set(acc)
    return acc


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
def _load_artifacts() -> None:
    models_dir = PROJECT_ROOT / "models"
    data_dir = PROJECT_ROOT / "data" / "baselines"

    model_path = models_dir / "best_model.joblib"
    if model_path.exists():
        _state["model"] = joblib.load(model_path)
        _state["model_source"] = "local-joblib"
        _state["model_version"] = "1.0"
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.warning(f"Model not found at {model_path}")

    scaler_path = models_dir / "scaler.joblib"
    if scaler_path.exists():
        _state["scaler"] = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")

    baselines_path = data_dir / "drift_baselines.json"
    if baselines_path.exists():
        _state["baselines"] = load_json(str(baselines_path))
        _state["baselines_loaded_at"] = baselines_path.stat().st_mtime
        logger.info(f"Baselines loaded from {baselines_path}")

    # Reference samples for drift KS / PSI tests (real, not synthetic Gaussian)
    ref_samples_path = data_dir / "ref_samples.json"
    if ref_samples_path.exists():
        try:
            _state["ref_samples"] = load_json(str(ref_samples_path))
            logger.info(f"Reference samples loaded from {ref_samples_path}")
        except Exception as e:
            logger.warning(f"Failed to load ref_samples ({ref_samples_path}): {e}")
            _state["ref_samples"] = None
    else:
        logger.info(
            f"No ref_samples.json at {ref_samples_path} — drift will fall back "
            "to synthetic Gaussian until preprocessing pipeline runs"
        )

    metrics_path = models_dir / "test_metrics.json"
    if metrics_path.exists():
        metrics = load_json(str(metrics_path))
        MODEL_ACCURACY.set(metrics.get("accuracy", 0))
        MODEL_F1_SCORE.set(metrics.get("f1_score", 0))
        MODEL_INFO.info({
            "version": _state["model_version"],
            "source": _state["model_source"],
            "algorithm": "RandomForest",
            "f1_score": str(metrics.get("f1_score", 0)),
        })
        # Emit training timestamp based on metrics file mtime
        LAST_TRAINING_TIMESTAMP.set(metrics_path.stat().st_mtime)
        try:
            MODEL_VERSION_NUMERIC.set(float(_state["model_version"]))
        except (ValueError, TypeError):
            MODEL_VERSION_NUMERIC.set(0)

    _init_feedback_db()
    _state["start_time"] = time.time()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
async def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not x_api_key or not secrets.compare_digest(x_api_key, RETRAIN_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    if not os.environ.get("RETRAIN_API_KEY"):
        logger.warning("RETRAIN_API_KEY not set — generated random dev key: %s", RETRAIN_API_KEY)
    logger.info("Predictive Maintenance API started")
    yield
    logger.info("Predictive Maintenance API shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Predictive Maintenance API",
    description="Machine failure prediction with drift detection and feedback loop.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _trigger_airflow_dag(reason: str = "manual") -> dict:
    """
    Start a DagRun for the predictive_maintenance_pipeline via Airflow's
    stable REST API. Returns a dict with the trigger result.

    Without this, the /retrain endpoint only increments a Prometheus counter
    and sends an alert — no training actually happens until someone opens
    the Airflow UI manually. Calling this from the API closes the loop:
    click in Streamlit → DAG actually runs → metrics update.

    Non-fatal: if Airflow is unreachable or returns an error, we log and
    return status=skipped/failed but don't 500 the API request — the alert
    + Prometheus counter still fired upstream.
    """
    if not (AIRFLOW_USERNAME and AIRFLOW_PASSWORD):
        logger.warning(
            "AIRFLOW_USERNAME / AIRFLOW_PASSWORD not set — DAG trigger skipped"
        )
        return {"status": "skipped", "detail": "airflow_credentials_not_set"}

    auth_str = f"{AIRFLOW_USERNAME}:{AIRFLOW_PASSWORD}".encode("utf-8")
    headers = {
        "Authorization": "Basic " + base64.b64encode(auth_str).decode("ascii"),
        "Content-Type": "application/json",
    }
    # Unique dag_run_id per trigger so two clicks in the same minute don't collide
    dag_run_id = f"api_{reason}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    payload = {
        "dag_run_id": dag_run_id,
        "conf": {"triggered_by": "api", "reason": reason},
    }
    url = f"{AIRFLOW_API_URL}/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns"

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        if r.status_code in (200, 201):
            logger.info(f"Airflow DAG triggered: {dag_run_id}")
            return {"status": "triggered", "dag_run_id": dag_run_id}
        logger.warning(
            f"Airflow DAG trigger returned {r.status_code}: {r.text[:200]}"
        )
        return {
            "status": "failed",
            "http_status": r.status_code,
            "detail": r.text[:200],
        }
    except Exception as e:
        logger.warning(f"Airflow DAG trigger failed (non-fatal): {e}")
        return {"status": "failed", "detail": str(e)}


def _build_features(reading: SensorInput) -> pd.DataFrame:
    raw = pd.DataFrame([{
        "Air temperature [K]": reading.air_temperature,
        "Process temperature [K]": reading.process_temperature,
        "Rotational speed [rpm]": reading.rotational_speed,
        "Torque [Nm]": reading.torque,
        "Tool wear [min]": reading.tool_wear,
        "Type": reading.product_type,
    }])
    featured = engineer_features(raw)
    feature_cols = get_feature_columns()
    X = featured[feature_cols]
    if _state["scaler"] is not None:
        X = pd.DataFrame(_state["scaler"].transform(X), columns=feature_cols)
    return X


def _risk_level(prob: float) -> str:
    if prob < 0.2:   return "LOW"
    elif prob < 0.5: return "MEDIUM"
    elif prob < 0.8: return "HIGH"
    return "CRITICAL"


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=_state["model"] is not None,
        scaler_loaded=_state["scaler"] is not None,
        uptime_seconds=time.time() - (_state["start_time"] or time.time()),
    )


@app.get("/ready", tags=["System"])
async def readiness_check():
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if _state["scaler"] is None:
        raise HTTPException(status_code=503, detail="Scaler not loaded")
    return {"status": "ready"}


@app.get("/model/info", tags=["System"])
async def model_info():
    metrics_path = PROJECT_ROOT / "models" / "test_metrics.json"
    report_path = PROJECT_ROOT / "models" / "classification_report.json"
    info = {
        "model_loaded": _state["model"] is not None,
        "model_version": _state["model_version"],
        "model_source": _state["model_source"],
        "scaler_loaded": _state["scaler"] is not None,
        "baselines_loaded": _state["baselines"] is not None,
    }
    if metrics_path.exists():
        info["test_metrics"] = load_json(str(metrics_path))
    if report_path.exists():
        info["classification_report"] = load_json(str(report_path))
    return info


@app.get("/model/feature-importance", tags=["System"])
async def feature_importance():
    """Return feature importances from the loaded RandomForest model."""
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    from src.data_preprocessing import get_feature_columns
    cols = get_feature_columns()
    importances = _state["model"].feature_importances_.tolist()
    paired = sorted(
        zip(cols, importances), key=lambda x: x[1], reverse=True
    )
    return {
        "feature_importance": {k: round(v, 6) for k, v in paired},
        "top_feature": paired[0][0] if paired else None,
        "model_version": _state["model_version"],
    }


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(reading: SensorInput):
    ACTIVE_REQUESTS.inc()
    start = time.time()
    try:
        if _state["model"] is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        X = _build_features(reading)
        infer_start = time.time()
        prediction = int(_state["model"].predict(X)[0])
        proba = float(_state["model"].predict_proba(X)[0][1])
        inference_ms = round((time.time() - infer_start) * 1000, 3)
        risk = _risk_level(proba)

        # Persist for feedback loop
        prediction_id = _log_prediction(reading, prediction, proba)

        # Buffer for drift detection
        _state["prediction_buffer"].append(reading.model_dump())
        if len(_state["prediction_buffer"]) > 1000:
            _state["prediction_buffer"] = _state["prediction_buffer"][-500:]

        PREDICTION_COUNT.labels(endpoint="/predict", status="success").inc()
        FAILURE_PREDICTIONS.labels(risk_level=risk).inc()
        FAILURE_PROBABILITY.observe(proba)
        PREDICTION_LATENCY.labels(endpoint="/predict").observe(time.time() - start)

        return PredictionResponse(
            prediction=prediction,
            failure_probability=round(proba, 4),
            risk_level=risk,
            model_version=_state["model_version"],
            prediction_id=prediction_id,
            inference_time_ms=inference_ms,
        )
    except HTTPException:
        PREDICTION_COUNT.labels(endpoint="/predict", status="error").inc()
        raise
    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        PREDICTION_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(batch: BatchInput):
    ACTIVE_REQUESTS.inc()
    start = time.time()
    try:
        if _state["model"] is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        if not batch.readings:
            raise HTTPException(status_code=422, detail="Empty batch")

        predictions = []
        for reading in batch.readings:
            X = _build_features(reading)
            pred = int(_state["model"].predict(X)[0])
            proba = float(_state["model"].predict_proba(X)[0][1])
            risk = _risk_level(proba)
            pid = _log_prediction(reading, pred, proba)
            predictions.append(PredictionResponse(
                prediction=pred, failure_probability=round(proba, 4),
                risk_level=risk, model_version=_state["model_version"],
                prediction_id=pid,
            ))
            FAILURE_PREDICTIONS.labels(risk_level=risk).inc()
            FAILURE_PROBABILITY.observe(proba)

        failures = sum(1 for p in predictions if p.prediction == 1)
        PREDICTION_COUNT.labels(endpoint="/predict/batch", status="success").inc()
        PREDICTION_LATENCY.labels(endpoint="/predict/batch").observe(time.time() - start)

        return BatchResponse(predictions=predictions, total=len(predictions), failures_detected=failures)
    except HTTPException:
        PREDICTION_COUNT.labels(endpoint="/predict/batch", status="error").inc()
        raise
    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        PREDICTION_COUNT.labels(endpoint="/predict/batch", status="error").inc()
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()


# ---------------------------------------------------------------------------
# Feedback (ground-truth) loop — REQUIRED by MLOps guideline §II.E.2
# ---------------------------------------------------------------------------
@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(feedback: FeedbackInput):
    """Record the ground-truth label for a previous prediction."""
    try:
        with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
            row = conn.execute(
                "SELECT predicted_label FROM predictions WHERE prediction_id = ?",
                (feedback.prediction_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"prediction_id {feedback.prediction_id} not found")
            predicted = int(row[0])
            correct = int(predicted == feedback.actual_label)
            conn.execute(
                "INSERT INTO feedback (prediction_id, timestamp, actual_label, correct) VALUES (?,?,?,?)",
                (feedback.prediction_id, time.time(), feedback.actual_label, correct),
            )
            conn.commit()

        FEEDBACK_COUNT.labels(outcome="correct" if correct else "incorrect").inc()
        rolling = _update_feedback_accuracy()

        # Fire accuracy degradation alert when rolling accuracy drops below threshold
        if rolling is not None and rolling < float(os.environ.get("ACCURACY_ALERT_THRESHOLD", "0.7")):
            try:
                from src.alert_notifier import send_accuracy_alert
                send_accuracy_alert(rolling, window=FEEDBACK_WINDOW)
            except Exception as _ae:
                logger.warning(f"Accuracy alert failed (non-fatal): {_ae}")

        return FeedbackResponse(
            status="recorded", prediction_id=feedback.prediction_id,
            correct=bool(correct), rolling_accuracy=rolling,
        )
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        logger.error(f"Feedback error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feedback/stats", response_model=FeedbackStats, tags=["Feedback"])
async def feedback_stats():
    """Return aggregate feedback statistics."""
    with sqlite3.connect(FEEDBACK_DB_PATH) as conn:
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        if total == 0:
            return FeedbackStats(total_feedback=0, overall_accuracy=None, rolling_accuracy=None, window=FEEDBACK_WINDOW)
        overall = conn.execute("SELECT AVG(correct) FROM feedback").fetchone()[0]
        rolling_row = conn.execute(
            "SELECT AVG(correct) FROM (SELECT correct FROM feedback ORDER BY feedback_id DESC LIMIT ?)",
            (FEEDBACK_WINDOW,),
        ).fetchone()
        rolling = rolling_row[0] if rolling_row else None

    return FeedbackStats(
        total_feedback=int(total),
        overall_accuracy=float(overall) if overall is not None else None,
        rolling_accuracy=float(rolling) if rolling is not None else None,
        window=FEEDBACK_WINDOW,
    )


# ---------------------------------------------------------------------------
# Drift endpoint
# ---------------------------------------------------------------------------
@app.post("/drift/check", response_model=DriftReport, tags=["Monitoring"])
async def check_drift(batch: BatchInput):
    DRIFT_CHECK_COUNT.inc()
    if _state["baselines"] is None:
        raise HTTPException(status_code=503, detail="Baselines not loaded")
    if not batch.readings:
        raise HTTPException(status_code=422, detail="Empty batch")

    rows = [{
        "Air temperature [K]": r.air_temperature,
        "Process temperature [K]": r.process_temperature,
        "Rotational speed [rpm]": r.rotational_speed,
        "Torque [Nm]": r.torque,
        "Tool wear [min]": r.tool_wear,
        "Type": r.product_type,
    } for r in batch.readings]

    df = engineer_features(pd.DataFrame(rows))
    feature_cols = get_feature_columns()
    # Pass ref_samples (real training samples) so KS test isn't comparing
    # against a synthetic Gaussian. baselines_path is a fallback so detect_drift
    # can auto-discover ref_samples.json on disk if _state was never populated.
    data_dir = PROJECT_ROOT / "data" / "baselines"
    report = detect_drift(
        df, _state["baselines"], feature_cols,
        ref_samples=_state.get("ref_samples"),
        baselines_path=str(data_dir / "drift_baselines.json"),
    )

    DRIFT_DETECTED.set(1 if report["overall_drift"] else 0)
    DRIFT_FEATURES_COUNT.set(report["n_drifted"])

    # Send alert when drift detected via API (complements Airflow DAG alert)
    if report["overall_drift"]:
        try:
            from src.alert_notifier import send_drift_alert
            send_drift_alert(report["drifted_features"], report["n_drifted"])
        except Exception as _de:
            logger.warning(f"Drift alert failed (non-fatal): {_de}")

    return DriftReport(**report)


# ---------------------------------------------------------------------------
# Admin: reload baselines + ref_samples from disk (called by Airflow DAG
# at the end of preprocessing so the live API picks up new files without
# a container restart).
# ---------------------------------------------------------------------------
@app.post(
    "/admin/reload-baselines", tags=["Maintenance"],
    dependencies=[Depends(require_api_key)],
)
async def reload_baselines():
    """
    Reload `drift_baselines.json` and `ref_samples.json` from disk into
    process memory. Intended to be called by the Airflow DAG after the
    preprocessing task rewrites these files; without this, the API keeps
    using stale in-memory baselines until the container is restarted.
    """
    data_dir = PROJECT_ROOT / "data" / "baselines"
    baselines_path = data_dir / "drift_baselines.json"
    ref_samples_path = data_dir / "ref_samples.json"

    if not baselines_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"baselines file not found at {baselines_path}",
        )

    try:
        _state["baselines"] = load_json(str(baselines_path))
        _state["baselines_loaded_at"] = baselines_path.stat().st_mtime
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"baselines reload failed: {e}")

    ref_loaded = False
    if ref_samples_path.exists():
        try:
            _state["ref_samples"] = load_json(str(ref_samples_path))
            ref_loaded = True
        except Exception as e:
            logger.warning(f"ref_samples reload failed (non-fatal): {e}")
            _state["ref_samples"] = None
    else:
        _state["ref_samples"] = None

    logger.info(
        f"Reloaded baselines ({len(_state['baselines'])} features) | "
        f"ref_samples={'yes' if ref_loaded else 'no'}"
    )
    return {
        "status": "reloaded",
        "baselines_features": len(_state["baselines"]),
        "ref_samples_loaded": ref_loaded,
        "baselines_mtime": _state["baselines_loaded_at"],
    }


# ---------------------------------------------------------------------------
# Admin: reload model + training metrics from disk (called by Airflow DAG
# at the end of training so Prometheus / Grafana see the new f1/accuracy/
# version / last-training-timestamp without a container restart).
# ---------------------------------------------------------------------------
@app.post(
    "/admin/reload-model", tags=["Maintenance"],
    dependencies=[Depends(require_api_key)],
)
async def reload_model(data_source: str = "unknown"):
    """
    Reload `best_model.joblib`, `scaler.joblib`, and `test_metrics.json`
    from disk and refresh the corresponding Prometheus gauges:
      - model_accuracy
      - model_f1_score
      - last_training_timestamp_seconds
      - model_version_numeric
      - retrain_data_source  (1 = uploaded CSV, 0 = default)

    Without this, Grafana panels for "Model F1 Score", "Model Accuracy",
    "Time Since Last Training", etc. show stale values because the API
    only loaded these at startup.

    `data_source` query param: "uploaded" | "default" | "unknown".
    Sent by the Airflow DAG so retrain_data_source reflects the truth
    of what the training task actually used.
    """
    models_dir = PROJECT_ROOT / "models"
    model_path = models_dir / "best_model.joblib"
    scaler_path = models_dir / "scaler.joblib"
    metrics_path = models_dir / "test_metrics.json"

    if not model_path.exists():
        raise HTTPException(
            status_code=404, detail=f"model file not found at {model_path}"
        )

    try:
        _state["model"] = joblib.load(model_path)
        _state["model_source"] = "local-joblib"
        # Bump version on every reload so Grafana sees a step change
        try:
            current = float(_state.get("model_version", "0") or "0")
        except (ValueError, TypeError):
            current = 0.0
        _state["model_version"] = f"{current + 0.1:.1f}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"model reload failed: {e}")

    if scaler_path.exists():
        try:
            _state["scaler"] = joblib.load(scaler_path)
        except Exception as e:
            logger.warning(f"scaler reload failed (non-fatal): {e}")

    metrics_loaded = False
    if metrics_path.exists():
        try:
            metrics = load_json(str(metrics_path))
            MODEL_ACCURACY.set(metrics.get("accuracy", 0))
            MODEL_F1_SCORE.set(metrics.get("f1_score", 0))
            MODEL_INFO.info({
                "version": _state["model_version"],
                "source": _state["model_source"],
                "algorithm": "RandomForest",
                "f1_score": str(metrics.get("f1_score", 0)),
            })
            LAST_TRAINING_TIMESTAMP.set(metrics_path.stat().st_mtime)
            try:
                MODEL_VERSION_NUMERIC.set(float(_state["model_version"]))
            except (ValueError, TypeError):
                MODEL_VERSION_NUMERIC.set(0)
            metrics_loaded = True
        except Exception as e:
            logger.warning(f"metrics reload failed (non-fatal): {e}")

    # Reflect the data source the DAG actually used in the gauge
    if data_source == "uploaded":
        RETRAIN_DATA_SOURCE.set(1)
    elif data_source == "default":
        RETRAIN_DATA_SOURCE.set(0)
    # else: leave gauge as-is

    logger.info(
        f"Reloaded model (version={_state['model_version']}, "
        f"data_source={data_source}, metrics={'yes' if metrics_loaded else 'no'})"
    )
    return {
        "status": "reloaded",
        "model_version": _state["model_version"],
        "metrics_loaded": metrics_loaded,
        "data_source": data_source,
    }


# ---------------------------------------------------------------------------
# Retrain (authenticated)
# ---------------------------------------------------------------------------
@app.post(
    "/retrain", response_model=RetrainResponse, tags=["Maintenance"],
    dependencies=[Depends(require_api_key)],
)
async def trigger_retrain(reason: str = "manual"):
    """
    Trigger model retraining (requires X-API-Key header).

    Steps:
      1. Increment Prometheus retrain counter (visible in Grafana).
      2. Send retrain alert (email + log).
      3. Actually start the Airflow DAG via REST API so training runs
         end-to-end without a human opening the Airflow UI.
    """
    RETRAIN_TRIGGERS.labels(reason=reason).inc()
    logger.info(f"Retraining triggered — reason: {reason}")

    # Detect data source for accurate alerts + Prometheus gauge
    is_uploaded = (
        Path(UPLOADS_DIR).exists()
        and any(Path(UPLOADS_DIR).glob("*.csv"))
    )
    data_source = "uploaded" if is_uploaded else "default"

    # Throttle: skip the email if we already sent one for the same reason
    # within RETRAIN_ALERT_THROTTLE_SECONDS. Catches accidental double-clicks.
    now = time.time()
    last_sent_map = _state["last_manual_retrain_alert"]
    last_ts = last_sent_map.get(reason, 0)
    suppress_alert = (now - last_ts) < RETRAIN_ALERT_THROTTLE_SECONDS

    if suppress_alert:
        logger.info(
            f"Retrain alert for reason='{reason}' was sent "
            f"{int(now - last_ts)}s ago — suppressing duplicate email"
        )
    else:
        try:
            from src.alert_notifier import send_retrain_alert
            send_retrain_alert(
                reason=reason,
                model_version=_state.get("model_version", "unknown"),
                triggered_by="api",
                data_source=data_source,
            )
            last_sent_map[reason] = now
        except Exception as _ra:
            logger.warning(f"Retrain alert failed (non-fatal): {_ra}")

    # Actually start the Airflow DAG so training runs without a human
    # opening the Airflow UI. Result is reported back to the caller so
    # the frontend can show whether training really kicked off.
    airflow_result = _trigger_airflow_dag(reason=reason)
    if airflow_result["status"] == "triggered":
        message = (
            f"Retraining pipeline started. "
            f"Airflow DAG run: {airflow_result['dag_run_id']}. "
            "Watch Airflow UI for progress."
        )
    else:
        message = (
            "Retrain alert + counter recorded, but Airflow DAG could NOT "
            f"be started ({airflow_result['status']}: "
            f"{airflow_result.get('detail', '')}). "
            "Open Airflow UI to trigger manually."
        )

    return RetrainResponse(
        status="triggered" if airflow_result["status"] == "triggered" else "partial",
        message=message,
        triggered_by=reason,
    )


@app.post(
    "/retrain/upload", response_model=UploadRetrainResponse, tags=["Maintenance"],
    dependencies=[Depends(require_api_key)],
)
async def retrain_with_upload(
    file: UploadFile = File(...),
    reason: str = Form(default="csv_upload"),
):
    """
    Upload a new raw CSV dataset, validate it, store it, and trigger retraining.
    Requires X-API-Key header. CSV must contain the standard AI4I schema columns.
    """
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only CSV files are accepted (.csv)")

    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        UPLOAD_COUNT.labels(status="failed").inc()
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    # Validate required columns (raw schema — same as ai4i2020.csv)
    required_cols = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        "Machine failure",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        UPLOAD_COUNT.labels(status="failed").inc()
        raise HTTPException(
            status_code=422,
            detail=f"Missing required columns: {missing}. "
                   f"Expected: {required_cols}",
        )

    if len(df) < 50:
        UPLOAD_COUNT.labels(status="failed").inc()
        raise HTTPException(
            status_code=422,
            detail=f"Dataset too small ({len(df)} rows). Minimum 50 rows required.",
        )

    # Persist to uploads directory (writable via feedback_data volume)
    uploads_path = Path(UPLOADS_DIR)
    uploads_path.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    safe_name = Path(file.filename).stem[:40]
    filename = f"upload_{timestamp}_{safe_name}.csv"
    save_path = uploads_path / filename

    with open(save_path, "wb") as fout:
        fout.write(content)

    RETRAIN_TRIGGERS.labels(reason=reason).inc()
    UPLOAD_COUNT.labels(status="success").inc()
    UPLOAD_ROWS.set(len(df))
    UPLOAD_FILE_SIZE_BYTES.set(len(content))
    RETRAIN_DATA_SOURCE.set(1)   # uploaded CSV will be picked by next training run
    logger.info(f"CSV uploaded and stored: {save_path} ({len(df)} rows) | reason={reason}")

    # Suppress the retrain email if this exact CSV content was uploaded within
    # the TTL window. Counter still ticks, file is still saved, DAG still
    # runs — only the duplicate "training has been triggered" email is skipped.
    is_duplicate = _is_duplicate_upload(content)
    if is_duplicate:
        logger.info(
            f"Duplicate CSV upload detected (same content within "
            f"{RECENT_UPLOAD_TTL}s) — suppressing retrain alert email"
        )
    else:
        try:
            from src.alert_notifier import send_retrain_alert
            send_retrain_alert(
                reason=reason,
                model_version=_state.get("model_version", "unknown"),
                triggered_by="upload",
                data_source="uploaded",
            )
        except Exception as _ra:
            logger.warning(f"Upload retrain alert failed (non-fatal): {_ra}")

    # Actually start the Airflow DAG so training picks up the new CSV.
    airflow_result = _trigger_airflow_dag(reason=reason)
    dup_note = "Duplicate upload detected — alert suppressed. " if is_duplicate else ""
    if airflow_result["status"] == "triggered":
        msg = (
            f"CSV validated and stored ({len(df)} rows). "
            f"{dup_note}"
            f"Airflow DAG started: {airflow_result['dag_run_id']}. "
            "Training will use this upload as the data source."
        )
    else:
        msg = (
            f"CSV validated and stored ({len(df)} rows). "
            f"{dup_note}"
            "Training will automatically use this file on the next pipeline run "
            f"(Airflow auto-trigger {airflow_result['status']}: "
            f"{airflow_result.get('detail', '')}). "
            "Trigger the Airflow DAG manually to start now."
        )

    return UploadRetrainResponse(
        status="uploaded_duplicate" if is_duplicate else "uploaded_and_triggered",
        filename=filename,
        rows=len(df),
        columns=list(df.columns),
        message=msg,
        triggered_by=reason,
        save_path=str(save_path),
    )


@app.get("/retrain/uploads", response_model=UploadListResponse, tags=["Maintenance"])
async def list_uploads():
    """List previously uploaded CSV files (most recent 20)."""
    uploads_path = Path(UPLOADS_DIR)
    if not uploads_path.exists():
        return UploadListResponse(uploads=[])

    files = []
    for f in sorted(uploads_path.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
        stat = f.stat()
        # Try to read row count cheaply
        try:
            with open(f, "r") as fh:
                row_count = sum(1 for _ in fh) - 1  # subtract header
        except Exception:
            row_count = None
        files.append(UploadedFile(
            filename=f.name,
            size_bytes=stat.st_size,
            uploaded_at=stat.st_mtime,
            rows=max(row_count, 0) if row_count is not None else None,
        ))

    return UploadListResponse(uploads=files)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
# Rollback (authenticated)
# ---------------------------------------------------------------------------
@app.post(
    "/rollback", response_model=RollbackResponse, tags=["Maintenance"],
    dependencies=[Depends(require_api_key)],
)
async def rollback_model(request: RollbackRequest):
    """
    Roll back to a previous model version (requires X-API-Key header).

    Strategy:
    - If ``target_version`` is given, attempt to pull that version from the
      MLflow Model Registry and load it.
    - If not given, reload the local best_model.joblib + scaler.joblib on disk
      (useful after a bad automated retrain overwrote the artifacts).

    The in-memory model is hot-swapped — the API keeps serving during rollback.
    """
    previous_version = _state.get("model_version", "unknown")
    models_dir = PROJECT_ROOT / "models"

    try:
        if request.target_version:
            # ── MLflow registry rollback ───────────────────────────────
            import mlflow.sklearn
            from src.utils import load_config
            cfg = load_config()
            mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
            model_name = cfg["mlflow"]["model_name"]
            mlflow.set_tracking_uri(mlflow_uri)
            model_uri = f"models:/{model_name}/{request.target_version}"
            logger.info(f"Rolling back to MLflow model {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)
            target_ver = request.target_version
        else:
            # ── Local artifact reload ──────────────────────────────────
            model_path = models_dir / "best_model.joblib"
            if not model_path.exists():
                ROLLBACK_TRIGGERS.labels(status="failed").inc()
                raise HTTPException(
                    status_code=503,
                    detail="No local model artifact found to roll back to.",
                )
            model = joblib.load(model_path)
            target_ver = "local-reload"
            logger.info(f"Rolled back to local artifact: {model_path}")

        # Hot-swap in-memory state
        _state["model"] = model
        _state["model_version"] = target_ver
        _state["model_source"] = "rollback"

        # Reload scaler
        scaler_path = models_dir / "scaler.joblib"
        if scaler_path.exists():
            _state["scaler"] = joblib.load(scaler_path)

        # Reload metrics for Prometheus
        metrics_path = models_dir / "test_metrics.json"
        if metrics_path.exists():
            metrics = load_json(str(metrics_path))
            MODEL_ACCURACY.set(metrics.get("accuracy", 0))
            MODEL_F1_SCORE.set(metrics.get("f1_score", 0))

        ROLLBACK_TRIGGERS.labels(status="success").inc()
        logger.warning(
            f"[ROLLBACK] {previous_version} → {target_ver} | reason={request.reason}"
        )

        try:
            from src.alert_notifier import send_retrain_alert
            send_retrain_alert(
                reason=f"rollback:{request.reason}",
                model_version=previous_version,
                triggered_by="rollback",
                data_source="rollback",
            )
        except Exception:
            pass

        return RollbackResponse(
            status="success",
            previous_version=previous_version,
            target_version=target_ver,
            model_loaded=True,
            message=(
                f"Model rolled back from version {previous_version} to {target_ver}. "
                "The API is now serving the restored model."
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        ROLLBACK_TRIGGERS.labels(status="failed").inc()
        logger.error(f"Rollback failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rollback failed: {e}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)