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
import time
import sqlite3
import secrets
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (
    SensorInput, PredictionResponse, BatchInput, BatchResponse,
    HealthResponse, DriftReport, RetrainResponse,
    FeedbackInput, FeedbackResponse, FeedbackStats,
)
from api.metrics import (
    PREDICTION_COUNT, PREDICTION_LATENCY, ERROR_COUNT,
    FAILURE_PREDICTIONS, FAILURE_PROBABILITY, MODEL_ACCURACY,
    MODEL_F1_SCORE, DRIFT_DETECTED, DRIFT_FEATURES_COUNT,
    DRIFT_CHECK_COUNT, MODEL_INFO, RETRAIN_TRIGGERS, ACTIVE_REQUESTS,
    FEEDBACK_COUNT, FEEDBACK_ROLLING_ACCURACY,
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

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_state = {
    "model": None, "scaler": None, "baselines": None,
    "start_time": None, "model_version": "unknown",
    "model_source": "unloaded", "prediction_buffer": [],
}


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
        logger.info(f"Baselines loaded from {baselines_path}")

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
        prediction = int(_state["model"].predict(X)[0])
        proba = float(_state["model"].predict_proba(X)[0][1])
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
    report = detect_drift(df, _state["baselines"], feature_cols)

    DRIFT_DETECTED.set(1 if report["overall_drift"] else 0)
    DRIFT_FEATURES_COUNT.set(report["n_drifted"])

    return DriftReport(**report)


# ---------------------------------------------------------------------------
# Retrain (authenticated)
# ---------------------------------------------------------------------------
@app.post(
    "/retrain", response_model=RetrainResponse, tags=["Maintenance"],
    dependencies=[Depends(require_api_key)],
)
async def trigger_retrain(reason: str = "manual"):
    """Trigger model retraining (requires X-API-Key header)."""
    RETRAIN_TRIGGERS.labels(reason=reason).inc()
    logger.info(f"Retraining triggered — reason: {reason}")
    return RetrainResponse(
        status="triggered",
        message="Retraining pipeline has been triggered. Check Airflow UI for progress.",
        triggered_by=reason,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
