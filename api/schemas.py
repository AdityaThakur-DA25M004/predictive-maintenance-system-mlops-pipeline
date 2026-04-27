"""
Pydantic schemas for API request/response validation.
Includes feedback loop schemas for ground-truth logging.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


# Prediction
class SensorInput(BaseModel):
    """Single sensor reading for prediction."""
    model_config = ConfigDict(
        protected_namespaces=(),    # Fix: prevent 'model_' namespace conflict
        json_schema_extra={
            "example": {
                "air_temperature": 300.0, "process_temperature": 310.0,
                "rotational_speed": 1500, "torque": 40.0,
                "tool_wear": 100, "product_type": "L",
            }
        },
    )

    air_temperature: float = Field(..., description="Air temperature in Kelvin", ge=250, le=350)
    process_temperature: float = Field(..., description="Process temperature in Kelvin", ge=250, le=400)
    rotational_speed: int = Field(..., description="Rotational speed in RPM", ge=0, le=10_000)
    torque: float = Field(..., description="Torque in Nm", ge=0, le=500)
    tool_wear: int = Field(..., description="Tool wear in minutes", ge=0, le=1000)
    product_type: str = Field(..., description="Product type: H, M, or L", pattern="^[HML]$")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int = Field(..., description="0 = No failure, 1 = Failure")
    failure_probability: float = Field(..., description="Probability of failure", ge=0, le=1)
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH / CRITICAL")
    model_version: str = Field(default="latest")
    prediction_id: Optional[int] = Field(
        default=None,
        description="Unique prediction ID — use with POST /feedback to submit ground truth.",
    )
    inference_time_ms: Optional[float] = Field(
        default=None,
        description="Pure model inference time in milliseconds (excludes API/network overhead)",
    )


# Batch
class BatchInput(BaseModel):
    readings: list[SensorInput] = Field(..., min_length=1, max_length=10_000)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    failures_detected: int


# System
class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    scaler_loaded: bool
    uptime_seconds: float


# Drift
class DriftReport(BaseModel):
    overall_drift: bool
    n_drifted: int
    total_features_checked: int
    drifted_features: list[str]
    features: dict
    reference_type: str = "unknown" 


# Retrain
class RetrainResponse(BaseModel):
    status: str
    message: str
    triggered_by: Optional[str] = None


# Feedback loop (ground-truth)
class FeedbackInput(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={"example": {"prediction_id": 42, "actual_label": 1}},
    )
    prediction_id: int = Field(..., description="ID returned by POST /predict", gt=0)
    actual_label: int = Field(..., description="Observed label: 0=no failure, 1=failure", ge=0, le=1)


class FeedbackResponse(BaseModel):
    status: str
    prediction_id: int
    correct: bool
    rolling_accuracy: Optional[float] = None


class FeedbackStats(BaseModel):
    total_feedback: int
    overall_accuracy: Optional[float] = None
    rolling_accuracy: Optional[float] = None
    window: int


# Upload & Retrain
class UploadRetrainResponse(BaseModel):
    status: str
    filename: str
    rows: int
    columns: list[str]
    message: str
    triggered_by: Optional[str] = None
    save_path: Optional[str] = None


class UploadedFile(BaseModel):
    filename: str
    size_bytes: int
    uploaded_at: float
    rows: Optional[int] = None


class UploadListResponse(BaseModel):
    uploads: list[UploadedFile]


# Rollback
class RollbackRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    target_version: Optional[str] = Field(
        default=None,
        description=(
            "MLflow model version to roll back to. "
            "If None, the API reloads the most recent local best_model.joblib "
            "and its companion scaler + metrics."
        ),
    )
    reason: str = Field(
        default="manual_rollback",
        description="Why rollback is being performed (logged to Prometheus).",
    )


class RollbackResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    previous_version: str
    target_version: str
    model_loaded: bool
    message: str