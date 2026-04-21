"""
Pydantic schemas for API request/response validation.
Includes feedback loop schemas for ground-truth logging.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class SensorInput(BaseModel):
    """Single sensor reading for prediction."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "air_temperature": 300.0, "process_temperature": 310.0,
            "rotational_speed": 1500, "torque": 40.0,
            "tool_wear": 100, "product_type": "L",
        }
    })

    air_temperature: float = Field(..., description="Air temperature in Kelvin", ge=250, le=350)
    process_temperature: float = Field(..., description="Process temperature in Kelvin", ge=250, le=400)
    rotational_speed: int = Field(..., description="Rotational speed in RPM", ge=0, le=10_000)
    torque: float = Field(..., description="Torque in Nm", ge=0, le=500)
    tool_wear: int = Field(..., description="Tool wear in minutes", ge=0, le=1000)
    product_type: str = Field(..., description="Product type: H, M, or L", pattern="^[HML]$")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = No failure, 1 = Failure")
    failure_probability: float = Field(..., description="Probability of failure", ge=0, le=1)
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH / CRITICAL")
    model_version: str = Field(default="latest")
    prediction_id: Optional[int] = Field(
        default=None,
        description="Unique prediction ID — use with POST /feedback to submit ground truth.",
    )


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------
class BatchInput(BaseModel):
    readings: list[SensorInput] = Field(..., min_length=1, max_length=10_000)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    failures_detected: int


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------
class DriftReport(BaseModel):
    overall_drift: bool
    n_drifted: int
    total_features_checked: int
    drifted_features: list[str]
    features: dict


# ---------------------------------------------------------------------------
# Retrain
# ---------------------------------------------------------------------------
class RetrainResponse(BaseModel):
    status: str
    message: str
    triggered_by: Optional[str] = None


# ---------------------------------------------------------------------------
# Feedback loop (ground-truth)
# ---------------------------------------------------------------------------
class FeedbackInput(BaseModel):
    """Submit the ground-truth label for a previous prediction."""
    model_config = ConfigDict(json_schema_extra={
        "example": {"prediction_id": 42, "actual_label": 1}
    })
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
