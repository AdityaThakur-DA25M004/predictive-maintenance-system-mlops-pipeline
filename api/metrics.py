"""
Prometheus Metrics Instrumentation for the Predictive Maintenance API.

Defines custom metrics for monitoring prediction latency, counts,
error rates, drift status, model performance, and real-world
performance decay via the feedback loop.
"""

from prometheus_client import Counter, Histogram, Gauge, Info

# ---------------------------------------------------------------------------
# Request metrics
# ---------------------------------------------------------------------------
PREDICTION_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["endpoint", "status"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
)

ERROR_COUNT = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"],
)

ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of currently active requests",
)

# ---------------------------------------------------------------------------
# Model metrics
# ---------------------------------------------------------------------------
FAILURE_PREDICTIONS = Counter(
    "failure_predictions_total",
    "Total number of failure predictions",
    ["risk_level"],
)

FAILURE_PROBABILITY = Histogram(
    "failure_probability_distribution",
    "Distribution of failure probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

MODEL_ACCURACY = Gauge("model_accuracy", "Current model accuracy on test set")
MODEL_F1_SCORE = Gauge("model_f1_score", "Current model F1 score on test set")
MODEL_INFO = Info("model", "Information about the loaded model")

# ---------------------------------------------------------------------------
# Drift metrics
# ---------------------------------------------------------------------------
DRIFT_DETECTED = Gauge("drift_detected", "Whether data drift has been detected (1=yes, 0=no)")
DRIFT_FEATURES_COUNT = Gauge("drift_features_count", "Number of features with detected drift")
DRIFT_CHECK_COUNT = Counter("drift_checks_total", "Total number of drift detection checks performed")

# ---------------------------------------------------------------------------
# Feedback loop metrics (guideline §II.E.2 — real-world performance decay)
# ---------------------------------------------------------------------------
FEEDBACK_COUNT = Counter(
    "feedback_total",
    "Total number of ground-truth feedback events received",
    ["outcome"],  # "correct" | "incorrect"
)

FEEDBACK_ROLLING_ACCURACY = Gauge(
    "feedback_rolling_accuracy",
    "Rolling accuracy over the last N ground-truth feedback events",
)

# ---------------------------------------------------------------------------
# Maintenance metrics
# ---------------------------------------------------------------------------
RETRAIN_TRIGGERS = Counter(
    "retrain_triggers_total",
    "Total number of retraining triggers",
    ["reason"],
)
