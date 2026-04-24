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

# ---------------------------------------------------------------------------
# Upload & retraining observability metrics
# ---------------------------------------------------------------------------
UPLOAD_COUNT = Counter(
    "dataset_uploads_total",
    "Total number of CSV dataset uploads via /retrain/upload",
    ["status"],          # "success" | "failed"
)

UPLOAD_ROWS = Gauge(
    "upload_last_rows",
    "Row count of the most recently uploaded dataset CSV",
)

UPLOAD_FILE_SIZE_BYTES = Gauge(
    "upload_last_file_size_bytes",
    "File size in bytes of the most recently uploaded dataset CSV",
)

RETRAIN_DATA_SOURCE = Gauge(
    "retrain_data_source",
    "Data source used in the last triggered retrain: 1=uploaded CSV, 0=default dataset",
)

TRAINING_DURATION_SECONDS = Histogram(
    "training_duration_seconds",
    "Wall-clock duration of the full training pipeline run in seconds",
    buckets=[30, 60, 90, 120, 180, 240, 300, 420, 600],
)

LAST_TRAINING_TIMESTAMP = Gauge(
    "last_training_timestamp_seconds",
    "Unix timestamp of the most recent completed training run",
)

MODEL_VERSION_NUMERIC = Gauge(
    "model_version_numeric",
    "Numeric version of the currently deployed model (from MLflow registry)",
)

# ---------------------------------------------------------------------------
# Alert notification metrics
# ---------------------------------------------------------------------------
ALERT_NOTIFICATIONS_TOTAL = Counter(
    "alert_notifications_total",
    "Total number of alert notifications dispatched by the system",
    ["alert_type", "channel"],   # alert_type: drift|retrain|accuracy|error_rate|training_complete
                                  # channel: email|log
)

ROLLBACK_TRIGGERS = Counter(
    "rollback_triggers_total",
    "Total number of model rollback operations",
    ["status"],   # "success" | "failed"
)