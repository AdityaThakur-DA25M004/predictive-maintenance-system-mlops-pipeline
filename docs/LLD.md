# Low-Level Design (LLD)

> Predictive Maintenance System — module boundaries, file responsibilities, and full API endpoint I/O specification.

## 1. Module / file responsibilities

### Backend (`src/`)

| File | Responsibility |
|---|---|
| `data_ingestion.py` | Load raw CSV, validate schema, drop leaky columns (TWF/HDF/PWF/OSF/RNF + UDI/Product ID), stratified train/test split, write `train.csv` + `test.csv`. Selects between default raw path and latest user upload. |
| `data_preprocessing.py` | Feature engineering (`temp_diff`, `power`, `wear_degree`, `type_encoded`, `speed_torque_ratio`), StandardScaler fit/transform, drift baseline computation (per-feature mean/std/quantiles), reference-sample persistence for KS test. |
| `drift_detection.py` | Per-feature KS test against reference samples + PSI computation. Emits `drift_report.json`. CLI entry point for DVC. |
| `model_training.py` | RandomForest grid search, MLflow run logging, model + metrics persistence, registry registration, classification report. |
| `alert_notifier.py` | SMTP email dispatch helpers: `send_drift_alert`, `send_retrain_alert`, `send_training_complete_alert`, `send_accuracy_degradation_alert`, `send_error_rate_alert`. |
| `utils.py` | `load_config`, `setup_logger`, `get_project_root`, `ensure_dir`, `save_json`, `load_json`. |

### API (`api/`)

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app, all route handlers, model + scaler + baseline lifecycle, Prometheus instrumentation, feedback DB integration, Airflow trigger client. |
| `metrics.py` | Prometheus metric definitions (Counter, Histogram, Gauge, Info) and recording helpers. |
| `schemas.py` | Pydantic models for request/response validation. |

### Frontend (`frontend/`)

| File | Responsibility |
|---|---|
| `app.py` | Main dashboard page. |
| `pages/1_Predict.py` | Single + batch prediction UI, feedback submission. |
| `pages/2_Pipeline.py` | DAG status, dataset upload, retrain trigger, rollback. |
| `pages/3_Monitoring.py` | Drift checks, model quality, alerts. |
| `pages/4_User_Manual.py` | End-user documentation. |
| `common.py` | API client (`APIClient`), config loader, shared CSS, sidebar/header helpers. |

### Orchestration (`dags/`)

| File | Responsibility |
|---|---|
| `ml_pipeline_dag.py` | Single Airflow DAG `predictive_maintenance_pipeline` with 8 tasks: ingest, drift_check, preprocess, reload_api_baselines, train, reload_api_model, drift_branch, retrain_notification / no_drift_end → pipeline_end. |

## 2. API endpoint specification

> **Base URL.** `http://localhost:8000` (host) or `http://api:8000` (Docker DNS).
> **Auth.** Endpoints marked 🔒 require header `X-API-Key: <RETRAIN_API_KEY>`.
> **OpenAPI doc.** Auto-generated at `/docs`.

### 2.1 System endpoints

#### `GET /health`
Liveness probe. Returns 200 always (provided the process is alive).

| | |
|---|---|
| Request body | — |
| Response 200 | `{ "status": "healthy", "model_loaded": bool, "scaler_loaded": bool, "uptime_seconds": int }` |

#### `GET /ready`
Readiness probe — returns 503 if model or scaler isn't loaded.

| | |
|---|---|
| Request body | — |
| Response 200 | `{ "ready": true, "components": {...} }` |
| Response 503 | `{ "ready": false, "components": {...} }` |

#### `GET /model/info`
Currently-loaded model metadata.

| | |
|---|---|
| Request body | — |
| Response 200 | `{ "model_version": str, "algorithm": str, "test_metrics": { "f1_score", "accuracy", "precision", "recall", "roc_auc" }, "trained_at": str, "data_source": "default" \| "uploaded" }` |
| Response 503 | If model not loaded |

#### `GET /model/feature-importance`
RandomForest feature importance vector.

| | |
|---|---|
| Request body | — |
| Response 200 | `{ "features": [str], "importance": [float] }` |
| Response 503 | If model not loaded |

### 2.2 Prediction endpoints

#### `POST /predict`
Single sensor reading → failure probability.

| | |
|---|---|
| Request body | `SensorInput` — `{ "type": "L"\|"M"\|"H", "air_temperature": float (250–350), "process_temperature": float, "rotational_speed": int, "torque": float, "tool_wear": int }` |
| Response 200 | `{ "prediction_id": int, "failure_predicted": bool, "failure_probability": float (0–1), "model_version": str, "latency_ms": float }` |
| Response 422 | Pydantic validation error (out-of-range temp, invalid type, missing field) |
| Response 503 | If model not loaded |

#### `POST /predict/batch`
Multiple readings in one call.

| | |
|---|---|
| Request body | `{ "readings": [SensorInput] }` (max 1000 entries) |
| Response 200 | `{ "predictions": [PredictionResponse], "total_latency_ms": float, "batch_size": int }` |
| Response 422 | Validation error |

### 2.3 Feedback endpoints

#### `POST /feedback`
Log ground-truth label for a previous prediction.

| | |
|---|---|
| Request body | `{ "prediction_id": int, "actual_label": 0\|1 }` |
| Response 200 | `{ "logged": true, "running_accuracy": float }` |
| Response 404 | If `prediction_id` is unknown |

#### `GET /feedback/stats`
Rolling + overall accuracy from logged feedback.

| | |
|---|---|
| Request body | — |
| Response 200 | `{ "total_feedback": int, "overall_accuracy": float \| null, "rolling_accuracy": float \| null, "window": int, "correct": int, "incorrect": int }` |

### 2.4 Monitoring endpoints

#### `POST /drift/check`
KS + PSI drift report against current baselines.

| | |
|---|---|
| Request body | `{ "readings": [SensorInput] }` (≥ 30 readings recommended) |
| Response 200 | `{ "overall_drift": bool, "n_drifted": int, "drifted_features": [str], "feature_details": { feature: { "ks_pvalue", "psi", "mean_shift_std", "drift_detected" } }, "reference_type": str }` |
| Response 422 | Empty or malformed batch |

#### `GET /metrics`
Prometheus exposition format. Scraped by Prometheus every 15 s.

| | |
|---|---|
| Request body | — |
| Response 200 | `text/plain; version=0.0.4` Prometheus metrics |

### 2.5 Maintenance endpoints (🔒 require `X-API-Key`)

#### `POST /retrain`
Trigger Airflow DAG with the existing data source (no upload).

| | |
|---|---|
| Query params | `reason: str` |
| Headers | `X-API-Key: <key>` |
| Response 200 | `{ "triggered": true, "dag_run_id": str, "started_at": str }` |
| Response 401 | Missing API key |
| Response 403 | Wrong API key |

#### `POST /retrain/upload`
Validate + persist a CSV → trigger Airflow DAG using that upload as the data source.

| | |
|---|---|
| Form data | `file: UploadFile (multipart/form-data, CSV), reason: str` |
| Headers | `X-API-Key: <key>` |
| Response 200 | `{ "validated_rows": int, "filename": str, "dag_run_id": str, "started_at": str }` |
| Response 400 | Schema invalid / parse error |
| Response 401/403 | Auth |

#### `GET /retrain/uploads`
List previously uploaded datasets.

| | |
|---|---|
| Response 200 | `{ "uploads": [ { "filename": str, "uploaded_at": float, "rows": int, "reason": str } ] }` |

#### `POST /rollback`
Roll the API back to a previous registered model version.

| | |
|---|---|
| Query params | `target_version: str` |
| Headers | `X-API-Key: <key>` |
| Response 200 | `{ "rolled_back_to": str, "previous_version": str }` |
| Response 404 | Target version not in registry |

#### `POST /admin/reload-baselines`
Hot-reload `drift_baselines.json` + `ref_samples.json` from disk into the running API. Called by Airflow after preprocessing.

| | |
|---|---|
| Headers | `X-API-Key: <key>` |
| Response 200 | `{ "reloaded": true, "baselines_path": str, "ref_samples_path": str }` |

#### `POST /admin/reload-model`
Hot-reload `best_model.joblib` + `scaler.joblib` + `test_metrics.json` from disk. Called by Airflow after training.

| | |
|---|---|
| Query params | `data_source: "default" \| "uploaded" \| "unknown"` |
| Headers | `X-API-Key: <key>` |
| Response 200 | `{ "reloaded": true, "model_version": str, "data_source": str }` |

## 3. MLflow Models Serve endpoint (parallel)

The `mlflow-serve` sidecar (port 5001) exposes a single endpoint conforming to the MLflow standard:

#### `POST /invocations`
| | |
|---|---|
| Request body | `{ "dataframe_split": { "columns": [...], "data": [[...], ...] } }` — see [MLflow scoring server docs](https://mlflow.org/docs/latest/cli.html#mlflow-models-serve) |
| Response 200 | `{ "predictions": [ ... ] }` |

#### `GET /ping`
Health probe — returns 200 if model is loaded.

## 4. Pydantic schemas (request / response)

Defined in `api/schemas.py`:

- `SensorInput` — single reading; field validators clamp `air_temperature` to (250, 350), enforce `type` ∈ {L, M, H}.
- `PredictionResponse` — wraps a single prediction with metadata.
- `BatchInput` / `BatchResponse` — list-of-readings + list-of-predictions.
- `FeedbackInput` / `FeedbackResponse` / `FeedbackStats` — feedback domain.
- `DriftReport` — full per-feature report.
- `RetrainResponse` / `UploadRetrainResponse` / `UploadListResponse` — maintenance domain.
- `RollbackResponse` — rollback domain.
- `HealthResponse` — system domain.

## 5. Database schema (feedback DB)

Single SQLite database at `/app/data/feedback/feedback.db` (volume-backed), one table:

```sql
CREATE TABLE IF NOT EXISTS predictions (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at      REAL NOT NULL,           -- unix epoch float
  model_version   TEXT,
  features_json   TEXT NOT NULL,           -- serialized SensorInput
  prediction      INTEGER NOT NULL,        -- 0 or 1
  probability     REAL NOT NULL,           -- 0..1
  actual_label    INTEGER,                 -- nullable until feedback POSTed
  feedback_at     REAL                     -- nullable
);
CREATE INDEX IF NOT EXISTS ix_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS ix_predictions_actual    ON predictions(actual_label);
```

## 6. Configuration surface

| Path | Purpose |
|---|---|
| `configs/config.yaml` | Default config (Docker) |
| `configs/config.local.yaml` | Local override (selected via `PM_CONFIG_PATH`) |
| `.env` | Secrets, ports, API keys, SMTP creds, DagsHub creds |

Key config sections: `data` (paths + split config), `features` (column lists), `model` (algorithm + hyperparameter grid), `mlflow`, `api`, `monitoring` (drift_threshold, error_rate_threshold), `docker` (port mappings).

## 7. Error handling contract

- All API errors return `{ "detail": str }` with an appropriate HTTP status.
- Airflow tasks log to stdout; `local_task_job_runner` captures + persists to `airflow/logs/<dag>/<task>/<run>/`.
- Frontend catches `APIError` and renders an `st.error(...)` banner with the message.
- Retry policy: Airflow tasks retry once with 2-min delay; API has no automatic retry on the request side (clients are expected to back off).

## 8. Deployment topology

8 containers managed by Docker Compose, plus one optional sidecar:
- `postgres` (Airflow metadata DB)
- `mlflow` (tracking + registry UI)
- `api` (FastAPI inference + admin)
- `frontend` (Streamlit)
- `airflow-init`, `airflow-webserver`, `airflow-scheduler`
- `prometheus`
- `grafana`
- _Optional_: `mlflow-serve` (parallel MLflow models serve sidecar via override file)

Single Docker network `pm-net` connects all services. Bind mounts are used for `models/`, `data/baselines/`, and `configs/` (read-only into the API); named volumes for postgres, MLflow DB, MLflow artifacts, feedback DB, prometheus TSDB.