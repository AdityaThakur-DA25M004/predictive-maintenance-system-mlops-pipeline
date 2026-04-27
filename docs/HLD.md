# High-Level Design (HLD)

> Predictive Maintenance System — design choices, rationale, and component decomposition.

## 1. Problem statement

**Business problem.** Industrial machine downtime is expensive. Predictive maintenance — flagging machines likely to fail in the near future — lets operators schedule preventive servicing before catastrophic failure.

**Data.** AI4I 2020 dataset (10,000 rows, 14 columns) with sensor readings (air/process temperature, rotational speed, torque, tool wear) and a binary `Machine failure` label plus 5 deterministic per-failure-mode flags (TWF, HDF, PWF, OSF, RNF).

**ML metric.** F1-score (target ≥ 0.80) — class imbalance (~3.4% failure rate) makes accuracy a misleading metric.

**Business metric.** Inference latency < 200 ms per request (configurable in `config.yaml: api.latency_threshold_ms`).

## 2. Design principles

- **Separation of concerns.** Frontend, backend, orchestration, monitoring, and storage are independent containers.
- **Loose coupling.** Frontend ↔ Backend over REST only. No shared imports across the boundary.
- **Reproducibility.** Every model run is tied to a Git commit + MLflow run ID + DVC lock entry.
- **Observability first.** Every external surface is instrumented (Prometheus counters, histograms, gauges).
- **Fail soft.** Alerts and hot-reloads are best-effort — pipeline failures don't cascade.

## 3. Component decomposition

### 3.1 Frontend (Streamlit, container `frontend`)

Multi-page Streamlit app with a shared `common.py` API client. Pages:
- **Dashboard** — system health, model performance gauges, pipeline timeline, quick actions.
- **Predict** — single + batch sensor-reading inference, feedback submission.
- **Pipeline** — DAG status, dataset upload + retrain trigger, manual rollback.
- **Monitoring** — model quality, drift detection, retrain controls, live alerts.
- **User Manual** — non-technical end-user guide.

The frontend never imports backend modules; it only makes HTTP calls.

### 3.2 Backend (FastAPI, container `api`)

REST API with Pydantic schemas (`schemas.py`). Endpoint groups:
- **System** — `/health`, `/ready`, `/model/info`, `/model/feature-importance`
- **Prediction** — `/predict`, `/predict/batch`
- **Feedback** — `/feedback`, `/feedback/stats`
- **Monitoring** — `/drift/check`, `/metrics` (Prometheus exporter)
- **Maintenance** (API-key gated) — `/retrain`, `/retrain/upload`, `/retrain/uploads`, `/rollback`, `/admin/reload-baselines`, `/admin/reload-model`

The API loads `best_model.joblib` and `scaler.joblib` from a read-only mount at startup, plus the drift baselines from `data/baselines/`. Admin reload endpoints flip in-memory references without restarting the container.

### 3.3 Parallel MLflow Models Serve (container `mlflow-serve`, optional)

`mlflow models serve` against `models:/predictive-maintenance-model/latest` exposes `POST /invocations` on port 5001. This is the literal MLflow-native serving path that the assignment rubric asks for. Production traffic continues to flow through FastAPI which has the full feature set; mlflow-serve is a parallel demonstration endpoint.

### 3.4 Orchestration (Airflow, containers `airflow-webserver` + `airflow-scheduler`)

Single DAG `predictive_maintenance_pipeline` with the following task order:

```
data_ingestion
   → drift_check          (against OLD baselines)
      → preprocessing     (writes NEW baselines)
         → reload_api_baselines
            → model_training
               → reload_api_model
                  → drift_branch
                     ├── retrain_notification → pipeline_end
                     └── no_drift_end → pipeline_end
```

**Key design choice — drift_check before preprocessing.** Preprocessing always overwrites the baselines from the current training data. If drift_check ran *after* preprocessing (an earlier version did), it would always compare new data against itself and report zero drift. Moving drift_check *before* preprocessing means the comparison is honest: new data vs the previously-deployed model's expected distribution.

Triggers:
- **Scheduled** — daily.
- **API-driven** — `/retrain/upload` triggers an ad-hoc DAG run with the freshly uploaded CSV.

### 3.5 Tracking + Registry (MLflow, container `mlflow`)

- **Tracking** — every run logs hyperparameters, metrics (F1, ROC-AUC, accuracy, precision, recall), confusion matrix, classification report.
- **Registry** — best model per run is registered as `predictive-maintenance-model`; new versions roll forward automatically.
- **Backend store** — SQLite for run metadata; local artifact store for binaries.

### 3.6 Monitoring (Prometheus + Grafana)

- **Prometheus** scrapes `/metrics` every 15 seconds.
- **Custom metrics** — request count, request latency histogram, prediction count by class, drift status (1=drifted, 0=not), retrain trigger count, feedback accuracy gauge, model version info gauge, training duration histogram.
- **Alert rules** (`predictive_maintenance.json`) — drift active, retrain storm (>5 retrains in 15 min), low rolling accuracy, high error rate.
- **Grafana dashboard** — 25 panels covering all of the above.
- **Email alerts** via `src/alert_notifier.py` driven by SMTP env vars.

### 3.7 Versioning (DVC + DagsHub)

- `dvc.yaml` defines four stages: `data_ingestion → drift_check → preprocessing → training`.
- `dvc.lock` pins exact content hashes for every input and output.
- DagsHub remote stores tracked binaries (data, models, baselines).
- Git tags `v1.0` and `v2.0` mark release points.

## 4. Technology choices and rationale

| Concern | Choice | Why |
|---|---|---|
| API framework | FastAPI | Async, OpenAPI auto-gen, Pydantic validation, low overhead |
| Frontend | Streamlit | Fast iteration, native Python, multi-page support |
| Orchestration | Airflow | Mature DAG semantics, branch operators, web UI, Docker-friendly |
| Tracking | MLflow | Industry standard, registry built in, language-agnostic clients |
| Versioning | DVC + DagsHub | Git-native data versioning; DagsHub adds free remote + UI |
| Monitoring | Prometheus + Grafana | De facto observability stack, rich PromQL, free tier |
| Container runtime | Docker Compose | Single-host multi-service, no Kubernetes overhead for an academic project |
| Persistence | SQLite (feedback) | Embedded, zero-ops, sufficient for low write rate |

## 5. Cross-cutting concerns

- **Configuration** — single `config.yaml` (Docker) + `config.local.yaml` (local dev). Selected by `PM_CONFIG_PATH` env var.
- **Logging** — `utils.setup_logger` provides a consistent format across all modules.
- **Error handling** — backend raises typed `HTTPException`; frontend catches `APIError` and renders user-friendly messages.
- **Security** — admin endpoints require `X-API-Key` header. No PII in the dataset, so no encryption-at-rest required for this dataset; bind-mounted volumes use OS permissions.

## 6. Failure modes and mitigations

| Failure | Detection | Mitigation |
|---|---|---|
| Model file corrupt / missing | API `/health` returns `model_loaded: false` | Startup fails fast; log emitted; rollback to previous version via `/rollback` |
| Airflow DAG fails | Task failure email; pipeline_end uses `none_failed_min_one_success` so partial success still completes | Retry once with 2-min backoff; on permanent failure, alert fires |
| Drift detected | `drift_check` task XCom; Prometheus alert | Branch routes to `retrain_notification`; new training run triggered |
| API ↔ MLflow disconnect | Hot-reload best-effort, returns `unreachable` | Pipeline doesn't fail; next reload attempt at next DAG run |
| User uploads schema-invalid CSV | `validate_schema` in ingestion raises | API returns 400 before storing; Streamlit shows error |

## 7. Out of scope (deliberate non-goals)

- Multi-tenant deployment
- Online learning / streaming inference
- Distributed training (single-node Random Forest is sufficient for AI4I scale)
- Cloud deployment (assignment forbids cloud)
- A/B testing of models (rolled forward via registry instead)
