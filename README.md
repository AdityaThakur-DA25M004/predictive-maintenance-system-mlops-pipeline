# Predictive Maintenance System — End-to-End MLOps Pipeline

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](.github/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-42%20passed-brightgreen)](docs/test_plan.md)
[![F1 Score](https://img.shields.io/badge/F1-0.906-success)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.977-success)]()
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)](docker-compose.yml)
[![DVC](https://img.shields.io/badge/DVC-3.x-945DD6)](dvc.yaml)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2)](MLproject)

A production-grade predictive maintenance system that flags industrial machine failures *before* they happen, built on the AI4I 2020 dataset. Demonstrates the complete MLOps lifecycle — from data ingestion and reproducible training, through containerized serving and observability, to drift detection and automated retraining.

> **Course context.** Submission for the MLOps assignment, RollNo.- DA25M004, Aditya Thakur. Repo hosted on DagsHub at [`AdityaThakur-DA25M004/predictive-maintenance-system-mlops-pipeline`](https://dagshub.com/AdityaThakur-DA25M004/predictive-maintenance-system-mlops-pipeline).

---

## Table of contents

- [What this project does](#what-this-project-does)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Performance](#performance)
- [Quick start](#quick-start)
- [Service map](#service-map)
- [ML pipeline](#ml-pipeline)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Versioning and reproducibility](#versioning-and-reproducibility)
- [Monitoring and alerts](#monitoring-and-alerts)
- [Working system — visual evidence](#working-system--visual-evidence)
- [Troubleshooting](#troubleshooting)
- [Documentation index](#documentation-index)

---

## What this project does

**Business problem.** Industrial machine downtime is expensive. Predictive maintenance — flagging machines likely to fail in the near future — lets operators schedule preventive servicing before catastrophic failure.

**ML problem.** Binary classification on the [AI4I 2020 dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (10,000 rows × 14 columns). Sensor inputs: air/process temperature, rotational speed, torque, tool wear, product type. Target: `Machine failure ∈ {0, 1}`.

**Class imbalance.** ~3.4% positive class. F1-score is the headline metric.

**What's in the box.** A complete MLOps deployment with reproducible training, model registry, drift detection, automated retraining, observability, and a polished UI — all running locally via Docker Compose.

---

## Architecture

```
┌─────────────┐  REST   ┌──────────────┐  hot-reload  ┌─────────────┐
│  Streamlit  │ ──────▶ │   FastAPI    │ ◀─────────── │   Airflow   │
│   :8501     │         │    :8000     │              │    :8080    │
└─────────────┘         └──────┬───────┘              └──────┬──────┘
                               │                             │
                               │ Prometheus /metrics         │ orchestrates
                               ▼                             ▼
                        ┌─────────────┐            ┌────────────────┐
                        │ Prometheus  │            │  ingest →      │
                        │   :9090     │            │  drift_check → │
                        └──────┬──────┘            │  preprocess →  │
                               │                   │  train         │
                               ▼                   └────────┬───────┘
                        ┌─────────────┐                     │ logs runs
                        │   Grafana   │                     ▼
                        │   :3000     │            ┌─────────────────┐
                        │ 25 panels   │            │ MLflow Tracking │
                        └─────────────┘            │ + Registry :5000│
                                                   └────────┬────────┘
                                                            │
                                              ┌─────────────┴───────┐
                                              ▼                     ▼
                                  ┌────────────────────┐   ┌──────────────┐
                                  │ MLflow Models Serve│   │ DVC + DagsHub│
                                  │     :5001          │   │ remote (data │
                                  │ POST /invocations  │   │  + models)   │
                                  └────────────────────┘   └──────────────┘
```

Detailed mermaid diagram in [`docs/architecture_diagram.md`](docs/architecture_diagram.md).

### Key design decisions

| Decision | Rationale |
|---|---|
| **Two inference paths** | FastAPI (`:8000`) for the full feature set (predictions + feedback + drift + admin); MLflow Models Serve (`:5001`) for the protocol-clean MLflow-native endpoint |
| **drift_check before preprocessing** | Preprocessing always overwrites baselines from the current training data. Running drift_check first ensures the comparison is honest: new data vs the previously-deployed model's expected distribution |
| **Hot-reload over restart** | Airflow's `reload_api_baselines` and `reload_api_model` tasks POST to API admin endpoints; no container restart needed when a new model trains |
| **DVC outside Docker** | DVC is a build-time tool, not a runtime service. Running it on the dev machine avoids coupling Git credentials into the runtime stack |
| **Dedicated mlflow-serve image** | The registered model was logged with `mlflow 2.13.2 / scikit-learn 1.4.2`; the serving env must replicate the training env exactly to avoid wrapper / cloudpickle drift |

---

## Tech stack

| Concern | Tool | Role |
|---|---|---|
| **Frontend** | Streamlit | Multi-page UI: predict, pipeline, monitoring, manual |
| **Backend** | FastAPI + Pydantic | Inference API, feedback loop, admin endpoints |
| **Orchestration** | Apache Airflow | DAG: ingest → drift → preprocess → train |
| **Tracking + Registry** | MLflow | Experiment tracking, model versioning, parallel serving |
| **Versioning** | DVC + DagsHub | Data + model versioning tied to Git commits |
| **Monitoring** | Prometheus + Grafana | 25-panel NRT dashboard + alert rules |
| **Persistence** | SQLite (feedback) + Postgres (Airflow) | Lightweight feedback store + Airflow metadata |
| **Containerization** | Docker Compose | Single-host multi-service deployment |
| **CI/CD** | GitHub Actions | Lint, tests, DVC validation, Docker build |
| **Email** | smtplib | Drift, retrain, accuracy-degradation alerts |
| **Testing** | pytest | 42 unit + integration tests |

---

## Performance

| Metric | Value |
|---|---|
| **F1 score** | **0.906** |
| **ROC-AUC** | **0.977** |
| **Accuracy** | 0.994 |
| **Precision** | 0.967 |
| **Recall** | 0.853 |
| Inference latency (p99) | < 200 ms |
| Test pass rate | 42 / 42 (100 %) |
| Docker stack size | 9 services, ~3 GB total images |

Algorithm: RandomForest with grid search across `n_estimators × max_depth × min_samples_split`, logged via MLflow.

---

## Quick start

### Prerequisites

- **Docker Desktop** 24+ (Windows / macOS / Linux)
- **Git** with credentials configured for DagsHub
- **Python 3.11** + a virtualenv (only for tests + DVC operations)
- **8 GB RAM** free, **20 GB disk** free

### 1. Clone

```bash
git clone https://dagshub.com/AdityaThakur-DA25M004/predictive-maintenance-system-mlops-pipeline.git
cd predictive-maintenance-system-mlops-pipeline
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env and set: RETRAIN_API_KEY, AIRFLOW_ADMIN_*, AIRFLOW__CORE__FERNET_KEY,
# (optional) MLFLOW_TRACKING_*, SMTP_*, *_PORT
```

Generate a Fernet key for Airflow:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### 3. Pull data + models from DagsHub

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate    # Linux/Mac

pip install -r requirements.txt
python -m dvc pull
```

### 4. Bring the stack up

```powershell
# PowerShell — set both compose files for the session
$env:COMPOSE_FILE = "docker-compose.yml;docker-compose.mlflow-serve.yml"

docker compose up -d --build
docker compose ps    # wait until all show "healthy"
```

### 5. Verify endpoints

| Service | URL | Notes |
|---|---|---|
| Streamlit dashboard | http://localhost:8501 | Main UI |
| FastAPI Swagger UI | http://localhost:8000/docs | Auto-generated API docs |
| MLflow UI | http://localhost:5000 | Experiments + registry |
| MLflow Models Serve | http://localhost:5001/invocations | POST only — protocol-clean |
| Airflow | http://localhost:8080 | Login from `.env` |
| Prometheus | http://localhost:9090 | Targets + alerts |
| Grafana | http://localhost:3000 | admin / admin (change first) |

### 6. End-to-end smoke test

```powershell
# Health
Invoke-RestMethod http://localhost:8000/health

# Single prediction
$body = @{
    type = "M"
    air_temperature = 298.1
    process_temperature = 308.6
    rotational_speed = 1551
    torque = 42.8
    tool_wear = 108
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType "application/json" -Body $body

# Trigger retrain (requires API key from .env)
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/retrain?reason=manual" `
  -Headers @{ "X-API-Key" = "<your-key>" }
```

---

## Service map

| Container | Image | Internal port | External port | Purpose |
|---|---|---|---|---|
| `pm-postgres` | `postgres:15` | 5432 | `${POSTGRES_PORT}` | Airflow metadata DB |
| `pm-mlflow` | custom | 5000 | 5000 | MLflow tracking + registry |
| `pm-api` | custom | 8000 | `${API_PORT}` | FastAPI inference + admin |
| `pm-frontend` | custom | 8501 | `${FRONTEND_PORT}` | Streamlit UI |
| `pm-airflow-init` | custom | — | — | One-shot DB init |
| `pm-airflow-web` | custom | 8080 | `${AIRFLOW_PORT}` | Airflow webserver |
| `pm-airflow-scheduler` | custom | — | — | Airflow scheduler |
| `pm-prometheus` | `prom/prometheus` | 9090 | `${PROMETHEUS_PORT}` | Metrics scrape + alert rules |
| `pm-grafana` | `grafana/grafana` | 3000 | `${GRAFANA_PORT}` | NRT dashboard |
| `pm-mlflow-serve` *(optional)* | custom (pinned) | 5001 | 5001 | MLflow-native scoring endpoint |

All connected via the `pm-net` bridge network.

---

## ML pipeline

The pipeline runs in two complementary orchestrators that share the same logical stages:

### Airflow DAG (runtime)

```
data_ingestion
   └─→ drift_check          (against OLD baselines on disk)
        └─→ preprocessing   (writes NEW baselines)
             └─→ reload_api_baselines    (hot-reload into running API)
                  └─→ model_training
                       └─→ reload_api_model     (hot-reload into running API)
                            └─→ drift_branch
                                 ├─→ retrain_notification → pipeline_end
                                 └─→ no_drift_end → pipeline_end
```

**Triggers:**
- **Scheduled** — daily
- **API-driven** — `POST /retrain/upload` triggers an ad-hoc DAG run with a freshly uploaded CSV
- **Manual** — Airflow UI

### DVC pipeline (build-time)

`dvc.yaml` mirrors the same four logical stages for reproducibility:

```bash
data_ingestion → drift_check → preprocessing → training
```

```powershell
python -m dvc repro                # run full pipeline
python -m dvc dag                  # render DAG
python -m dvc metrics show         # current metric values
python -m dvc metrics diff HEAD~1  # compare to previous commit
```

### What each stage does

| Stage | Module | Inputs | Outputs |
|---|---|---|---|
| **data_ingestion** | `src/data_ingestion.py` | `data/raw/ai4i2020.csv`, latest user upload | `train.csv`, `test.csv` (leaky cols dropped, stratified split) |
| **drift_check** | `src/drift_detection.py` | `test.csv`, existing baselines | `drift_report.json` (full), `drift_summary.json` (flat metric) |
| **preprocessing** | `src/data_preprocessing.py` | `train.csv`, `test.csv` | engineered features, scaler, baselines, ref samples |
| **training** | `src/model_training.py` | processed splits, config | `best_model.joblib`, `test_metrics.json`, MLflow run + registered version |

### Drift detection mechanics

- **KS test** per feature against persisted reference samples (`ref_samples.json` written by preprocessing).
- **PSI** per feature; threshold `0.2` flags drift.
- A feature is "drifted" if both KS p-value < 0.05 AND PSI > 0.2.
- Overall drift flagged when ≥ 1 feature drifts (configurable in `configs/config.yaml`).

---

## Repository layout

```
predictive-maintenance-system-mlops-pipeline/
├── .github/workflows/
│   ├── ci.yml                      # 6-job CI pipeline
│   └── README.md                   # CI setup + secrets configuration
├── airflow/dags/
│   └── ml_pipeline_dag.py          # Airflow DAG
├── api/
│   ├── main.py                     # FastAPI app — 15+ endpoints
│   ├── schemas.py                  # Pydantic models
│   └── metrics.py                  # Prometheus metric definitions
├── configs/
│   ├── config.yaml                 # Default (Docker)
│   └── config.local.yaml           # Local dev override
├── data/
│   ├── raw/ai4i2020.csv            # Input dataset (DVC-tracked)
│   ├── processed/                  # train.csv, test.csv (DVC outs)
│   ├── baselines/                  # drift baselines + reports + summaries
│   └── feedback/                   # uploads/ + feedback.db (runtime)
├── docker/
│   ├── api/, frontend/, airflow/, mlflow/
│   ├── mlflow-serve/Dockerfile     # Pinned env for MLflow models serve
│   ├── trainer/, dvc/
│   └── requirements/               # Per-image requirements
├── docs/
│   ├── HLD.md                      # High-level design
│   ├── LLD.md                      # Low-level design + API spec
│   ├── architecture_diagram.md
│   ├── test_plan.md
│   └── screenshots/                # Visual evidence (see below)
├── frontend/
│   ├── app.py                      # Main dashboard
│   ├── common.py                   # API client + shared CSS
│   └── pages/
│       ├── 1_Predict.py
│       ├── 2_Pipeline.py
│       ├── 3_Monitoring.py
│       └── 4_User_Manual.py
├── monitoring/
│   ├── prometheus/                 # scrape config + alert rules
│   └── grafana/predictive_maintenance.json    # 25-panel dashboard
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── drift_detection.py
│   ├── model_training.py
│   ├── alert_notifier.py
│   └── utils.py
├── tests/
│   ├── test_api.py                 # 18 integration tests
│   ├── test_data_ingestion.py      # 11 unit tests
│   ├── test_preprocessing.py       # 6 unit tests
│   ├── test_drift.py               # 4 unit tests
│   ├── test_model.py               # 3 unit tests
│   └── fixtures/heavily_drifted_data.csv
├── docker-compose.yml
├── docker-compose.mlflow-serve.yml
├── dvc.yaml                        # DVC pipeline definition
├── dvc.lock                        # Pinned input/output hashes
├── MLproject                       # MLflow project entry points
├── pytest.ini
├── requirements.txt
└── README.md                       # This file
```

---

## Configuration

### Layered config

| File | Used when | Selected by |
|---|---|---|
| `configs/config.yaml` | Inside Docker | Default |
| `configs/config.local.yaml` | Local dev | `PM_CONFIG_PATH` env var |

Both are tracked by Git so changes to either trigger DVC re-runs.

### Key environment variables

| Variable | Purpose |
|---|---|
| `RETRAIN_API_KEY` | Auth for `/retrain*`, `/admin/*`, `/rollback` |
| `AIRFLOW_ADMIN_USERNAME` / `AIRFLOW_ADMIN_PASSWORD` | Airflow web UI + REST API auth |
| `AIRFLOW__CORE__FERNET_KEY` | Airflow encryption key (required) |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | DagsHub registry creds (optional) |
| `SMTP_*`, `ALERT_EMAIL_TO` | Email alert delivery (optional) |
| `*_PORT` | Host-side port mappings |

Full surface in `.env.example`.

---

## Testing

42 tests covering data ingestion, preprocessing, drift detection, model evaluation, and the API surface — 100% pass rate. Detailed coverage map: [`docs/test_plan.md`](docs/test_plan.md).

```powershell
# All tests
pytest -v

# With coverage
pytest --cov=src --cov=api --cov-report=html

# Inside Docker
docker compose exec api pytest -v
```

`pytest.ini` scopes collection to `tests/` to avoid pytest walking into Airflow's `latest` symlink (which Windows can't follow).

---

## CI/CD

GitHub Actions workflow at [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and PR:

```
lint (ruff) ─┬─ test (pytest) ─┬─ docker-build ──┐
             │                  │                 ├── ci-summary
             ├─ dvc-validate ──┤                 │
             │                  │                 │
             └──────────────────┴─ dvc-repro ────┘
                                   (gated)
```

| Job | Purpose | Always runs? |
|---|---|---|
| `lint` | ruff style + import order | ✅ |
| `test` | pytest + JUnit XML upload | ✅ |
| `dvc-validate` | `dvc dag` + `dvc status` | ✅ |
| `dvc-repro` | Full pipeline reproduction | Conditional on secrets |
| `docker-build` | `docker compose build` all images | ✅ |
| `ci-summary` | Fan-in success gate | ✅ |

Setup details in [`.github/workflows/README.md`](.github/workflows/README.md).

---

## Versioning and reproducibility

### DVC + DagsHub

```powershell
python -m dvc pull                     # fetch data + models for current commit
python -m dvc push                     # upload after local changes
python -m dvc dag                      # render pipeline DAG
python -m dvc metrics show             # current metrics
python -m dvc metrics diff HEAD~1      # diff vs previous commit
python -m dvc metrics diff v2.1 v2.2   # diff between tags
```

### Git tags

| Tag | Marks |
|---|---|
| `v1.0` | Initial production-ready build |
| `v2.0` | Drift detection + retraining loop |
| `v2.1` | Drift branch fix, MLflow serve sidecar, design docs |
| `v2.2` | GitHub Actions CI workflow |
| `metrics-baseline` | Reference metrics on default dataset |
| `metrics-drifted` | Metrics after retraining on drifted dataset |

### MLflow registry

Every training run registers a new version of `predictive-maintenance-model`. Latest version is what `mlflow-serve` and the FastAPI hot-reload pull from. Browse at http://localhost:5000.

### MLproject

```powershell
mlflow run . -e ingest        --env-manager=local
mlflow run . -e preprocess    --env-manager=local
mlflow run . -e drift_check   --env-manager=local
mlflow run . -e train         --env-manager=local
mlflow run . --env-manager=local                   # default = `dvc repro`
```

---

## Monitoring and alerts

### Prometheus metrics

| Metric | Type | Description |
|---|---|---|
| `prediction_requests_total` | Counter | Per-class prediction count |
| `prediction_latency_seconds` | Histogram | End-to-end inference latency |
| `drift_active` | Gauge | 1 if drift detected, 0 otherwise |
| `retrain_triggers_total` | Counter | Number of retrain invocations |
| `feedback_accuracy_rolling` | Gauge | Rolling accuracy from feedback |
| `model_version_info` | Info | Current model version + algorithm |
| `training_duration_seconds` | Histogram | Per-run training wall-clock |

### Grafana dashboard — 25 panels

1. **System Health** — uptime, request rate, error rate
2. **Predictions** — class distribution, latency p50/p95/p99
3. **Model Quality** — F1, ROC-AUC, precision, recall (from feedback)
4. **Drift** — drift_active gauge, per-feature PSI, drift events timeline
5. **Pipeline** — training duration, retrain triggers, last run status
6. **Infrastructure** — CPU, memory, container status

Provisioned automatically from `monitoring/grafana/predictive_maintenance.json` on first start.

### Alert rules

`monitoring/prometheus/alert_rules.yml` defines:

- `MultipleFeaturesDrifted` — ≥ 3 features drifted simultaneously (critical)
- `RetrainStorm` — > 5 retrains in 15 minutes (warning)
- `LowFeedbackAccuracy` — rolling accuracy < 0.8 (warning)
- `HighErrorRate` — API 5xx rate > 5% (critical)
- `APIDown` — `/health` failing for > 2 minutes (critical)

Alerts fire to email via `src/alert_notifier.py`.

---

## Working system — visual evidence

Screenshots demonstrating each component is operational. Captured during a clean end-to-end run; all images live in [`docs/screenshots/`](docs/screenshots/).

### 1. Streamlit frontend (port 8501)

| | |
|---|---|
| ![Dashboard](docs/screenshots/01_frontend/00_dashboard.png) | ![Predict single](docs/screenshots/01_frontend/01_predict_single.png) |
| **Main dashboard** — hero stats, status pills, performance gauges, pipeline timeline | **Predict page** — single sensor reading with failure probability |
| ![Pipeline upload](docs/screenshots/01_frontend/02_predict_batch.png) | ![Drift detection](docs/screenshots/01_frontend/04_monitoring_drift.png) |
| **Pipeline page** — dataset upload triggers Airflow DAG | **Monitoring page** — drift detected in 09/10 features |
| ![System alerts](docs/screenshots/01_frontend/05_monitoring_alerts.png) | ![User manual](docs/screenshots/01_frontend/06_user_manual.png) |
| **Alerts tab** — Prometheus alert rules currently firing | **User manual** — non-technical onboarding guide |

### 2. FastAPI backend (port 8000)

| | |
|---|---|
| ![Swagger UI](docs/screenshots/02_api/01_swagger_docs.png) | ![Health check](docs/screenshots/02_api/02_health.png) |
| **Auto-generated OpenAPI docs** — every endpoint discoverable | **Health endpoint** — model + scaler loaded, uptime visible |
| ![Model info](docs/screenshots/02_api/03_model_info.png) | ![Predict response](docs/screenshots/02_api/04_predict_response.png) |
| **Model info** — registered version + test metrics | **Live prediction** — request and response payload |

### 3. MLflow tracking and registry (port 5000)

| | |
|---|---|
| ![Experiments list](docs/screenshots/03_mlflow/01_experiments.png) | ![Run detail](docs/screenshots/03_mlflow/02_run_detail.png) |
| **Experiments** — all training runs with metrics | **Run detail** — params, metrics, artifacts |
| ![Metric chart](docs/screenshots/03_mlflow/03_metric_chart.png) | ![Model registry](docs/screenshots/03_mlflow/04_registry_versions.png) |
| **F1 across runs** — grid search results | **Registry** — versioned models, ready for serving |

### 4. Airflow orchestration (port 8080)

![Airflow Graph view — drift fix verified](docs/screenshots/04_airflow/03_graph_drift_detected.png)

> **The drift branch fix, proven.** After uploading a drifted dataset, `retrain_notification` correctly fires (green) and `no_drift_end` is correctly skipped (pink). Earlier the colors were inverted because `drift_check` ran *after* preprocessing overwrote baselines — fixed by reordering the DAG so `drift_check` compares against the **previous** baselines.

| | |
|---|---|
| ![DAG list](docs/screenshots/04_airflow/01_dag_list.png) | ![Normal run](docs/screenshots/04_airflow/02_graph_success.png) |
| **DAG home** — `predictive_maintenance_pipeline` registered | **Normal run** — all stages green, no drift detected |

### 5. Prometheus (port 9090)

| | |
|---|---|
| ![Targets](docs/screenshots/05_prometheus/01_targets.png)<br>**All targets UP** — scrape configuration healthy | ![Model Degraded Email](docs/screenshots/05_prometheus/03_model_degraded_email.png)<br>**Model Degraded Alert** — email notification triggered |
| ![Alert rules](docs/screenshots/05_prometheus/01_Alerts_rules.png)<br>**Alert Rules** — drift, retrain storm, error rate, API down | ![Firing alert](docs/screenshots/05_prometheus/04_multiple_alerts_firing.png)<br>**Multiple Alerts Firing** — active alerts during demo |

### 6. Grafana (port 3000) — 25-panel dashboard

![Grafana overview](docs/screenshots/06_grafana/01_grafana_dashboard.png)

| | |
|---|---|
| ![Drift panels](docs/screenshots/06_grafana/02_heavily_degraded_model.png) | ![Predictions panels](docs/screenshots/06_grafana/03_training_data.png) |
| **Drift panels** — per-feature PSI + drift event timeline | **Retraining-Activity** |


> Full 25-panel dashboard covering system health, predictions, model quality, drift, pipeline events, and infrastructure.

### 7. DVC pipeline

| | |
|---|---|
| ![DVC DAG](docs/screenshots/07_dvc/01_dvc_dag.png) | ![DVC status](docs/screenshots/07_dvc/02_dvc_status.png) |
| **4-stage DAG** — ingest → drift_check → preprocess → train | **Stage status** — clean / changed indicators |
| ![Metrics show](docs/screenshots/07_dvc/03_dvc_metrics.png) | ![Metrics diff](docs/screenshots/07_dvc/05_dvc_metrics_diff.png) |
| **`dvc metrics show`** — current run summary | **`dvc metrics diff`** — comparison across commits |

> The metrics diff is the reproducibility proof: F1 went from 0.906 (baseline) to lower (drifted) — exactly the kind of regression a model registry should surface.

### 8. MLflow Models Serve sidecar (port 5001)

| | |
|---|---|
| ![Version](docs/screenshots/08_mlflow_serve/01_version.png) | ![Ping](docs/screenshots/08_mlflow_serve/02_ping.png) |
| **`/version`** — `2.13.2` confirms pinned env match | **`/ping`** — `200 OK` (empty body per spec) |
| ![Invocations](docs/screenshots/08_mlflow_serve/03_invocations.png) | ![Clean logs](docs/screenshots/08_mlflow_serve/04_clean_logs.png) |
| **`POST /invocations`** — `{"predictions": [0]}` | **Clean boot logs** — no version mismatches, no worker timeouts |

> Dedicated container with version-pinned dependencies (`mlflow 2.13.2`, `scikit-learn 1.4.2`) matching the registered model. `GUNICORN_CMD_ARGS=--timeout=300` prevents the default worker recycling under healthcheck load.

### 9. DagsHub — code, data, experiments

| | |
|---|---|
| ![Repo home](docs/screenshots/09_dagshub/01_repo_home.png) | ![Tags](docs/screenshots/09_dagshub/02_tags.png) |
| **Repo on DagsHub** — clean structure with all artifacts | **Release tags** — v1.0 → v2.2 + metric tags |
| ![DVC data preview](docs/screenshots/09_dagshub/03_dvc_data_preview.png) | ![Commits](docs/screenshots/09_dagshub/04_commit_history.png) |
| **DVC data preview** — DagsHub renders CSVs natively | **Commit history** — atomic, well-described commits |

![DagsHub MLflow remote](docs/screenshots/09_dagshub/05_mlflow_remote.png)

> Hosted MLflow registry on DagsHub mirrors the local one — proves the registry pattern works against an external remote.

### 10. Tests + CI

| | |
|---|---|
| ![Pytest 42 passed](docs/screenshots/10_tests_ci/01_pytest.png) | ![Pytest summary](docs/screenshots/10_tests_ci/02_pytest_summary.png) |
| **All 42 tests pass** | **Concise summary** for the test report |
| ![Coverage](docs/screenshots/10_tests_ci/03_coverage.png)  |
| **Code coverage** — `src/` and `api/` | **CI run** — green across lint, test, dvc-validate, docker-build |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `docker compose down` leaves `mlflow-serve` running | Override file not loaded | `$env:COMPOSE_FILE = "docker-compose.yml;docker-compose.mlflow-serve.yml"` |
| Airflow `retrain_notification` skipped on drifted upload | DAG ordering regression | Confirm `ml_pipeline_dag.py` runs `drift_check` BEFORE `preprocess` |
| `pytest` collection fails on `airflow/logs/scheduler/latest` | Windows symlink unfollowable | Confirm `pytest.ini` has `testpaths = tests` |
| `mlflow models serve` worker timeouts | Default `--timeout=60` too aggressive | `GUNICORN_CMD_ARGS=--timeout=300` already set in compose override |
| `dvc dag` complains "already specified in stage" | Same file in `outs:` and `metrics:` | Use separate files for outs vs metrics |
| `dvc pull` 401 | DagsHub creds not configured for DVC | `dvc remote modify origin --local user/password` |
| Streamlit container "unhealthy" | Internal port mismatch | Confirm internal port hardcoded to 8501 |
| BuildKit "parent snapshot does not exist" | Build cache corrupted | `docker builder prune -af` |
| `dvc metrics diff HEAD~1` shows `-` for HEAD~1 | Metric files not Git-tracked at HEAD~1 | Confirm `.gitignore` doesn't block `*test_metrics.json` / `drift_summary.json` |

For more diagnostic help, run:

```powershell
.\diagnose.ps1
```

---

## Documentation index

| Document | Purpose |
|---|---|
| [`docs/HLD.md`](docs/HLD.md) | High-level design — components, technology rationale, failure modes |
| [`docs/LLD.md`](docs/LLD.md) | Low-level design — full API spec, schemas, DB schema, deployment |
| [`docs/architecture_diagram.md`](docs/architecture_diagram.md) | Mermaid block diagram + ML pipeline DAG |
| [`docs/test_plan.md`](docs/test_plan.md) | Test strategy, coverage map, acceptance criteria |
| [`.github/workflows/README.md`](.github/workflows/README.md) | CI workflow setup + secrets configuration |

---

## Acknowledgements

- **Dataset** — [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (UCI ML Repository)
- **MLOps tooling** — Airflow, MLflow, DVC, DagsHub, Prometheus, Grafana
- **Course** — DA25M004, MLOps Assignment, IIT Madras

---

*Built with care to demonstrate end-to-end MLOps best practices: reproducibility, observability, automation, and testability.*