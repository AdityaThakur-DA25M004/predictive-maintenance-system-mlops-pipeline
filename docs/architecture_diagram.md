# Architecture Diagram & High-Level Design

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DOCKER COMPOSE NETWORK                       │
│                                                                     │
│  ┌─────────────┐        ┌──────────────┐        ┌───────────────┐  │
│  │  Streamlit   │──REST──│   FastAPI     │──log───│    MLflow      │  │
│  │  Frontend    │  API   │   Backend    │  track │   Tracking     │  │
│  │  :8501       │        │   :8000      │        │   Server :5000 │  │
│  └─────────────┘        └──────┬───────┘        └───────────────┘  │
│                                │                                    │
│                         ┌──────┴───────┐                           │
│                         │  /metrics     │                           │
│                         └──────┬───────┘                           │
│                                │                                    │
│  ┌─────────────┐        ┌──────▼───────┐        ┌───────────────┐  │
│  │   Grafana    │◄─query─│  Prometheus   │        │   Airflow      │  │
│  │   :3000      │        │   :9090      │        │   :8080        │  │
│  └─────────────┘        └──────────────┘        └───────────────┘  │
│                                                         │           │
│                                                  ┌──────▼───────┐  │
│                                                  │  PostgreSQL   │  │
│                                                  │   :5432       │  │
│                                                  └──────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Shared Volumes                             │   │
│  │  data/raw  │  data/processed  │  models/  │  configs/        │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

External:
  ┌──────────┐       ┌──────────┐
  │  DagsHub  │       │   Git     │
  │  (DVC     │       │  (source  │
  │   remote) │       │   control)│
  └──────────┘       └──────────┘
```

## 2. Component Descriptions

### 2.1 Frontend (Streamlit)
- **Port**: 8501
- **Purpose**: User-facing web dashboard
- **Pages**: Dashboard, Predict, Pipeline, Monitoring, User Manual
- **Communication**: REST API calls to FastAPI backend
- **Design Choice**: Streamlit provides rapid prototyping with rich data visualization.
  Loose coupling via REST ensures the UI can be swapped independently.

### 2.2 Backend API (FastAPI)
- **Port**: 8000
- **Purpose**: Model serving, drift detection, metrics export
- **Endpoints**: /predict, /predict/batch, /drift/check, /health, /ready, /retrain, /metrics, /model/info
- **Design Choice**: FastAPI provides async support, auto-generated OpenAPI docs,
  Pydantic validation, and native compatibility with Prometheus.

### 2.3 MLflow Tracking Server
- **Port**: 5000
- **Purpose**: Experiment tracking, model registry, artifact storage
- **Backend Store**: SQLite (upgradeable to PostgreSQL)
- **Artifact Store**: Local filesystem (/mlflow/artifacts)
- **Design Choice**: MLflow is the industry standard for experiment tracking.
  Central server enables team collaboration.

### 2.4 Apache Airflow
- **Port**: 8080
- **Purpose**: Pipeline orchestration and scheduling
- **Executor**: LocalExecutor with PostgreSQL metadata DB
- **DAG**: predictive_maintenance_pipeline (ingest → preprocess → train → drift check → branch)
- **Design Choice**: Airflow provides DAG-based scheduling, retry logic,
  task dependencies, and a web UI for pipeline management.

### 2.5 Prometheus
- **Port**: 9090
- **Purpose**: Metrics collection and alerting
- **Scrape Targets**: FastAPI /metrics endpoint
- **Alert Rules**: High error rate (>5%), drift detected, high latency (p95>200ms)
- **Design Choice**: Prometheus pull-based model integrates cleanly with containerized services.

### 2.6 Grafana
- **Port**: 3000
- **Purpose**: Real-time dashboards and visualization
- **Datasource**: Prometheus (auto-provisioned)
- **Dashboard**: Auto-provisioned with panels for requests, latency, errors, drift, model performance
- **Design Choice**: Grafana provides customizable dashboards without code changes.

### 2.7 PostgreSQL
- **Port**: 5432
- **Purpose**: Airflow metadata database
- **Design Choice**: Production-grade database for Airflow's LocalExecutor.

## 3. Data Flow

```
Raw CSV → Data Ingestion → Validation → Split (train/test)
  → Feature Engineering → Drift Baselines
  → Model Training (MLflow tracking) → Evaluation → Registration
  → Deployment (FastAPI) → Monitoring (Prometheus/Grafana)
  → Drift Detection → [If drift] → Retrain Trigger
```

## 4. Design Rationale

| Decision | Rationale |
|----------|-----------|
| Microservices architecture | Loose coupling, independent scaling, separate concerns |
| REST API between frontend and backend | Language-agnostic, testable, swappable UI |
| Docker Compose | Reproducible environments, easy local development |
| DVC for data/model versioning | Git-like workflow for large binary artifacts |
| MLflow for experiment tracking | Industry standard, model registry, artifact logging |
| Prometheus + Grafana for monitoring | Pull-based metrics, customizable dashboards, alerting |
| RandomForest classifier | Handles imbalanced data, interpretable feature importance |
| Stratified splitting | Preserves class distribution in train/test sets |
| KS-test + PSI for drift detection | Statistical rigor, complementary approaches |
