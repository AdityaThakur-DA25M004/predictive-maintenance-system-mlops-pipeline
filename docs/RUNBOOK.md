# Operations Runbook — Predictive Maintenance System

> **Audience.** Whoever deploys, operates, or demos the system.
> Follow this in order for a fresh install; use the troubleshooting section for recurring ops.

---

## 1. Prerequisites

| Tool | Version | Why |
|------|---------|-----|
| Docker Engine | ≥ 24.0 | Runs all services |
| Docker Compose | v2 (`docker compose`) | Orchestrates the stack |
| Git | ≥ 2.30 | Source control |
| Python | 3.11 (optional, only for local training) | Run the DVC pipeline outside containers |
| Free RAM | ≥ 6 GB | 8 containers run concurrently |
| Free disk | ≥ 10 GB | Docker images + volumes |

**Ports used on host** (configurable in `.env`):

| Port | Service | .env variable |
|------|---------|---------------|
| 8000 | FastAPI backend | `API_PORT` |
| 3000 | Streamlit frontend | `FRONTEND_PORT` |
| 5000 | MLflow | (hardcoded in compose) |
| 8080 | Airflow webserver | `AIRFLOW_PORT` |
| 9090 | Prometheus | `PROMETHEUS_PORT` |
| 3001 | Grafana | `GRAFANA_PORT` |
| 5432 | PostgreSQL | `POSTGRES_PORT` |

Verify Docker:
```bash
docker --version          # ≥ 24.0
docker compose version    # v2.x
```

---

## 2. First-Time Setup

### 2.1 Clone and enter the repo

```bash
git clone <your-repo-url> predictive-maintenance
cd predictive-maintenance
```

### 2.2 Create the `.env` file

```bash
cp .env.example .env
```

Open `.env` and replace every `CHANGE_ME` placeholder. Generate secrets:

```bash
# RETRAIN_API_KEY & SECRET_KEY
python -c "import secrets; print(secrets.token_hex(32))"

# Airflow Fernet key (REQUIRED — Airflow won't start without it)
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Fill in your **DagsHub** credentials:
1. Generate a token at https://dagshub.com/user/settings/tokens
2. Set `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`
3. Set `MLFLOW_TRACKING_URI` to your repo's MLflow URL

> ⚠️ **Never commit `.env`.** Verify: `git check-ignore .env` should print `.env`.

### 2.3 Place the raw dataset

```bash
mkdir -p data/raw data/processed data/baselines data/feedback models
cp /path/to/ai4i2020.csv data/raw/
```

### 2.4 (Optional) Pull data from DVC / DagsHub

```bash
pip install dvc dvc-dagshub
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user "$DAGSHUB_USERNAME"
dvc remote modify dagshub --local password "$DAGSHUB_TOKEN"
dvc pull
```

### 2.5 Build the stack

```bash
docker compose build
```

This takes 5–10 minutes on first run (cached after that).

### 2.6 Start services

```bash
# Start everything
docker compose up -d

# Watch logs during startup
docker compose logs -f --tail=50
```

The `airflow-init` container runs once (creates DB + admin user) then exits — this is expected.

### 2.7 Verify all services

```bash
docker compose ps
```

You should see 8+ containers. Smoke-test each:

```bash
curl -sf http://localhost:8000/health          # FastAPI
curl -sf http://localhost:3000/_stcore/health   # Streamlit (port from .env)
curl -sf http://localhost:5000/                 # MLflow
curl -sf http://localhost:9090/-/healthy        # Prometheus
curl -sf http://localhost:3001/api/health       # Grafana (port from .env)
curl -sf http://localhost:8080/health           # Airflow
```

---

## 3. Train the First Model

### Path A — Local (recommended for first time)

```bash
pip install -r requirements.txt

# Step 1: Ingest data (validate, drop leaky cols, split)
python -m src.data_ingestion

# Step 2: Feature engineering + scaling
python -m src.data_preprocessing

# Step 3: Train with MLflow tracking
MLFLOW_TRACKING_URI=sqlite:///mlflow.db python -m src.model_training

# Step 4: Restart API to pick up the new model
docker compose restart api
```

Verify the model loaded:
```bash
curl -s http://localhost:8000/model/info | python -m json.tool
# Should show: "model_loaded": true, "f1_score": ~0.84
```

### Path B — Via Airflow DAG

1. Open http://localhost:8080 (credentials from `AIRFLOW_ADMIN_USERNAME` / `AIRFLOW_ADMIN_PASSWORD`)
2. Enable the DAG `predictive_maintenance_pipeline`
3. Click **Trigger DAG**
4. Watch tasks: ingestion → preprocessing → training → drift_check → branch
5. After success, restart API:

```bash
docker compose restart api
```

### Path C — Via DVC

```bash
dvc repro          # Runs all stages defined in dvc.yaml
dvc push           # Push artifacts to DagsHub remote
docker compose restart api
```

---

## 4. Daily Operations

### 4.1 Access the UIs

| Service | URL | Credentials |
|---------|-----|-------------|
| Streamlit dashboard | http://localhost:3000 | none |
| FastAPI docs (Swagger) | http://localhost:8000/docs | none (`/retrain` needs API key) |
| MLflow | http://localhost:5000 | DagsHub token for remote |
| Airflow | http://localhost:8080 | `AIRFLOW_ADMIN_*` from `.env` |
| Grafana | http://localhost:3001 | `GRAFANA_ADMIN_*` from `.env` |
| Prometheus | http://localhost:9090 | none |

### 4.2 Submit a prediction (CLI)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "air_temperature": 300.0,
    "process_temperature": 310.0,
    "rotational_speed": 1500,
    "torque": 40.0,
    "tool_wear": 100,
    "product_type": "L"
  }' | python -m json.tool
```

Note the `prediction_id` in the response — use it for feedback:

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"prediction_id": 1, "actual_label": 0}'
```

### 4.3 Check feedback accuracy

```bash
curl -s http://localhost:8000/feedback/stats | python -m json.tool
```

### 4.4 Trigger a manual retrain

```bash
# Use the RETRAIN_API_KEY from your .env
curl -X POST "http://localhost:8000/retrain?reason=scheduled" \
  -H "X-API-Key: $RETRAIN_API_KEY"
```

### 4.5 Run drift check (CLI)

```bash
curl -X POST http://localhost:8000/drift/check \
  -H "Content-Type: application/json" \
  -d '{"readings": [
    {"air_temperature":300,"process_temperature":310,"rotational_speed":1500,"torque":40,"tool_wear":100,"product_type":"L"},
    {"air_temperature":301,"process_temperature":311,"rotational_speed":1400,"torque":45,"tool_wear":120,"product_type":"M"}
  ]}' | python -m json.tool
```

### 4.6 View logs

```bash
docker compose logs -f api              # Follow API logs
docker compose logs --since 10m mlflow   # Last 10 min of MLflow
docker compose logs airflow-scheduler    # Scheduler logs
```

### 4.7 Check Prometheus metrics

```bash
# Raw metrics from API
curl -s http://localhost:8000/metrics | head -30

# Query Prometheus
open "http://localhost:9090/graph?g0.expr=drift_detected"

# All active alerts
curl -s http://localhost:9090/api/v1/alerts | python -m json.tool
```

---

## 5. Common Operations

### 5.1 Restart one service

```bash
docker compose restart api
docker compose restart frontend
```

### 5.2 Rebuild after code change

```bash
# Single service
docker compose build api && docker compose up -d api

# Everything
docker compose build && docker compose up -d
```

### 5.3 Stop the stack

```bash
docker compose down           # Stop containers (keeps volumes/data)
docker compose down -v        # WIPE all data (mlflow db, postgres, feedback)
```

### 5.4 Run tests

```bash
# Inside the project directory (local Python)
pip install -r requirements.txt
pytest tests/ -v --tb=short

# With coverage
pytest tests/ -v --cov=src --cov=api --cov-report=term
```

### 5.5 Back up volumes

```bash
mkdir -p backups
for vol in mlflow_db mlflow_artifacts postgres_data feedback_data grafana_data; do
  docker run --rm \
    -v "predictive-maintenance_${vol}:/data" \
    -v "$PWD/backups:/backup" \
    alpine tar czf "/backup/${vol}_$(date +%F).tgz" -C /data .
done
```

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `docker compose up` fails: "port already in use" | Another process uses the port | Change the port in `.env` or: `lsof -i :8000` then kill |
| API shows `model_loaded: false` | `models/best_model.joblib` missing | Run the training pipeline (§3) |
| Streamlit says "API offline" | API container crashing | `docker compose logs api` — look for tracebacks |
| Airflow webserver restart-loops | Fernet key missing or malformed | Regenerate `AIRFLOW__CORE__FERNET_KEY` (§2.2) |
| `/retrain` returns 401 | `X-API-Key` header missing or wrong | Must match `RETRAIN_API_KEY` in `.env` exactly |
| MLflow shows 401 on push | DagsHub token expired | Rotate at DagsHub → update `.env` → restart |
| Grafana shows "No data" | Prometheus can't reach `api:8000` | Verify both are on `pm-net`: `docker network inspect predictive-maintenance_pm-net` |
| Build fails: `"https:// "` error | Space after `=` in `.env` | Remove all trailing spaces from env values |
| `permission denied` on volume | Host dir owned by root | `sudo chown -R 1000:1000 ./data ./models` |
| Airflow DAG import error | `PYTHONPATH` not set | Confirm `PYTHONPATH=/opt/airflow:/app` in compose |
| `airflow-init` exits with code 0 | **Normal** — it's a one-shot init | Not an error; check `docker compose ps` for other services |
| Compose warns "variable not set" | `.env` missing a required var | Run: `grep -oP '\$\{\K[A-Z_]+' docker-compose.yml \| sort -u` and ensure all exist in `.env` |

### 6.1 Hard reset (dev only — destroys all data)

```bash
docker compose down -v
docker system prune -f
rm -rf data/feedback/* data/processed/* models/*.joblib models/*.json
docker compose build --no-cache
docker compose up -d
```

---

## 7. Port Mapping Quick Reference

Based on the default `.env`:

```
Browser → http://localhost:3000   →  Streamlit (container :8501)
Browser → http://localhost:8000   →  FastAPI   (container :8000)
Browser → http://localhost:5000   →  MLflow    (container :5000)
Browser → http://localhost:8080   →  Airflow   (container :8080)
Browser → http://localhost:9090   →  Prometheus(container :9090)
Browser → http://localhost:3001   →  Grafana   (container :3000)
```

---

## 8. Environment Variable Reference

Variables marked **[R]** are required — compose will fail without them.

| Variable | Used by | Default | Notes |
|----------|---------|---------|-------|
| `API_PORT` **[R]** | compose | 8000 | Host port for FastAPI |
| `RETRAIN_API_KEY` **[R]** | api | — | Auth header for `/retrain` |
| `FRONTEND_PORT` **[R]** | compose | 3000 | Host port for Streamlit |
| `POSTGRES_USER` **[R]** | compose | airflow | Postgres superuser |
| `POSTGRES_PASSWORD` **[R]** | compose | — | Postgres password |
| `POSTGRES_DB` **[R]** | compose | airflow | Postgres database name |
| `POSTGRES_PORT` **[R]** | compose | 5432 | Host port for Postgres |
| `AIRFLOW__CORE__EXECUTOR` **[R]** | airflow | SequentialExecutor | Or `LocalExecutor` (needs Postgres) |
| `AIRFLOW__CORE__FERNET_KEY` **[R]** | airflow | — | Encryption key; generate with Fernet |
| `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` **[R]** | airflow | sqlite:///... | Or `postgresql+psycopg2://...` |
| `AIRFLOW_ADMIN_USERNAME` **[R]** | airflow-init | admin | WebUI login |
| `AIRFLOW_ADMIN_PASSWORD` **[R]** | airflow-init | — | WebUI password |
| `AIRFLOW_PORT` **[R]** | compose | 8080 | Host port for Airflow |
| `GRAFANA_ADMIN_USER` **[R]** | grafana | admin | Dashboard login |
| `GRAFANA_ADMIN_PASSWORD` **[R]** | grafana | — | Dashboard password |
| `GRAFANA_PORT` **[R]** | compose | 3001 | Host port for Grafana |
| `PROMETHEUS_PORT` **[R]** | compose | 9090 | Host port for Prometheus |
| `MLFLOW_TRACKING_USERNAME` **[R]** | airflow | — | DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` **[R]** | airflow | — | DagsHub token |

---

## 9. Where Things Live

```
predictive-maintenance/
├── api/                         # FastAPI source (main.py, schemas.py, metrics.py)
├── frontend/                    # Streamlit source (app.py, common.py, pages/)
├── src/                         # Shared ML modules (ingestion, preprocessing, training, drift)
├── airflow/dags/                # Airflow DAG definition
├── configs/                     # YAML config
├── data/
│   ├── raw/                     # Input CSV (ai4i2020.csv)
│   ├── processed/               # Train/test splits (generated)
│   ├── baselines/               # Drift baselines (generated)
│   └── feedback/                # SQLite feedback store (Docker volume)
├── models/                      # Joblib artifacts (generated)
├── monitoring/
│   ├── prometheus/              # prometheus.yml + alert_rules.yml
│   └── grafana/
│       ├── dashboards/          # Dashboard JSON (24 panels)
│       └── provisioning/        # Auto-provisioning configs
├── docker/
│   ├── api/Dockerfile           # Multi-stage, non-root
│   ├── frontend/Dockerfile      # Streamlit, non-root
│   ├── airflow/Dockerfile       # Official Airflow base
│   ├── mlflow/Dockerfile        # Dedicated tracking server
│   └── requirements/            # Per-service pinned deps
├── tests/                       # pytest suite (42 test cases)
├── docs/                        # HLD, LLD, test plan, user manual
├── .env.example                 # Template (commit this)
├── .env                         # Real secrets (NEVER commit)
├── docker-compose.yml           # 8 services on pm-net
├── dvc.yaml                     # 3-stage DVC pipeline
├── MLproject                    # MLflow Projects entry points
└── python_env.yaml              # MLflow Python environment
```

---

## 10. Clean Shutdown

```bash
docker compose stop              # Pause containers (preserves state)
docker compose down              # Remove containers (volumes preserved)
docker compose down -v           # Remove containers AND volumes (destructive)
```
