# Operations Runbook — Predictive Maintenance System

> **Audience.** Whoever deploys, operates, or demos the system. Follow
> this in order for a fresh install; use the troubleshooting section
> for recurring ops.
>
> **Project version.** v2.x — incorporates drift_check ordering fix,
> dedicated mlflow-serve sidecar, hot-reload admin endpoints, MLproject
> entry points, and CI/CD workflow.

---

## 1. Prerequisites

| Tool | Version | Why |
|---|---|---|
| Docker Engine | ≥ 24.0 | Runs all services |
| Docker Compose | v2 (`docker compose`) | Orchestrates the stack |
| Git | ≥ 2.30 | Source control |
| Python | 3.11 | DVC pipeline + tests outside containers |
| Free RAM | ≥ 8 GB | 9 containers run concurrently |
| Free disk | ≥ 20 GB | Docker images + volumes |

### Ports used on host (configurable in `.env`)

| Port | Service | `.env` variable |
|---|---|---|
| 8000 | FastAPI backend | `API_PORT` |
| 8501 | Streamlit frontend | `FRONTEND_PORT` |
| 5000 | MLflow tracking + registry | (hardcoded in compose) |
| 5001 | MLflow Models Serve sidecar | (hardcoded; optional) |
| 8080 | Airflow webserver | `AIRFLOW_PORT` |
| 9090 | Prometheus | `PROMETHEUS_PORT` |
| 3000 | Grafana | `GRAFANA_PORT` |
| 5432 | PostgreSQL | `POSTGRES_PORT` |

> **Note on Streamlit port.** The frontend container's internal port
> is hardcoded to **8501** to avoid conflicts with Docker's default
> port 3000. Set `FRONTEND_PORT=8501` in `.env` for matching host:container.

Verify Docker:
```bash
docker --version          # ≥ 24.0
docker compose version    # v2.x
```

---

## 2. First-Time Setup

### 2.1 Clone and enter the repo

```bash
git clone https://dagshub.com/<user>/<repo>.git predictive-maintenance
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
3. Set `MLFLOW_TRACKING_URI` to your repo's MLflow URL (or leave default for local container)

> ⚠️ **Never commit `.env`.** Verify: `git check-ignore .env` should print `.env`.

### 2.3 Place the raw dataset

```bash
mkdir -p data/raw data/processed data/baselines data/feedback/uploads models
cp /path/to/ai4i2020.csv data/raw/
```

### 2.4 Pull data from DVC / DagsHub (recommended)

If the repo's data is already pushed to DagsHub:

```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
# .\venv\Scripts\Activate.ps1      # Windows PowerShell

pip install -r requirements.txt

# Configure DVC remote auth (stored in .dvc/config.local — gitignored)
python -m dvc remote modify origin --local auth basic
python -m dvc remote modify origin --local user "$DAGSHUB_USERNAME"
python -m dvc remote modify origin --local password "$DAGSHUB_TOKEN"

python -m dvc pull
```

This fetches `data/raw/ai4i2020.csv`, the trained model, scaler,
baselines, and reference samples — everything DVC tracks.

### 2.5 Set the compose file env (for the mlflow-serve sidecar)

The optional `mlflow-serve` sidecar is defined in a separate override
file. Set this env var so every `docker compose` command in the session
loads both:

**Linux/Mac (bash):**
```bash
export COMPOSE_FILE="docker-compose.yml:docker-compose.mlflow-serve.yml"
```

**Windows PowerShell:**
```powershell
$env:COMPOSE_FILE = "docker-compose.yml;docker-compose.mlflow-serve.yml"
```

If you skip this, the main stack still works — only the mlflow-serve
sidecar will be unreachable.

### 2.6 Build the stack

```bash
docker compose build
```

5–10 minutes on first run (cached after that).

### 2.7 Start services

```bash
# Start everything
docker compose up -d

# Watch logs during startup
docker compose logs -f --tail=50
```

The `airflow-init` container runs once (creates DB + admin user) then
exits — this is **expected**, not a failure.

### 2.8 Verify all services

```bash
docker compose ps
```

You should see 9+ containers (8 main + optional mlflow-serve). Smoke-test
each:

```bash
curl -sf http://localhost:8000/health             # FastAPI
curl -sf http://localhost:8501/_stcore/health     # Streamlit
curl -sf http://localhost:5000/                   # MLflow tracking
curl -sf http://localhost:5001/ping               # MLflow Models Serve (200 with empty body)
curl -sf http://localhost:9090/-/healthy          # Prometheus
curl -sf http://localhost:3000/api/health         # Grafana
curl -sf http://localhost:8080/health             # Airflow
```

---

## 3. Train the First Model

### Path A — Via DVC pipeline (recommended)

DVC drives the canonical reproducible pipeline:

```bash
source venv/bin/activate
python -m dvc repro          # Runs all 4 stages: ingest → drift_check → preprocess → train
python -m dvc push           # Push artifacts to DagsHub remote
```

After successful training, the API picks up the new model automatically
on its next admin reload (or restart):

```bash
# Manually hot-reload the API (no container restart needed)
curl -X POST http://localhost:8000/admin/reload-baselines \
  -H "X-API-Key: $RETRAIN_API_KEY"

curl -X POST "http://localhost:8000/admin/reload-model?data_source=default" \
  -H "X-API-Key: $RETRAIN_API_KEY"
```

### Path B — Via Airflow DAG

1. Open http://localhost:8080 (credentials from `.env`)
2. Enable the DAG `predictive_maintenance_pipeline`
3. Click **Trigger DAG**
4. Watch the task graph:
   `data_ingestion → drift_check → preprocessing → reload_api_baselines
   → model_training → reload_api_model → drift_branch
   → (retrain_notification | no_drift_end) → pipeline_end`
5. The `reload_api_*` tasks automatically hot-reload the API after each
   successful stage. **No manual restart needed.**

### Path C — Via MLflow Projects

The `MLproject` file defines named entry points that wrap each pipeline
stage. Useful for running individual stages from another orchestrator:

```bash
mlflow run . -e ingest        --env-manager=local
mlflow run . -e drift_check   --env-manager=local
mlflow run . -e preprocess    --env-manager=local
mlflow run . -e train         --env-manager=local
mlflow run . --env-manager=local                    # default: dvc repro
```

### Verify the model loaded

```bash
curl -s http://localhost:8000/model/info | python -m json.tool
# Should show: "model_loaded": true, "f1_score": > 0.8, model_version: <number>
```

---

## 4. Daily Operations

### 4.1 Access the UIs

| Service | URL | Credentials |
|---|---|---|
| Streamlit dashboard | http://localhost:8501 | none |
| FastAPI Swagger UI | http://localhost:8000/docs | none (admin endpoints need API key) |
| MLflow tracking + registry | http://localhost:5000 | DagsHub token for hosted remote |
| MLflow Models Serve | http://localhost:5001/invocations | none — POST only |
| Airflow webserver | http://localhost:8080 | `AIRFLOW_ADMIN_*` from `.env` |
| Grafana dashboard | http://localhost:3000 | `GRAFANA_ADMIN_*` from `.env` |
| Prometheus | http://localhost:9090 | none |

### 4.2 Submit a prediction (CLI)

**FastAPI path:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "type": "M",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 108
  }' | python -m json.tool
```

**MLflow Models Serve path** (protocol-clean alternative):
```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]","temp_diff","power","wear_degree","type_encoded","speed_torque_ratio"],
      "data": [[298.1, 308.6, 1551, 42.8, 108, 10.5, 6912, 4622, 1, 36.2]]
    }
  }'
```

Note: the MLflow path requires pre-engineered features in the request,
since it talks to the raw model. The FastAPI path engineers them
internally.

### 4.3 Submit feedback

```bash
# Use the prediction_id from a previous /predict response
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"prediction_id": 1, "actual_label": 0}'
```

### 4.4 Check feedback accuracy

```bash
curl -s http://localhost:8000/feedback/stats | python -m json.tool
```

### 4.5 Trigger a retrain

```bash
# Manual retrain (uses existing data source)
curl -X POST "http://localhost:8000/retrain?reason=manual" \
  -H "X-API-Key: $RETRAIN_API_KEY"

# Upload + retrain (multipart CSV upload)
curl -X POST "http://localhost:8000/retrain/upload" \
  -H "X-API-Key: $RETRAIN_API_KEY" \
  -F "file=@/path/to/new_data.csv" \
  -F "reason=quarterly_refresh"
```

### 4.6 Run drift check (CLI)

```bash
curl -X POST http://localhost:8000/drift/check \
  -H "Content-Type: application/json" \
  -d '{"readings": [
    {"type":"M","air_temperature":300,"process_temperature":310,"rotational_speed":1500,"torque":40,"tool_wear":100},
    {"type":"L","air_temperature":301,"process_temperature":311,"rotational_speed":1400,"torque":45,"tool_wear":120}
  ]}' | python -m json.tool
```

### 4.7 Roll back to a previous model version

```bash
# List available versions in the registry first (open MLflow UI or):
curl -s http://localhost:8000/model/info | python -m json.tool

# Roll back
curl -X POST "http://localhost:8000/rollback?target_version=5" \
  -H "X-API-Key: $RETRAIN_API_KEY"
```

### 4.8 View logs

```bash
docker compose logs -f api                  # Follow API logs
docker compose logs --since 10m mlflow      # Last 10 min of MLflow
docker compose logs airflow-scheduler       # Scheduler logs
docker compose logs -f mlflow-serve         # MLflow Models Serve sidecar
```

### 4.9 Check Prometheus metrics

```bash
# Raw metrics from API
curl -s http://localhost:8000/metrics | head -30

# Active alerts
curl -s http://localhost:9090/api/v1/alerts | python -m json.tool

# Specific metric
curl -s 'http://localhost:9090/api/v1/query?query=drift_active'
```

### 4.10 Inspect DVC metrics

```bash
python -m dvc metrics show               # current state
python -m dvc metrics diff HEAD~1        # vs previous commit
python -m dvc metrics diff v2.1 v2.2     # vs specific tag
```

---

## 5. Common Operations

### 5.1 Restart one service

```bash
docker compose restart api
docker compose restart frontend
docker compose restart mlflow-serve
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

> **Important:** if you started with the override file (`COMPOSE_FILE`
> set), `docker compose down` automatically tears down the
> `mlflow-serve` sidecar too. If you didn't set the env var, the
> sidecar is left running. Tear it down explicitly if needed:
> `docker stop pm-mlflow-serve && docker rm pm-mlflow-serve`.

### 5.4 Run tests

```bash
# Local (recommended)
source venv/bin/activate
pytest -v --tb=short

# With coverage
pytest --cov=src --cov=api --cov-report=term-missing

# Inside Docker (if API container has tests in image)
docker compose exec api pytest -v
```

`pytest.ini` scopes collection to `tests/` to avoid the collector
walking into `airflow/logs/scheduler/latest` (Windows symlink that
the OS can't follow).

### 5.5 Run CI locally before pushing

```bash
# Lint
ruff check src api airflow/dags frontend tests --select E,F,W,I --ignore E501

# Tests
pytest -v

# DVC validation
python -m dvc dag
python -m dvc status

# Docker validation
docker compose -f docker-compose.yml config > /dev/null
docker compose -f docker-compose.yml -f docker-compose.mlflow-serve.yml config > /dev/null
docker compose build --parallel
```

If all pass locally, CI on push will pass too.

### 5.6 Back up volumes

```bash
mkdir -p backups
for vol in mlflow_db mlflow_artifacts postgres_data feedback_data grafana_data prometheus_data; do
  docker run --rm \
    -v "predictive-maintenance_${vol}:/data" \
    -v "$PWD/backups:/backup" \
    alpine tar czf "/backup/${vol}_$(date +%F).tgz" -C /data .
done
```

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `docker compose up` fails: "port already in use" | Another process uses the port | Change the port in `.env` or: `lsof -i :8000` then kill |
| API shows `model_loaded: false` | `models/best_model.joblib` missing | Run the training pipeline (§3) or `dvc pull` |
| Streamlit "API offline" | API container crashing | `docker compose logs api` — look for tracebacks |
| Streamlit shows port 3000 conflict | `FRONTEND_PORT` collides with another service | Set `FRONTEND_PORT=8501` in `.env` (matches internal) |
| Airflow webserver restart-loops | Fernet key missing or malformed | Regenerate `AIRFLOW__CORE__FERNET_KEY` (§2.2) |
| Airflow `reload_api_baselines` fails with `TypeError: missing 'params'` | Stale DAG file | Confirm DAG has `params: dict \| None = None` default |
| Airflow `retrain_notification` skipped on drifted upload | DAG ordering regression | Confirm DAG runs `drift_check` BEFORE `preprocess` |
| `/retrain` returns 401 | `X-API-Key` header missing or wrong | Must match `RETRAIN_API_KEY` exactly |
| `/retrain` returns 200 with `status: partial` | `AIRFLOW_USERNAME`/`PASSWORD` not set | Expected when Airflow REST API auth not configured; set the vars to enable trigger |
| MLflow shows 401 on push | DagsHub token expired | Rotate at DagsHub → update `.env` → restart |
| MLflow Models Serve container fails to start | Model not registered or version mismatch | Run a training job first; check version pin in `docker/mlflow-serve/Dockerfile` |
| MLflow Models Serve worker timeouts | Default `--timeout=60` too aggressive | `GUNICORN_CMD_ARGS=--timeout=300` already set in compose override |
| `docker compose down` leaves `mlflow-serve` running | Override file not loaded | Set `COMPOSE_FILE` env var (§2.5) before running down |
| Grafana shows "No data" | Prometheus can't reach `api:8000` | `docker network inspect predictive-maintenance_pm-net` — confirm both attached |
| `dvc dag` "already specified in stage" | Same file in `outs:` and `metrics:` | Use separate files for outs vs metrics |
| `dvc repro` "specified in two places" | Stale `.dvc` file from old `dvc add` | `python -m dvc remove <file>.dvc` |
| `dvc metrics diff HEAD~1` shows `-` | Metric files were gitignored at HEAD~1 | Confirm `.gitignore` doesn't block `test_metrics.json` / `drift_summary.json` |
| `pytest` collection fails on `airflow/logs/scheduler/latest` | Windows symlink unfollowable | Confirm `pytest.ini` has `testpaths = tests` |
| BuildKit "parent snapshot does not exist" | Build cache corrupted from interrupted build | `docker builder prune -af` |
| Build fails: `"https:// "` error | Space after `=` in `.env` | Remove all trailing spaces from env values |
| `permission denied` on volume | Host dir owned by root | `sudo chown -R 1000:1000 ./data ./models` |
| Airflow DAG import error | `PYTHONPATH` not set | Confirm `PYTHONPATH=/opt/airflow:/app` in compose |
| `airflow-init` exits with code 0 | **Normal** — it's a one-shot init | Not an error |
| Compose warns "variable not set" | `.env` missing a required var | `grep -oP '\$\{\K[A-Z_]+' docker-compose.yml \| sort -u` to find missing |

### 6.1 Diagnostic script

If something looks wrong, run the bundled diagnostic:

```bash
./diagnose.ps1     # Windows PowerShell
# or
bash diagnose.sh   # if a shell version exists
```

Outputs a snapshot of container status, recent logs, port bindings, and
environment variable presence.

### 6.2 Hard reset (dev only — destroys all data)

```bash
docker compose down -v
docker system prune -f
rm -rf data/feedback/*.db data/processed/* data/baselines/*.json models/*.joblib models/*.json
docker compose build --no-cache
docker compose up -d
```

---

## 7. Port Mapping Quick Reference

Based on the default `.env`:

```
Browser → http://localhost:8501   →  Streamlit  (container :8501)
Browser → http://localhost:8000   →  FastAPI    (container :8000)
Browser → http://localhost:5000   →  MLflow     (container :5000)
Browser → http://localhost:5001   →  MLflow Serve (container :5001) [optional]
Browser → http://localhost:8080   →  Airflow    (container :8080)
Browser → http://localhost:9090   →  Prometheus (container :9090)
Browser → http://localhost:3000   →  Grafana    (container :3000)
```

---

## 8. Environment Variable Reference

Variables marked **[R]** are required — compose will fail without them.

| Variable | Used by | Default | Notes |
|---|---|---|---|
| `API_PORT` **[R]** | compose | 8000 | Host port for FastAPI |
| `RETRAIN_API_KEY` **[R]** | api | — | Auth for `/retrain*`, `/admin/*`, `/rollback` |
| `FRONTEND_PORT` **[R]** | compose | 8501 | Host port for Streamlit (must equal internal 8501) |
| `POSTGRES_USER` **[R]** | compose | airflow | Postgres superuser |
| `POSTGRES_PASSWORD` **[R]** | compose | — | Postgres password |
| `POSTGRES_DB` **[R]** | compose | airflow | Postgres database name |
| `POSTGRES_PORT` **[R]** | compose | 5432 | Host port for Postgres |
| `AIRFLOW__CORE__EXECUTOR` **[R]** | airflow | LocalExecutor | Or `SequentialExecutor` (no Postgres) |
| `AIRFLOW__CORE__FERNET_KEY` **[R]** | airflow | — | Encryption key; generate with Fernet |
| `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` **[R]** | airflow | postgresql+psycopg2://... | DB connection string |
| `AIRFLOW_ADMIN_USERNAME` **[R]** | airflow-init | admin | WebUI login |
| `AIRFLOW_ADMIN_PASSWORD` **[R]** | airflow-init | — | WebUI password |
| `AIRFLOW_USERNAME` | api | — | For triggering DAG runs from API (optional) |
| `AIRFLOW_PASSWORD` | api | — | For triggering DAG runs from API (optional) |
| `AIRFLOW_PORT` **[R]** | compose | 8080 | Host port for Airflow |
| `GRAFANA_ADMIN_USER` **[R]** | grafana | admin | Dashboard login |
| `GRAFANA_ADMIN_PASSWORD` **[R]** | grafana | — | Dashboard password |
| `GRAFANA_PORT` **[R]** | compose | 3000 | Host port for Grafana |
| `PROMETHEUS_PORT` **[R]** | compose | 9090 | Host port for Prometheus |
| `MLFLOW_TRACKING_USERNAME` | mlflow, airflow | — | DagsHub username (for hosted registry) |
| `MLFLOW_TRACKING_PASSWORD` | mlflow, airflow | — | DagsHub token |
| `MLFLOW_TRACKING_URI` | mlflow, api, airflow | http://mlflow:5000 | Local container by default |
| `DAGSHUB_USERNAME` | DVC | — | For DVC pull/push |
| `DAGSHUB_TOKEN` | DVC | — | For DVC pull/push |
| `SMTP_HOST`, `SMTP_PORT` | alert_notifier | — | For email alerts (optional) |
| `SMTP_USERNAME`, `SMTP_PASSWORD` | alert_notifier | — | SMTP auth (optional) |
| `ALERT_EMAIL_TO` | alert_notifier | — | Recipient(s), comma-separated (optional) |

---

## 9. Where Things Live

```
predictive-maintenance-system-mlops-pipeline/
├── .github/workflows/ci.yml      # 6-job CI pipeline
├── airflow/
│   ├── dags/ml_pipeline_dag.py   # 8-task DAG (incl. drift_check + reload_api_*)
│   └── logs/                     # Runtime logs (gitignored)
├── api/
│   ├── main.py                   # FastAPI app — 15+ endpoints incl. /admin/*
│   ├── schemas.py                # Pydantic models
│   └── metrics.py                # Prometheus metric definitions
├── configs/
│   ├── config.yaml               # Default (Docker)
│   └── config.local.yaml         # Local dev override (PM_CONFIG_PATH)
├── data/
│   ├── raw/                      # Input dataset (DVC-tracked)
│   ├── processed/                # train.csv, test.csv (DVC outs)
│   ├── baselines/                # drift_baselines, ref_samples, drift_report,
│   │                             #   drift_summary (DVC metric)
│   └── feedback/
│       ├── uploads/              # User-uploaded CSVs (Git-tracked file list)
│       └── feedback.db           # SQLite (gitignored, runtime state)
├── docker/
│   ├── api/Dockerfile
│   ├── frontend/Dockerfile
│   ├── airflow/Dockerfile
│   ├── mlflow/Dockerfile
│   ├── mlflow-serve/Dockerfile   # Pinned env: mlflow 2.13.2, sklearn 1.4.2
│   ├── trainer/Dockerfile        # Optional: containerized training image
│   ├── dvc/Dockerfile            # Optional: containerized DVC operations
│   └── requirements/             # Per-image requirements files
├── docs/
│   ├── HLD.md, LLD.md
│   ├── architecture_diagram.md
│   ├── test_plan.md
│   ├── USER_MANUAL.md            # Non-technical user guide
│   └── screenshots/              # Visual evidence per component
├── frontend/
│   ├── app.py                    # Main dashboard (redesigned)
│   ├── common.py                 # API client + shared CSS
│   └── pages/
│       ├── 1_Predict.py
│       ├── 2_Pipeline.py
│       ├── 3_Monitoring.py
│       └── 4_User_Manual.py
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── alert_rules.yml       # MultipleFeaturesDrifted, RetrainStorm, etc.
│   └── grafana/
│       └── predictive_maintenance.json   # 25-panel dashboard
├── scripts/
│   ├── activate-local.ps1
│   └── dvc_init.sh
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── drift_detection.py        # __main__ block for DVC stage
│   ├── model_training.py
│   ├── alert_notifier.py         # SMTP email helpers
│   └── utils.py
├── tests/
│   ├── test_api.py               # 18 integration tests
│   ├── test_data_ingestion.py    # 11 unit tests
│   ├── test_preprocessing.py     # 6 unit tests
│   ├── test_drift.py             # 4 unit tests
│   ├── test_model.py             # 3 unit tests
│   └── fixtures/
│       └── heavily_drifted_data.csv
├── docker-compose.yml            # Main 8-service stack
├── docker-compose.mlflow-serve.yml  # mlflow-serve sidecar override
├── dvc.yaml                      # 4-stage DVC pipeline
├── dvc.lock                      # Pinned input/output hashes
├── MLproject                     # MLflow run entry points
├── pytest.ini                    # testpaths = tests
├── requirements.txt
├── diagnose.ps1                  # Operational diagnostic
└── README.md
```

---

## 10. Clean Shutdown

```bash
docker compose stop              # Pause containers (preserves state)
docker compose down              # Remove containers (volumes preserved)
docker compose down -v           # Remove containers AND volumes (destructive)
```

If you started with the `mlflow-serve` sidecar via the override:

```bash
# Both files loaded via COMPOSE_FILE env (recommended) → down handles both
docker compose down

# Otherwise, explicitly:
docker compose -f docker-compose.yml -f docker-compose.mlflow-serve.yml down
```

---

## 11. Release & Tagging

When you've made meaningful changes:

```bash
# Stage the changes
git add -A

# Commit with a descriptive message
git commit -m "<type>: <summary>"

# Tag the release
git tag -a v2.x -m "v2.x: <release description>"

# Push code and tag
git push origin main
git push origin v2.x

# Push DVC artifacts
python -m dvc push
```

Recent tags worth knowing:
- `v2.0` — drift detection + retraining loop
- `v2.1` — drift_check ordering fix, mlflow-serve sidecar, design docs
- `v2.2` — GitHub Actions CI workflow

---

## 12. Quick Sanity Checks

A 30-second health check before any demo:

```bash
docker compose ps                                          # all containers up
curl -sf http://localhost:8000/health                      # API alive
curl -sf http://localhost:8000/model/info                  # model loaded
curl -sf http://localhost:8501/_stcore/health              # frontend alive
curl -sf http://localhost:5001/ping                        # mlflow-serve alive
docker compose logs --tail 5 airflow-scheduler             # no error spam
python -m dvc metrics show                                 # baselines + model metrics present
```

If all six pass, the demo is ready to roll.