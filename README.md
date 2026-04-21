# 🏭 Predictive Maintenance System

An end-to-end MLOps-powered system for predicting industrial equipment failures from sensor data.

## Overview

This system uses machine learning to predict machine failures before they occur, analyzing real-time sensor readings (temperature, rotational speed, torque, tool wear) through an intuitive web dashboard.

**Dataset**: AI4I 2020 Predictive Maintenance Dataset (10,000 records, 5 failure types)

## Architecture

| Service | Port | Technology |
|---------|------|------------|
| Frontend Dashboard | 8501 | Streamlit |
| Backend API | 8000 | FastAPI |
| MLflow Tracking | 5000 | MLflow |
| Airflow Orchestration | 8080 | Apache Airflow |
| Prometheus Metrics | 9090 | Prometheus |
| Grafana Dashboards | 3000 | Grafana |

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Git

### Setup & Run

```bash
# Clone the repository
git clone <repo-url>
cd predictive-maintenance

# Initialize DVC
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"

# Run the ML pipeline locally first
pip install -r requirements.txt
python -m src.data_ingestion
python -m src.data_preprocessing
python -m src.model_training

# Start all services
docker compose up --build -d

# Check service health
docker compose ps
curl http://localhost:8000/health
```

### Access Points
- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Airflow UI**: http://localhost:8080 (admin/admin)
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## DVC Pipeline

```bash
# View the pipeline DAG
dvc dag

# Reproduce the full pipeline
dvc repro

# Push data/models to DagsHub remote
dvc remote add -d dagshub <dagshub-url>
dvc push
```

## API Usage

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"air_temperature": 300.0, "process_temperature": 310.0,
       "rotational_speed": 1500, "torque": 40.0,
       "tool_wear": 100, "product_type": "L"}'

# Drift check
curl -X POST http://localhost:8000/drift/check \
  -H "Content-Type: application/json" \
  -d '{"readings": [...]}'
```

## Testing

```bash
pytest tests/ -v --tb=short
pytest tests/ -v --cov=src --cov=api
```

## Project Structure

```
predictive-maintenance/
├── api/                 # FastAPI backend
├── frontend/            # Streamlit UI
├── src/                 # Core ML modules
├── airflow/dags/        # Airflow pipeline DAG
├── monitoring/          # Prometheus & Grafana config
├── tests/               # pytest test suite
├── docs/                # Design docs & test plan
├── data/                # Raw & processed data
├── models/              # Trained model artifacts
├── configs/             # YAML configuration
├── docker-compose.yml   # Multi-service orchestration
└── dvc.yaml             # DVC pipeline stages
```

## Documentation

- [Architecture & HLD](docs/architecture_diagram.md)
- [Low-Level Design](docs/LLD.md)
- [Test Plan](docs/test_plan.md)
- [User Manual](frontend/pages/4_User_Manual.py) (also accessible in the dashboard)

## MLOps Features

- **Experiment Tracking**: All training runs logged to MLflow (params, metrics, artifacts)
- **Model Registry**: Best model registered in MLflow Model Registry
- **Data Versioning**: DVC tracks raw data and model artifacts
- **CI Pipeline**: DVC DAG defines reproducible pipeline stages
- **Monitoring**: Prometheus metrics + Grafana dashboards
- **Alerting**: Alert rules for error rate > 5%, drift, high latency
- **Drift Detection**: KS-test + PSI on all features vs. training baselines
- **Containerization**: Docker Compose with 8 services
- **Pipeline Orchestration**: Airflow DAG with branching on drift
