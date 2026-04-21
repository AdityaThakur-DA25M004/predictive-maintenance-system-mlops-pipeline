"""
Airflow DAG for the Predictive Maintenance ML Pipeline.

Orchestrates the end-to-end workflow:
  1. Data Ingestion → Validate → Split
  2. Feature Engineering → Drift Baselines
  3. Model Training → Evaluation → Registration
  4. Drift Detection Check
  5. Conditional Retraining Trigger
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import sys
import os
import json
import logging

# Add project root to path
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/app")
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default DAG arguments
# ---------------------------------------------------------------------------
default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "start_date": datetime(2024, 1, 1),
}


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------
def task_data_ingestion(**kwargs):
    """Run data ingestion pipeline."""
    from src.data_ingestion import run_ingestion
    result = run_ingestion()
    logger.info(f"Ingestion result: {result['status']}")
    # Push result to XCom for downstream tasks
    kwargs["ti"].xcom_push(key="ingestion_result", value=json.dumps({
        "status": result["status"],
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "quality_status": result["quality_report"]["quality_status"],
    }))
    return result["status"]


def task_preprocessing(**kwargs):
    """Run feature engineering and scaling pipeline."""
    from src.data_preprocessing import run_preprocessing
    result = run_preprocessing()
    logger.info(f"Preprocessing result: {result['status']}")
    kwargs["ti"].xcom_push(key="preprocessing_result", value=json.dumps({
        "status": result["status"],
        "num_features": result["num_features"],
    }))
    return result["status"]


def task_model_training(**kwargs):
    """Run model training with MLflow tracking."""
    from src.model_training import run_training
    result = run_training()
    logger.info(f"Training result: {result['status']} | F1={result['best_f1']:.4f}")
    kwargs["ti"].xcom_push(key="training_result", value=json.dumps({
        "status": result["status"],
        "best_f1": result["best_f1"],
        "best_run_id": result["best_run_id"],
        "model_version": str(result["model_version"]),
    }))
    return result["status"]


def task_drift_check(**kwargs):
    """Check for data drift on test set against baselines."""
    from src.drift_detection import check_drift_from_file
    from src.data_preprocessing import get_feature_columns
    from src.utils import get_project_root, load_config

    config = load_config()
    root = get_project_root()
    data_path = str(root / config["data"]["processed_dir"] / "test.csv")
    baselines_path = str(root / config["data"]["baselines_path"])

    # Need to engineer features on test data first
    import pandas as pd
    from src.data_preprocessing import engineer_features
    df = pd.read_csv(data_path)
    df = engineer_features(df)

    feature_cols = get_feature_columns()
    from src.drift_detection import detect_drift
    from src.utils import load_json
    baselines = load_json(baselines_path)
    report = detect_drift(df, baselines, feature_cols)

    logger.info(f"Drift check: {report['n_drifted']} features drifted")
    kwargs["ti"].xcom_push(key="drift_result", value=json.dumps({
        "overall_drift": report["overall_drift"],
        "n_drifted": report["n_drifted"],
        "drifted_features": report["drifted_features"],
    }))
    return report


def branch_on_drift(**kwargs):
    """Branch based on drift detection result."""
    ti = kwargs["ti"]
    drift_json = ti.xcom_pull(key="drift_result", task_ids="drift_check")
    drift = json.loads(drift_json)

    if drift["overall_drift"]:
        logger.warning("Drift detected — branching to retraining path")
        return "retrain_notification"
    else:
        logger.info("No drift — pipeline complete")
        return "no_drift_end"


def task_retrain_notification(**kwargs):
    """Log retraining notification (in production, would trigger alerts)."""
    ti = kwargs["ti"]
    drift_json = ti.xcom_pull(key="drift_result", task_ids="drift_check")
    drift = json.loads(drift_json)
    logger.warning(
        f"RETRAINING RECOMMENDED: Drift detected in features: "
        f"{drift['drifted_features']}"
    )
    return "retrain_triggered"


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="predictive_maintenance_pipeline",
    default_args=default_args,
    description="End-to-end ML pipeline for predictive maintenance",
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=["ml", "predictive-maintenance", "mlops"],
    doc_md="""
    ## Predictive Maintenance ML Pipeline

    This DAG runs the full ML lifecycle:
    1. **Data Ingestion**: Load, validate, and split data.
    2. **Preprocessing**: Feature engineering and scaling.
    3. **Training**: Model training with MLflow experiment tracking.
    4. **Drift Check**: Compare current data against baselines.
    5. **Branch**: Trigger retraining if drift is detected.
    """,
) as dag:

    # Tasks
    ingest = PythonOperator(
        task_id="data_ingestion",
        python_callable=task_data_ingestion,
        provide_context=True,
    )

    preprocess = PythonOperator(
        task_id="preprocessing",
        python_callable=task_preprocessing,
        provide_context=True,
    )

    train = PythonOperator(
        task_id="model_training",
        python_callable=task_model_training,
        provide_context=True,
    )

    drift = PythonOperator(
        task_id="drift_check",
        python_callable=task_drift_check,
        provide_context=True,
    )

    branch = BranchPythonOperator(
        task_id="drift_branch",
        python_callable=branch_on_drift,
        provide_context=True,
    )

    retrain_notify = PythonOperator(
        task_id="retrain_notification",
        python_callable=task_retrain_notification,
        provide_context=True,
    )

    no_drift = EmptyOperator(task_id="no_drift_end")

    pipeline_end = EmptyOperator(
        task_id="pipeline_end",
        trigger_rule="none_failed_min_one_success",
    )

    # Dependencies
    ingest >> preprocess >> train >> drift >> branch
    branch >> [retrain_notify, no_drift]
    [retrain_notify, no_drift] >> pipeline_end
