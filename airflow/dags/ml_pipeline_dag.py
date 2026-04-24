"""
Airflow DAG for the Predictive Maintenance ML Pipeline.

Orchestrates: ingest → preprocess → train → drift check → branch
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import sys
import os
import json
import logging

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/app")
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "start_date": datetime(2024, 1, 1),
}


def task_data_ingestion(**kwargs):
    from src.data_ingestion import run_ingestion
    result = run_ingestion()
    logger.info(
        f"Ingestion result: {result['status']} | "
        f"source={result.get('data_source', 'unknown')} | "
        f"path={result.get('raw_path', 'unknown')}"
    )
    kwargs["ti"].xcom_push(key="ingestion_result", value=json.dumps({
        "status": result["status"],
        "data_source": result.get("data_source", "default"),
        "raw_path": result.get("raw_path", ""),
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "quality_status": result["quality_report"]["quality_status"],
    }))
    return result["status"]


def task_preprocessing(**kwargs):
    from src.data_preprocessing import run_preprocessing
    result = run_preprocessing()
    logger.info(f"Preprocessing result: {result['status']}")
    kwargs["ti"].xcom_push(key="preprocessing_result", value=json.dumps({
        "status": result["status"],
        "num_features": result["num_features"],
    }))
    return result["status"]


def task_model_training(**kwargs):
    import time as _time
    from src.model_training import run_training

    t0 = _time.time()
    result = run_training()
    duration_s = _time.time() - t0

    logger.info(
        f"Training result: {result['status']} | F1={result['best_f1']:.4f} | "
        f"data_source={result.get('data_source', 'unknown')} | "
        f"duration={duration_s:.0f}s"
    )

    # Send training-complete alert (email/log) — non-fatal
    try:
        from src.alert_notifier import send_training_complete_alert
        send_training_complete_alert(
            new_f1=result["best_f1"],
            model_version=str(result["model_version"]),
            run_id=result["best_run_id"],
            data_source=result.get("data_source", "unknown"),
            duration_seconds=duration_s,
        )
    except Exception as e:
        logger.warning(f"Training-complete alert failed (non-fatal): {e}")

    kwargs["ti"].xcom_push(key="training_result", value=json.dumps({
        "status": result["status"],
        "best_f1": result["best_f1"],
        "best_run_id": result["best_run_id"],
        "model_version": str(result["model_version"]),
        "data_source": result.get("data_source", "unknown"),
        "duration_seconds": round(duration_s, 1),
    }))
    return result["status"]


def task_drift_check(**kwargs):
    import pandas as pd
    from src.data_preprocessing import engineer_features, get_feature_columns
    from src.drift_detection import detect_drift
    from src.utils import load_config, get_project_root, load_json

    config = load_config()
    root = get_project_root()
    data_path = os.path.join(str(root), config["data"]["processed_dir"], "test.csv")
    baselines_path = os.path.join(str(root), config["data"]["baselines_path"])

    df = pd.read_csv(data_path)
    df = engineer_features(df)
    feature_cols = get_feature_columns()
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
    ti = kwargs["ti"]
    drift_json = ti.xcom_pull(key="drift_result", task_ids="drift_check")
    if not drift_json:
        logger.warning("drift_result XCom is empty — defaulting to no_drift_end")
        return "no_drift_end"
    drift = json.loads(drift_json)
    if drift["overall_drift"]:
        logger.warning("Drift detected — branching to retraining path")
        return "retrain_notification"
    else:
        logger.info("No drift — pipeline complete")
        return "no_drift_end"


def task_retrain_notification(**kwargs):
    ti = kwargs["ti"]
    drift_json = ti.xcom_pull(key="drift_result", task_ids="drift_check")
    if not drift_json:
        logger.warning("drift_result XCom empty in retrain_notification — skipping alerts")
        return "retrain_triggered"
    drift = json.loads(drift_json)
    logger.warning(f"RETRAINING RECOMMENDED: Drift in features: {drift['drifted_features']}")

    # Alert 1: drift detected
    try:
        from src.alert_notifier import send_drift_alert
        send_drift_alert(drift["drifted_features"], drift["n_drifted"])
    except Exception as e:
        logger.warning(f"Drift alert failed (non-fatal): {e}")

    # Alert 2: retraining triggered (from Airflow)
    try:
        from src.alert_notifier import send_retrain_alert
        send_retrain_alert(
            reason="drift_detected",
            triggered_by="airflow",
            data_source="unknown",   # will be known after ingestion runs
        )
    except Exception as e:
        logger.warning(f"Retrain alert failed (non-fatal): {e}")

    return "retrain_triggered"


with DAG(
    dag_id="predictive_maintenance_pipeline",
    default_args=default_args,
    description="End-to-end ML pipeline for predictive maintenance",
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=["ml", "predictive-maintenance", "mlops"],
) as dag:

    # NOTE: provide_context is removed — it's automatic in Airflow 2.x
    ingest = PythonOperator(
        task_id="data_ingestion",
        python_callable=task_data_ingestion,
    )

    preprocess = PythonOperator(
        task_id="preprocessing",
        python_callable=task_preprocessing,
    )

    train = PythonOperator(
        task_id="model_training",
        python_callable=task_model_training,
    )

    drift = PythonOperator(
        task_id="drift_check",
        python_callable=task_drift_check,
    )

    branch = BranchPythonOperator(
        task_id="drift_branch",
        python_callable=branch_on_drift,
    )

    retrain_notify = PythonOperator(
        task_id="retrain_notification",
        python_callable=task_retrain_notification,
    )

    no_drift = EmptyOperator(task_id="no_drift_end")

    pipeline_end = EmptyOperator(
        task_id="pipeline_end",
        trigger_rule="none_failed_min_one_success",
    )

    ingest >> preprocess >> train >> drift >> branch
    branch >> [retrain_notify, no_drift]
    [retrain_notify, no_drift] >> pipeline_end