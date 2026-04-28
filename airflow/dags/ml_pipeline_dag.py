"""
Airflow DAG for the Predictive Maintenance ML Pipeline.

Orchestrates: ingest → drift_check → preprocess → train → branch
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import sys
import os
import json
import logging
import pandas as pd
from src.data_preprocessing import engineer_features, get_feature_columns
from src.drift_detection import detect_drift
from src.utils import load_config, get_project_root, load_json
from src.alert_notifier import send_training_complete_alert
from src.data_ingestion import run_ingestion
from src.data_preprocessing import run_preprocessing
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


def task_drift_check(**kwargs):

    config = load_config()
    root = get_project_root()
    data_path = os.path.join(str(root), config["data"]["processed_dir"], "test.csv")
    baselines_path = os.path.join(str(root), config["data"]["baselines_path"])

    if not os.path.exists(baselines_path):
        logger.info(
            f"No baselines at {baselines_path} — first DAG run. "
            f"Skipping drift detection; preprocess will create baselines downstream."
        )
        kwargs["ti"].xcom_push(key="drift_result", value=json.dumps({
            "overall_drift": False,
            "n_drifted": 0,
            "drifted_features": [],
            "reference_type": "first_run_no_baseline",
        }))
        return {
            "overall_drift": False,
            "n_drifted": 0,
            "drifted_features": [],
            "reference_type": "first_run_no_baseline",
        }

    df = pd.read_csv(data_path)
    df = engineer_features(df)
    feature_cols = get_feature_columns()
    baselines = load_json(baselines_path)

    # pass baselines_path so detect_drift can find ref_samples.json (also pre-overwrite)
    report = detect_drift(
        df, baselines, feature_cols, baselines_path=baselines_path
    )

    logger.info(
        f"Drift check ({report.get('reference_type', 'unknown')}): "
        f"{report['n_drifted']} of {len(feature_cols)} features drifted | "
        f"overall_drift={report['overall_drift']}"
    )
    kwargs["ti"].xcom_push(key="drift_result", value=json.dumps({
        "overall_drift": report["overall_drift"],
        "n_drifted": report["n_drifted"],
        "drifted_features": report["drifted_features"],
        "reference_type": report.get("reference_type", "unknown"),
    }))
    return report


def task_preprocessing(**kwargs):
    
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

    try:
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


def _call_api_reload(endpoint: str, params: dict | None = None) -> str:
    import requests
    api_url = os.environ.get("API_INTERNAL_URL", "http://api:8000")
    api_key = os.environ.get("RETRAIN_API_KEY", "")
    if not api_key:
        logger.warning(f"RETRAIN_API_KEY not set in Airflow env — skipping {endpoint}")
        return "skipped"
    try:
        r = requests.post(
            f"{api_url}{endpoint}",
            headers={"X-API-Key": api_key},
            params=params or {},
            timeout=10,
        )
        if r.status_code == 200:
            logger.info(f"API {endpoint} OK: {r.json()}")
            return "reloaded"
        logger.warning(f"API {endpoint} non-200: {r.status_code} {r.text[:200]}")
        return f"non_200_{r.status_code}"
    except Exception as e:
        logger.warning(f"Could not reach API {endpoint} (non-fatal): {e}")
        return "unreachable"


def task_reload_api_baselines(**kwargs):
    """Hot-reload drift_baselines.json + ref_samples.json into the live API."""
    return _call_api_reload("/admin/reload-baselines")


def task_reload_api_model(**kwargs):
    """Hot-reload best_model.joblib + scaler.joblib + test_metrics.json into the live API."""
    import json as _json
    ti = kwargs["ti"]
    training_json = ti.xcom_pull(key="training_result", task_ids="model_training")
    data_source = "unknown"
    if training_json:
        try:
            data_source = _json.loads(training_json).get("data_source", "unknown")
        except Exception:
            pass
    return _call_api_reload(
        "/admin/reload-model",
        params={"data_source": data_source},
    )


def branch_on_drift(**kwargs):
    ti = kwargs["ti"]
    drift_json = ti.xcom_pull(key="drift_result", task_ids="drift_check")
    if not drift_json:
        logger.warning("drift_result XCom is empty — defaulting to no_drift_end")
        return "no_drift_end"
    drift = json.loads(drift_json)

    if drift["overall_drift"]:
        logger.warning(
            f"Drift detected in {drift['n_drifted']} features: "
            f"{drift['drifted_features']} — branching to retrain_notification"
        )
        return "retrain_notification"

    logger.info(
        f"No drift detected (reference_type={drift.get('reference_type', 'unknown')}) "
        f"— branching to no_drift_end"
    )
    return "no_drift_end"


def task_retrain_notification(**kwargs):
    """
    Send drift + retrain alerts. Uses the real data_source from ingestion
    XCom (uploaded vs default) so emails are accurate.
    """
    ti = kwargs["ti"]
    drift_json = ti.xcom_pull(key="drift_result", task_ids="drift_check")
    if not drift_json:
        logger.warning("drift_result XCom empty in retrain_notification — skipping alerts")
        return "retrain_triggered"
    drift = json.loads(drift_json)
    logger.warning(f"RETRAINING RECOMMENDED: Drift in features: {drift['drifted_features']}")

    # Pull the actual data source (uploaded vs default) for accurate alerting
    ingestion_json = ti.xcom_pull(key="ingestion_result", task_ids="data_ingestion")
    data_source = "unknown"
    if ingestion_json:
        try:
            data_source = json.loads(ingestion_json).get("data_source", "unknown")
        except Exception:
            pass

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
            data_source=data_source,
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

    ingest = PythonOperator(
        task_id="data_ingestion",
        python_callable=task_data_ingestion,
    )

    drift = PythonOperator(
        task_id="drift_check",
        python_callable=task_drift_check,
    )

    preprocess = PythonOperator(
        task_id="preprocessing",
        python_callable=task_preprocessing,
    )

    reload_api = PythonOperator(
        task_id="reload_api_baselines",
        python_callable=task_reload_api_baselines,
        retries=0,
    )

    train = PythonOperator(
        task_id="model_training",
        python_callable=task_model_training,
    )

    reload_api_model = PythonOperator(
        task_id="reload_api_model",
        python_callable=task_reload_api_model,
        retries=0,
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

    # NEW pipeline order:
    #   ingest → drift_check (against OLD baselines)
    #          → preprocess (writes NEW baselines)
    #          → reload_api_baselines (hot-reload baselines into live API)
    #          → train (on new data)
    #          → reload_api_model (hot-reload model so Grafana panels update)
    #          → drift_branch (uses drift_check XCom from earlier)
    ingest >> drift >> preprocess >> reload_api >> train >> reload_api_model >> branch
    branch >> [retrain_notify, no_drift]
    [retrain_notify, no_drift] >> pipeline_end