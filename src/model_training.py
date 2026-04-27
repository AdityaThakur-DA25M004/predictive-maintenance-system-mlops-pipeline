"""
Model Training Module with MLflow Experiment Tracking.

"""

import os
import sys
import time as _time
import subprocess
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, confusion_matrix,
    classification_report,
)
import joblib
import mlflow
import mlflow.sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import setup_logger, load_config, get_project_root, ensure_dir, save_json
from src.data_preprocessing import get_feature_columns

logger = setup_logger(__name__)

# Resolved at import time — same env var as data_ingestion and the API container.
UPLOADS_DIR = os.environ.get("UPLOADS_DIR", "data/feedback/uploads")


def _get_git_commit() -> str:
    """Return the current HEAD commit hash, or 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def get_latest_training_file(config: dict | None = None) -> str | None:
    """
    Return the path to the most-recently modified uploaded CSV, or None if
    no uploads exist.  Used by run_training() to decide whether the processed
    datasets need to be regenerated before model fitting.
    """
    dir_path = Path(UPLOADS_DIR)
    if not dir_path.exists():
        return None
    files = list(dir_path.glob("*.csv"))
    if not files:
        return None
    latest = max(files, key=lambda f: f.stat().st_mtime)
    logger.info(f"[UPLOAD] Latest upload found: {latest}")
    return str(latest)


# ---------------------------------------------------------------------------
# Model evaluation helper
# ---------------------------------------------------------------------------
def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Evaluate a trained model on the given data.

    Args:
        model: Trained sklearn estimator.
        X: Feature matrix.
        y: True labels.

    Returns:
        Dictionary of metric name → value.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
    }

    cm = confusion_matrix(y, y_pred)
    metrics["true_negatives"] = int(cm[0, 0])
    metrics["false_positives"] = int(cm[0, 1])
    metrics["false_negatives"] = int(cm[1, 0])
    metrics["true_positives"] = int(cm[1, 1])

    return metrics


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict,
) -> dict:
    """
    Train a RandomForest model with hyperparameter tuning, logging to MLflow.

    Args:
        X_train, y_train: Training features and labels.
        X_test, y_test: Test features and labels.
        config: Project configuration dict.

    Returns:
        Dictionary with best model, parameters, and metrics.
    """
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    experiment_name = config["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(mlflow_uri)

    from mlflow.tracking import MlflowClient
    _client = MlflowClient(tracking_uri=mlflow_uri)
    _existing = _client.get_experiment_by_name(experiment_name)

    def _is_http_artifact_loc(loc: str) -> bool:
        """True iff artifact_location uses HTTP-proxied storage."""
        return bool(loc) and loc.startswith("mlflow-artifacts:")

    def _create_fresh_experiment() -> str:
        new_id = _client.create_experiment(
            experiment_name,
            artifact_location="mlflow-artifacts:/",
        )
        logger.info(
            f"Created MLflow experiment '{experiment_name}' (id={new_id}) "
            f"with artifact_location='mlflow-artifacts:/' (HTTP proxy)"
        )
        return new_id

    if _existing is None:
        # Clean slate — first run.
        _create_fresh_experiment()

    elif _existing.lifecycle_stage == "deleted":
        # Soft-deleted with the same name. Restore if it's already correct;
        # otherwise archive it out of the way and create a fresh one.
        if _is_http_artifact_loc(_existing.artifact_location):
            _client.restore_experiment(_existing.experiment_id)
            logger.info(f"Restored soft-deleted experiment '{experiment_name}'")
        else:
            _archive_name = f"{experiment_name}_archived_{int(_time.time())}"
            _client.rename_experiment(_existing.experiment_id, _archive_name)
            _create_fresh_experiment()
            logger.warning(
                f"Soft-deleted experiment '{experiment_name}' had bad "
                f"artifact_location='{_existing.artifact_location}'. "
                f"Renamed to '{_archive_name}' and created a fresh one."
            )

    elif not _is_http_artifact_loc(_existing.artifact_location):
        _bad_loc = _existing.artifact_location
        _archive_name = f"{experiment_name}_archived_{int(_time.time())}"
        try:
            _client.rename_experiment(_existing.experiment_id, _archive_name)
            _create_fresh_experiment()
            logger.warning(
                f"AUTO-HEALED MLflow state: experiment '{experiment_name}' had "
                f"artifact_location='{_bad_loc}' (would cause PermissionError). "
                f"Old experiment archived as '{_archive_name}' and a fresh one "
                f"created with HTTP artifact storage. Old run history is still "
                f"accessible in the MLflow UI under the archived name."
            )
        except Exception as _heal_err:
            logger.error(
                f"AUTO-HEAL FAILED while fixing experiment '{experiment_name}' "
                f"(artifact_location='{_bad_loc}'): {_heal_err}. "
                f"Run scripts/reset_mlflow.ps1 manually to wipe the MLflow DB."
            )
            raise

    # If _existing has correct artifact_location, fall through unchanged.

    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow tracking URI: {mlflow_uri}")
    logger.info(f"Experiment: {experiment_name}")

    # Hyperparameter grid
    param_grid = {
        "n_estimators": config["model"]["hyperparameters"]["n_estimators"],
        "max_depth": config["model"]["hyperparameters"]["max_depth"],
        "min_samples_split": config["model"]["hyperparameters"]["min_samples_split"],
        "class_weight": config["model"]["hyperparameters"]["class_weight"],
    }

    best_model = None
    best_f1 = -1
    best_params = {}
    best_run_id = None

    # Resolve once so every run in the grid shares the same commit tag
    git_commit = _get_git_commit()
    logger.info(f"Git commit: {git_commit}")

    # Grid search with MLflow logging
    for n_est in param_grid["n_estimators"]:
        for max_d in param_grid["max_depth"]:
            for min_ss in param_grid["min_samples_split"]:
                for cw in param_grid["class_weight"]:
                    params = {
                        "n_estimators": n_est,
                        "max_depth": max_d,
                        "min_samples_split": min_ss,
                        "class_weight": cw,
                        "random_state": config["data"]["random_state"],
                    }

                    with mlflow.start_run(run_name=f"RF_n{n_est}_d{max_d}_s{min_ss}"):
                        # Log parameters
                        mlflow.log_params(params)
                        mlflow.log_param("algorithm", "RandomForest")
                        mlflow.log_param("n_features", X_train.shape[1])
                        mlflow.log_param("n_train_samples", X_train.shape[0])
                        # Reproducibility (guideline §I — every experiment reproducible
                        # via git commit hash + MLflow run ID)
                        mlflow.log_param("git_commit", git_commit)
                        mlflow.set_tag("git_commit", git_commit)

                        # Train
                        model = RandomForestClassifier(**params)
                        model.fit(X_train, y_train)

                        # Evaluate on train
                        train_metrics = evaluate_model(model, X_train, y_train)
                        for k, v in train_metrics.items():
                            mlflow.log_metric(f"train_{k}", v)

                        # Evaluate on test
                        test_metrics = evaluate_model(model, X_test, y_test)
                        for k, v in test_metrics.items():
                            mlflow.log_metric(f"test_{k}", v)

                        # Feature importance
                        feat_imp = dict(zip(
                            get_feature_columns(),
                            model.feature_importances_.tolist(),
                        ))
                        mlflow.log_dict(feat_imp, "feature_importance.json")

                        # Log model with input example to suppress MLflow signature warning
                        input_example = X_train.iloc[:1]
                        mlflow.sklearn.log_model(
                            model, "model",
                            input_example=input_example,
                        )

                        # Track best
                        current_f1 = test_metrics["f1_score"]
                        if current_f1 > best_f1:
                            best_f1 = current_f1
                            best_model = model
                            best_params = params
                            best_run_id = mlflow.active_run().info.run_id

                        logger.info(
                            f"Run: n_est={n_est}, max_d={max_d}, min_ss={min_ss} | "
                            f"Test F1={current_f1:.4f}, AUC={test_metrics['roc_auc']:.4f}"
                        )

    logger.info(f"Best model — F1={best_f1:.4f} | Params={best_params} | Run ID={best_run_id}")

    return {
        "model": best_model,
        "best_params": best_params,
        "best_f1": best_f1,
        "best_run_id": best_run_id,
    }



# Model registration
def register_best_model(run_id: str, model_name: str, config: dict) -> str:

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(mlflow_uri)

    model_uri = f"runs:/{run_id}/model"
    try:
        result = mlflow.register_model(model_uri, model_name)
        version = result.version
        logger.info(f"Registered model '{model_name}' version {version}")
        return version
    except Exception as e:
        logger.warning(f"Model registration skipped (MLflow server may not support it): {e}")
        return "local"


# Full training pipeline
def run_training(config: dict | None = None) -> dict:

    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = str(root / config["data"]["processed_dir"])
    models_dir = str(root / "models")
    ensure_dir(models_dir)

    # ── Step 0: Decide whether ingestion+preprocessing must run first ──
    train_processed = Path(processed_dir) / "train_processed.csv"
    latest_upload = get_latest_training_file(config)
    needs_reprocess = False

    if not train_processed.exists():
        logger.info("[PIPELINE] Processed data missing — running ingestion + preprocessing")
        needs_reprocess = True
    elif latest_upload:
        upload_mtime = Path(latest_upload).stat().st_mtime
        processed_mtime = train_processed.stat().st_mtime
        if upload_mtime > processed_mtime:
            logger.info(
                f"[UPLOAD] Uploaded file ({Path(latest_upload).name}) is newer than "
                f"processed data — re-running ingestion + preprocessing"
            )
            needs_reprocess = True
        else:
            logger.info(
                "[UPLOAD] Processed data is already up-to-date with latest upload"
            )

    if needs_reprocess:
        from src.data_ingestion import run_ingestion
        from src.data_preprocessing import run_preprocessing
        ingest_result = run_ingestion(config)
        logger.info(
            f"[PIPELINE] Ingestion done — source={ingest_result['data_source']} "
            f"rows={ingest_result['train_rows']}+{ingest_result['test_rows']}"
        )
        run_preprocessing(config)
        logger.info("[PIPELINE] Preprocessing done")

    # ── Step 1: Load processed data 
    train_df = pd.read_csv(os.path.join(processed_dir, "train_processed.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test_processed.csv"))

    feature_cols = get_feature_columns()
    target = config["features"]["target"]

    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    logger.info(f"Training data: {X_train.shape} | Test data: {X_test.shape}")
    logger.info(f"Target distribution (train): {y_train.value_counts().to_dict()}")

    # Train with MLflow — timed for Prometheus
    t0 = _time.time()
    result = train_model(X_train, y_train, X_test, y_test, config)
    training_duration = _time.time() - t0
    logger.info(f"Training completed in {training_duration:.1f}s")

    try:
        from api.metrics import TRAINING_DURATION_SECONDS, LAST_TRAINING_TIMESTAMP
        TRAINING_DURATION_SECONDS.observe(training_duration)
        LAST_TRAINING_TIMESTAMP.set(_time.time())
    except Exception:
        pass  # Airflow workers don't load the API metrics registry

    # Save best model locally
    model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(result["model"], model_path)
    logger.info(f"Best model saved to {model_path}")

    # Save test metrics
    test_metrics = evaluate_model(result["model"], X_test, y_test)
    save_json(test_metrics, os.path.join(models_dir, "test_metrics.json"))

    # Save classification report
    y_pred = result["model"].predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    save_json(report, os.path.join(models_dir, "classification_report.json"))

    # Register model
    model_name = config["mlflow"]["model_name"]
    version = register_best_model(result["best_run_id"], model_name, config)

    # Tag the best run with audit metadata (no run reopen — avoids URI conflicts)
    data_src = "uploaded" if latest_upload and needs_reprocess else "existing_processed"
    try:
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
        client.set_tag(result["best_run_id"], "data_source", data_src)
        client.set_tag(result["best_run_id"], "training_duration_s", f"{training_duration:.1f}")
        client.set_tag(result["best_run_id"], "model_version", str(version))
        logger.info(f"Tagged run {result['best_run_id']} with data_source={data_src}")
    except Exception as _te:
        logger.warning(f"Post-run tagging skipped: {_te}")

    return {
        "status": "success",
        "model_path": model_path,
        "best_f1": result["best_f1"],
        "best_params": result["best_params"],
        "best_run_id": result["best_run_id"],
        "model_version": version,
        "test_metrics": test_metrics,
        "data_source": data_src,
        "training_duration_seconds": round(training_duration, 1),
        "git_commit": _get_git_commit(),
    }


if __name__ == "__main__":
    result = run_training()
    print(f"Training complete: {result['status']}")
    print(f"Best F1: {result['best_f1']:.4f}")
    print(f"Test metrics: {result['test_metrics']}")