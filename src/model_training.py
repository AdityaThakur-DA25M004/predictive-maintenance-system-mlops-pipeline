"""
Model Training Module with MLflow Experiment Tracking.

Trains a Random Forest classifier for machine failure prediction,
logs all parameters/metrics/artifacts to MLflow, and registers
the best model in the MLflow Model Registry.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
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

                        # Log model artifact
                        mlflow.sklearn.log_model(model, "model")

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


# ---------------------------------------------------------------------------
# Model registration
# ---------------------------------------------------------------------------
def register_best_model(run_id: str, model_name: str, config: dict) -> str:
    """
    Register the best model in MLflow Model Registry.

    Args:
        run_id: MLflow run ID of the best model.
        model_name: Name for the registered model.
        config: Project configuration dict.

    Returns:
        Model version string.
    """
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


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------
def run_training(config: dict | None = None) -> dict:
    """
    Execute the full training pipeline.

    Steps:
        1. Load processed train/test data.
        2. Train models with hyperparameter search + MLflow logging.
        3. Save the best model locally.
        4. Register in MLflow Model Registry.

    Args:
        config: Optional config dict.

    Returns:
        Dictionary with training results.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = str(root / config["data"]["processed_dir"])
    models_dir = str(root / "models")
    ensure_dir(models_dir)

    # Load processed data
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

    # Train with MLflow
    result = train_model(X_train, y_train, X_test, y_test, config)

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

    return {
        "status": "success",
        "model_path": model_path,
        "best_f1": result["best_f1"],
        "best_params": result["best_params"],
        "best_run_id": result["best_run_id"],
        "model_version": version,
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    result = run_training()
    print(f"Training complete: {result['status']}")
    print(f"Best F1: {result['best_f1']:.4f}")
    print(f"Test metrics: {result['test_metrics']}")
