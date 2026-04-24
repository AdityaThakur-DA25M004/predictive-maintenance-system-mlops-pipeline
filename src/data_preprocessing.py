"""
Data Preprocessing & Feature Engineering Module.

Handles cleaning, transformation, feature creation, scaling,
and computation of drift baselines.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import setup_logger, load_config, get_project_root, ensure_dir, save_json

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw sensor data.

    New features:
        - temp_diff: Difference between process and air temperature.
        - power: Product of torque and rotational speed (W).
        - wear_degree: Interaction of tool wear and torque.
        - type_encoded: Label-encoded product type (H=0, L=1, M=2).
        - speed_torque_ratio: Rotational speed / Torque.

    Args:
        df: DataFrame with raw features.

    Returns:
        DataFrame with added engineered features.
    """
    df = df.copy()

    # Temperature difference
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    # Power approximation (torque × angular velocity)
    df["power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)

    # Wear-torque interaction
    df["wear_degree"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    # Speed-to-torque ratio
    df["speed_torque_ratio"] = df["Rotational speed [rpm]"] / (df["Torque [Nm]"] + 1e-6)

    # Encode product type
    type_map = {"H": 0, "L": 1, "M": 2}
    df["type_encoded"] = df["Type"].map(type_map)

    logger.info("Feature engineering complete — added 5 derived features")
    return df


def get_feature_columns() -> list[str]:
    """Return the list of feature column names used for model training."""
    return [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "type_encoded",
        "temp_diff",
        "power",
        "wear_degree",
        "speed_torque_ratio",
    ]


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------
def fit_scaler(df: pd.DataFrame, feature_cols: list[str],
               save_path: str | None = None) -> StandardScaler:
    """
    Fit a StandardScaler on the training data.

    Args:
        df: Training DataFrame.
        feature_cols: Columns to scale.
        save_path: Optional path to persist the scaler.

    Returns:
        Fitted StandardScaler.
    """
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    logger.info(f"Scaler fitted on {len(feature_cols)} features")

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        joblib.dump(scaler, save_path)
        logger.info(f"Scaler saved to {save_path}")

    return scaler


def apply_scaler(df: pd.DataFrame, feature_cols: list[str],
                 scaler: StandardScaler) -> pd.DataFrame:
    """
    Apply a fitted scaler to the DataFrame.

    Args:
        df: DataFrame to transform.
        feature_cols: Columns to scale.
        scaler: Fitted StandardScaler.

    Returns:
        DataFrame with scaled feature columns.
    """
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


# ---------------------------------------------------------------------------
# Drift Baselines
# ---------------------------------------------------------------------------
def compute_drift_baselines(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Compute statistical baselines for drift detection.

    For each feature, computes mean, std, min, max, median,
    25th/75th percentiles, skewness, and kurtosis.

    Args:
        df: Training DataFrame (before scaling).
        feature_cols: Feature columns to baseline.

    Returns:
        Dictionary mapping feature names to their statistics.
    """
    baselines = {}
    for col in feature_cols:
        series = df[col].dropna()
        baselines[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
            "count": int(len(series)),
        }
    logger.info(f"Computed drift baselines for {len(feature_cols)} features")
    return baselines


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------
def run_preprocessing(config: dict | None = None) -> dict:
    """
    Execute the full preprocessing pipeline.

    Steps:
        1. Load train/test CSVs from the processed directory.
        2. Apply feature engineering.
        3. Compute drift baselines on training data (pre-scaling).
        4. Fit scaler on training data, transform both sets.
        5. Save processed datasets, scaler, and baselines.

    Args:
        config: Optional config dict.

    Returns:
        Dictionary with output paths and summary.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = str(root / config["data"]["processed_dir"])
    baselines_path = str(root / config["data"]["baselines_path"])
    models_dir = str(root / "models")
    ensure_dir(models_dir)

    # Load split data
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    logger.info(f"Loaded train ({len(train_df)}) and test ({len(test_df)}) datasets")

    # Feature engineering
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    feature_cols = get_feature_columns()

    # Drift baselines (on raw-scale training features)
    baselines = compute_drift_baselines(train_df, feature_cols)
    save_json(baselines, baselines_path)
    logger.info(f"Drift baselines saved to {baselines_path}")

    # Fit & apply scaler
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    scaler = fit_scaler(train_df, feature_cols, save_path=scaler_path)
    train_scaled = apply_scaler(train_df, feature_cols, scaler)
    test_scaled = apply_scaler(test_df, feature_cols, scaler)

    # Save processed data
    train_out = os.path.join(processed_dir, "train_processed.csv")
    test_out = os.path.join(processed_dir, "test_processed.csv")
    train_scaled.to_csv(train_out, index=False)
    test_scaled.to_csv(test_out, index=False)
    logger.info(f"Saved processed train to {train_out}")
    logger.info(f"Saved processed test to {test_out}")

    return {
        "status": "success",
        "train_processed_path": train_out,
        "test_processed_path": test_out,
        "scaler_path": scaler_path,
        "baselines_path": baselines_path,
        "feature_columns": feature_cols,
        "num_features": len(feature_cols),
    }


if __name__ == "__main__":
    result = run_preprocessing()
    print(f"Preprocessing complete: {result['status']}")
    print(f"Features: {result['num_features']} | Scaler: {result['scaler_path']}")