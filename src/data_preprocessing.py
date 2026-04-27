"""
Data Preprocessing & Feature Engineering Module.
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

REF_SAMPLE_SIZE = int(os.environ.get("DRIFT_REF_SAMPLE_SIZE", "2000"))


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    df["wear_degree"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    df["speed_torque_ratio"] = df["Rotational speed [rpm]"] / (df["Torque [Nm]"] + 1e-6)

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


# Scaling
def fit_scaler(df: pd.DataFrame, feature_cols: list[str],
               save_path: str | None = None) -> StandardScaler:
    """Fit a StandardScaler on the training data."""
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
    """Apply a fitted scaler to the DataFrame."""
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


# Drift Baselines
def compute_drift_baselines(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Compute per-feature statistical baselines (mean/std/quantiles/etc)."""
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


def compute_reference_samples(df: pd.DataFrame, feature_cols: list[str],
                              n_samples: int = REF_SAMPLE_SIZE,
                              random_state: int = 42) -> dict:
    """
    Return a dict {feature_name: [list of float values]} sampled from the
    training data. These are used by drift_detection.detect_drift as the
    real reference distribution for the KS test (instead of a synthetic
    Gaussian).
    """
    n = min(n_samples, len(df))
    sample_df = df[feature_cols].dropna().sample(
        n=n, random_state=random_state, replace=False
    )
    ref = {col: sample_df[col].astype(float).tolist() for col in feature_cols}
    logger.info(
        f"Reference samples prepared: {n} rows × {len(feature_cols)} features"
    )
    return ref


# Full preprocessing pipeline
def run_preprocessing(config: dict | None = None) -> dict:
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = str(root / config["data"]["processed_dir"])
    baselines_path = str(root / config["data"]["baselines_path"])
    models_dir = str(root / "models")
    ensure_dir(models_dir)
    ensure_dir(os.path.dirname(baselines_path))

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

    # Reference samples for KS/PSI — sibling of baselines.json
    ref_samples_path = str(Path(baselines_path).parent / "ref_samples.json")
    ref_samples = compute_reference_samples(train_df, feature_cols)
    save_json(ref_samples, ref_samples_path)
    logger.info(f"Reference samples saved to {ref_samples_path}")

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
        "ref_samples_path": ref_samples_path,
        "feature_columns": feature_cols,
        "num_features": len(feature_cols),
    }


if __name__ == "__main__":
    result = run_preprocessing()
    print(f"Preprocessing complete: {result['status']}")
    print(f"Features: {result['num_features']} | Scaler: {result['scaler_path']}")
    print(f"Ref samples: {result['ref_samples_path']}")