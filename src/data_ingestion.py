"""
Data Ingestion Module for Predictive Maintenance System.

Handles loading raw data, schema validation, quality checks,
and train/test splitting with stratification on the target.

IMPORTANT — Note on leakage:
The columns TWF, HDF, PWF, OSF, RNF are per-failure-mode flags that are
deterministically set whenever `Machine failure = 1`. Including them as
features would cause label leakage. This module drops them (and the
identifier columns UDI / Product ID) from the saved train/test files so
no downstream consumer accidentally uses them.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import setup_logger, load_config, get_project_root, ensure_dir

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
EXPECTED_SCHEMA = {
    "UDI": "int", "Product ID": "str", "Type": "str",
    "Air temperature [K]": "float", "Process temperature [K]": "float",
    "Rotational speed [rpm]": "int", "Torque [Nm]": "float",
    "Tool wear [min]": "int", "Machine failure": "int",
    "TWF": "int", "HDF": "int", "PWF": "int", "OSF": "int", "RNF": "int",
}

# Columns removed to prevent label leakage and identifier noise
LEAKY_FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
IDENTIFIER_COLS = ["UDI", "Product ID"]


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}")
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(f"Raw data file is empty: {filepath}")
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """Validate that the DataFrame contains all expected columns."""
    missing = set(EXPECTED_SCHEMA.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Schema validation passed")
    return True


def validate_data_quality(df: pd.DataFrame) -> dict:
    """Run data quality checks and return a report."""
    report = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": {int(k): int(v) for k, v in df["Machine failure"].value_counts().to_dict().items()},
    }

    numeric_checks = {}
    if (df["Air temperature [K]"] < 250).any() or (df["Air temperature [K]"] > 350).any():
        numeric_checks["air_temp_outliers"] = True
    if (df["Rotational speed [rpm]"] < 0).any():
        numeric_checks["negative_rpm"] = True
    if (df["Torque [Nm]"] < 0).any():
        numeric_checks["negative_torque"] = True
    if (df["Tool wear [min]"] < 0).any():
        numeric_checks["negative_tool_wear"] = True

    report["numeric_anomalies"] = numeric_checks
    report["quality_status"] = "PASS" if not numeric_checks and report["total_missing"] == 0 else "WARNING"

    positives = report["target_distribution"].get(1, 0)
    report["failure_rate"] = float(positives / max(report["total_rows"], 1))
    report["class_imbalance_flag"] = report["failure_rate"] < 0.1

    logger.info(
        f"Data quality: {report['quality_status']} | "
        f"Missing={report['total_missing']} | Duplicates={report['duplicate_rows']} | "
        f"Failure rate={report['failure_rate']:.4f}"
    )
    return report


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove identifier and failure-mode indicator columns.

    Failure-mode columns (TWF/HDF/PWF/OSF/RNF) are deterministically set
    whenever Machine failure == 1, so including them as features would
    leak the label. Identifiers (UDI, Product ID) carry no signal.
    """
    to_drop = [c for c in LEAKY_FAILURE_MODE_COLS + IDENTIFIER_COLS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        logger.info(f"Dropped leaky/identifier columns: {to_drop}")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2,
               random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified random split on the target."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df["Machine failure"],
    )
    logger.info(f"Stratified split: train={len(train_df)}, test={len(test_df)}")
    logger.info(f"Train failure rate: {train_df['Machine failure'].mean():.4f}")
    logger.info(f"Test  failure rate: {test_df['Machine failure'].mean():.4f}")
    return train_df, test_df


def run_ingestion(config: dict | None = None) -> dict:
    """Execute the full ingestion pipeline."""
    if config is None:
        config = load_config()

    root = get_project_root()
    raw_path = str(root / config["data"]["raw_path"])
    processed_dir = str(root / config["data"]["processed_dir"])
    ensure_dir(processed_dir)

    df = load_raw_data(raw_path)
    validate_schema(df)
    quality_report = validate_data_quality(df)

    # Drop leaky + identifier columns BEFORE splitting
    df = drop_leaky_columns(df)

    test_size = config["data"].get("test_size", 0.2)
    random_state = config["data"].get("random_state", 42)
    train_df, test_df = split_data(df, test_size, random_state)

    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved train to {train_path}")
    logger.info(f"Saved test  to {test_path}")

    return {
        "status": "success",
        "train_path": train_path,
        "test_path": test_path,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "quality_report": quality_report,
    }


if __name__ == "__main__":
    result = run_ingestion()
    print(f"Ingestion complete: {result['status']}")
    print(f"Train: {result['train_rows']} rows | Test: {result['test_rows']} rows")
