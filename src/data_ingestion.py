"""
Data Ingestion Module for Predictive Maintenance System.

"""

import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import setup_logger, load_config, get_project_root, ensure_dir

logger = setup_logger(__name__)

# Resolved once at import time; Airflow/API containers both set this env var.
UPLOADS_DIR = os.environ.get("UPLOADS_DIR", "data/feedback/uploads")

# Schema
EXPECTED_SCHEMA = {
    "UDI": "int", "Product ID": "str", "Type": "str",
    "Air temperature [K]": "float", "Process temperature [K]": "float",
    "Rotational speed [rpm]": "int", "Torque [Nm]": "float",
    "Tool wear [min]": "int", "Machine failure": "int",
    "TWF": "int", "HDF": "int", "PWF": "int", "OSF": "int", "RNF": "int",
}

_OPTIONAL_COLS = {"UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"}

LEAKY_FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
IDENTIFIER_COLS = ["UDI", "Product ID"]


import re

_UPLOAD_FILENAME_RE = re.compile(r"^upload_(\d+)_")


def _upload_sort_key(path: Path) -> tuple[int, float]:

    m = _UPLOAD_FILENAME_RE.match(path.name)
    if m:
        return (int(m.group(1)), path.stat().st_mtime)
    # Files without the upload_<ts>_ prefix sort BEFORE proper uploads (ts=0),
    # falling back to mtime as a last-resort tiebreaker.
    return (0, path.stat().st_mtime)


def get_latest_upload(uploads_dir: str | None = None) -> str | None:

    dir_path = Path(uploads_dir or UPLOADS_DIR)
    if not dir_path.exists():
        return None
    files = list(dir_path.glob("*.csv"))
    if not files:
        return None
    latest = max(files, key=_upload_sort_key)
    # Log enough context to diagnose future selection issues
    sort_key = _upload_sort_key(latest)
    logger.info(
        f"[UPLOAD] Latest uploaded dataset: {latest} "
        f"(filename_ts={sort_key[0]}, mtime={sort_key[1]:.0f}, "
        f"total_uploads={len(files)})"
    )
    return str(latest)


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

    all_expected = set(EXPECTED_SCHEMA.keys())
    missing = all_expected - set(df.columns)
    critical_missing = missing - _OPTIONAL_COLS
    if critical_missing:
        raise ValueError(
            f"Missing required columns: {sorted(critical_missing)}. "
            f"Optional columns that may be absent: {sorted(_OPTIONAL_COLS)}"
        )
    if missing:
        logger.info(f"Optional columns absent (will be skipped): {sorted(missing)}")
    logger.info("Schema validation passed")
    return True


def validate_data_quality(df: pd.DataFrame) -> dict:
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

    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = str(root / config["data"]["processed_dir"])
    ensure_dir(processed_dir)

    #  Data source selection 
    uploaded = get_latest_upload()
    if uploaded:
        raw_path = uploaded
        data_source = "uploaded"
        logger.info(f"[UPLOAD] Using uploaded dataset for ingestion: {raw_path}")
    else:
        raw_path = str(root / config["data"]["raw_path"])
        data_source = "default"
        logger.info(f"[DEFAULT] No uploads found — using default dataset: {raw_path}")

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
        "data_source": data_source,
        "raw_path": raw_path,
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