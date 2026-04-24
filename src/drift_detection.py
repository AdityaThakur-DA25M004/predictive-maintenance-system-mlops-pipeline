"""
Drift Detection Module for Predictive Maintenance System.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import setup_logger, load_json

logger = setup_logger(__name__)


# 🔥 NEW: Safe numpy → python converter
def _to_python(obj):
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_python(v) for v in obj]
    elif hasattr(obj, "item"):  # numpy types
        return obj.item()
    return obj


def ks_test(reference: list[float], current: list[float],
            threshold: float = 0.05) -> dict:

    stat, p_value = stats.ks_2samp(reference, current)

    return {
        "test": "ks_test",
        "statistic": float(stat),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < threshold),  # ✅ FIX
    }


def compute_psi(reference: np.ndarray, current: np.ndarray,
                n_bins: int = 10) -> float:

    eps = 1e-6
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        n_bins + 1,
    )

    ref_hist = np.histogram(reference, bins=breakpoints)[0] / len(reference) + eps
    cur_hist = np.histogram(current, bins=breakpoints)[0] / len(current) + eps

    psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))

    return float(psi)


def detect_drift(
    current_data: pd.DataFrame,
    baselines: dict,
    feature_cols: list[str],
    ks_threshold: float = 0.05,
    psi_threshold: float = 0.25,
) -> dict:

    drift_report = {
        "features": {},
        "overall_drift": False,
        "drifted_features": [],
        "total_features_checked": len(feature_cols),
    }

    for col in feature_cols:
        if col not in baselines or col not in current_data.columns:
            continue

        baseline = baselines[col]
        current_values = current_data[col].dropna().values

        if len(current_values) < 3:
            drift_report["features"][col] = {
                "status": "insufficient_data",
                "n_samples": int(len(current_values)),
                "minimum_required": 3,
            }
            continue

        np.random.seed(42)
        ref_values = np.random.normal(
            baseline["mean"], baseline["std"], baseline["count"]
        )

        ks_result = ks_test(ref_values.tolist(), current_values.tolist(), ks_threshold)

        psi_value = compute_psi(ref_values, current_values)

        mean_shift = abs(current_values.mean() - baseline["mean"]) / (baseline["std"] + 1e-6)
        std_ratio = current_values.std() / (baseline["std"] + 1e-6)

        feature_drift = ks_result["drift_detected"] or psi_value > psi_threshold

        drift_report["features"][col] = {
            "ks_statistic": float(ks_result["statistic"]),
            "ks_p_value": float(ks_result["p_value"]),
            "ks_drift": bool(ks_result["drift_detected"]),
            "psi": float(psi_value),
            "psi_drift": bool(psi_value > psi_threshold),
            "current_mean": float(current_values.mean()),
            "baseline_mean": float(baseline["mean"]),
            "mean_shift_std": float(mean_shift),
            "std_ratio": float(std_ratio),
            "drift_detected": bool(feature_drift),  # ✅ FIX
        }

        if feature_drift:
            drift_report["drifted_features"].append(col)

    drift_report["overall_drift"] = bool(len(drift_report["drifted_features"]) > 0)
    drift_report["n_drifted"] = int(len(drift_report["drifted_features"]))

    logger.info(
        f"Drift detection complete: {drift_report['n_drifted']}/"
        f"{drift_report['total_features_checked']} features drifted"
    )

    # 🔥 FINAL FIX (ensures NO numpy types anywhere)
    return _to_python(drift_report)


def check_drift_from_file(data_path: str, baselines_path: str,
                          feature_cols: list[str]) -> dict:

    df = pd.read_csv(data_path)
    baselines = load_json(baselines_path)
    return detect_drift(df, baselines, feature_cols)