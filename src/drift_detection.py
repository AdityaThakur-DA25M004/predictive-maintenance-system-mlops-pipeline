"""
Drift Detection Module for Predictive Maintenance System.

Implements statistical tests to detect data distribution drift
by comparing incoming data against training baselines.
Uses Kolmogorov-Smirnov test and Population Stability Index.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import setup_logger, load_json

logger = setup_logger(__name__)


def ks_test(reference: list[float], current: list[float],
            threshold: float = 0.05) -> dict:
    """
    Perform Kolmogorov-Smirnov two-sample test.

    Args:
        reference: Baseline distribution values (from training).
        current: New distribution values (incoming data).
        threshold: p-value threshold for drift detection.

    Returns:
        Dict with statistic, p_value, and drift flag.
    """
    stat, p_value = stats.ks_2samp(reference, current)
    return {
        "test": "ks_test",
        "statistic": float(stat),
        "p_value": float(p_value),
        "drift_detected": p_value < threshold,
    }


def compute_psi(reference: np.ndarray, current: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Compute Population Stability Index.

    PSI < 0.1 → no significant shift
    PSI 0.1–0.25 → moderate shift
    PSI > 0.25 → significant shift

    Args:
        reference: Reference distribution.
        current: Current distribution.
        n_bins: Number of bins for discretization.

    Returns:
        PSI value.
    """
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
    """
    Run drift detection on incoming data vs. training baselines.

    Uses both KS-test (on synthetic reference from baseline stats)
    and PSI to flag drifted features.

    Args:
        current_data: New incoming DataFrame.
        baselines: Baseline statistics dict (from training).
        feature_cols: Feature columns to check.
        ks_threshold: p-value threshold for KS test.
        psi_threshold: PSI value threshold.

    Returns:
        Comprehensive drift report.
    """
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

        if len(current_values) < 10:
            drift_report["features"][col] = {
                "status": "insufficient_data",
                "n_samples": len(current_values),
            }
            continue

        # Generate synthetic reference from baseline stats
        np.random.seed(42)
        ref_values = np.random.normal(
            baseline["mean"], baseline["std"], baseline["count"]
        )

        # KS test
        ks_result = ks_test(ref_values.tolist(), current_values.tolist(), ks_threshold)

        # PSI
        psi_value = compute_psi(ref_values, current_values)

        # Mean/std comparison
        mean_shift = abs(current_values.mean() - baseline["mean"]) / (baseline["std"] + 1e-6)
        std_ratio = current_values.std() / (baseline["std"] + 1e-6)

        feature_drift = ks_result["drift_detected"] or psi_value > psi_threshold

        drift_report["features"][col] = {
            "ks_statistic": ks_result["statistic"],
            "ks_p_value": ks_result["p_value"],
            "ks_drift": ks_result["drift_detected"],
            "psi": psi_value,
            "psi_drift": psi_value > psi_threshold,
            "current_mean": float(current_values.mean()),
            "baseline_mean": baseline["mean"],
            "mean_shift_std": float(mean_shift),
            "std_ratio": float(std_ratio),
            "drift_detected": feature_drift,
        }

        if feature_drift:
            drift_report["drifted_features"].append(col)

    drift_report["overall_drift"] = len(drift_report["drifted_features"]) > 0
    drift_report["n_drifted"] = len(drift_report["drifted_features"])

    logger.info(
        f"Drift detection complete: {drift_report['n_drifted']}/"
        f"{drift_report['total_features_checked']} features drifted"
    )

    return drift_report


def check_drift_from_file(data_path: str, baselines_path: str,
                          feature_cols: list[str]) -> dict:
    """
    Convenience function to run drift detection from file paths.

    Args:
        data_path: Path to incoming data CSV.
        baselines_path: Path to baseline JSON.
        feature_cols: Feature columns to check.

    Returns:
        Drift report dictionary.
    """
    df = pd.read_csv(data_path)
    baselines = load_json(baselines_path)
    return detect_drift(df, baselines, feature_cols)
