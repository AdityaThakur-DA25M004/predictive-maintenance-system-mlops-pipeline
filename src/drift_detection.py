"""
Drift Detection Module for Predictive Maintenance System.

KEY FIX (vs previous version)
-----------------------------
Previously the KS test compared current data against synthetic samples drawn
from N(baseline_mean, baseline_std). For non-Gaussian features
(`type_encoded`, `tool_wear`, engineered ratios, …) that synthetic reference
diverges from the real distribution and the KS test reports drift on every
run — even when comparing the data against itself.

The fix: persist a real sample of training-time feature values in
`ref_samples.json` (written by run_preprocessing) and use those samples as
the KS / PSI reference. Falls back to the old synthetic behaviour if no
ref_samples file is found, so the change is fully backward-compatible.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import setup_logger, load_json

logger = setup_logger(__name__)


# Safe numpy → python converter
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
        "drift_detected": bool(p_value < threshold),
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


def _resolve_ref_samples_path(baselines_path: str | None) -> str | None:
    """
    Look for ref_samples.json next to drift_baselines.json.
    Honours the env var DRIFT_REF_SAMPLES_PATH if set (Docker friendly).
    """
    env_path = os.environ.get("DRIFT_REF_SAMPLES_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    if baselines_path:
        sibling = Path(baselines_path).parent / "ref_samples.json"
        if sibling.exists():
            return str(sibling)
    return None


def _build_reference(col: str, baseline: dict,
                     ref_samples: dict | None) -> np.ndarray:
    """
    Return the reference array for KS/PSI for a single feature.

    Preference order:
      1. Real samples from `ref_samples[col]`  ← new, robust
      2. Synthetic N(mean, std)                ← legacy fallback
    """
    if ref_samples and col in ref_samples and len(ref_samples[col]) >= 30:
        return np.asarray(ref_samples[col], dtype=float)

    # Legacy fallback — synthetic Gaussian
    np.random.seed(42)
    return np.random.normal(
        baseline["mean"], baseline["std"], baseline["count"]
    )


def detect_drift(
    current_data: pd.DataFrame,
    baselines: dict,
    feature_cols: list[str],
    ks_threshold: float = 0.05,
    psi_threshold: float = 0.25,
    ref_samples: dict | None = None,
    baselines_path: str | None = None,
) -> dict:
    """
    Compare `current_data` against training baselines/reference samples.

    Args:
        current_data:     DataFrame to test (post feature-engineering).
        baselines:        dict of per-feature stats (from compute_drift_baselines).
        feature_cols:     features to check.
        ks_threshold:     p-value threshold (drift if p < threshold).
        psi_threshold:    PSI threshold (drift if PSI > threshold).
        ref_samples:      Optional pre-loaded reference samples.
        baselines_path:   Optional path; used to auto-discover ref_samples.json.
    """
    # Auto-load reference samples if caller didn't pass them
    if ref_samples is None:
        rs_path = _resolve_ref_samples_path(baselines_path)
        if rs_path:
            try:
                ref_samples = load_json(rs_path)
                logger.info(f"Loaded reference samples from {rs_path}")
            except Exception as e:
                logger.warning(f"Could not load ref_samples ({rs_path}): {e}")
                ref_samples = None
        else:
            logger.info("No ref_samples.json found — using synthetic Gaussian fallback")

    drift_report = {
        "features": {},
        "overall_drift": False,
        "drifted_features": [],
        "total_features_checked": len(feature_cols),
        "reference_type": "real_samples" if ref_samples else "synthetic_gaussian",
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

        ref_values = _build_reference(col, baseline, ref_samples)

        ks_result = ks_test(
            ref_values.tolist(), current_values.tolist(), ks_threshold
        )
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
            "drift_detected": bool(feature_drift),
        }

        if feature_drift:
            drift_report["drifted_features"].append(col)

    drift_report["overall_drift"] = bool(len(drift_report["drifted_features"]) > 0)
    drift_report["n_drifted"] = int(len(drift_report["drifted_features"]))

    logger.info(
        f"Drift detection complete ({drift_report['reference_type']}): "
        f"{drift_report['n_drifted']}/{drift_report['total_features_checked']} "
        f"features drifted"
    )

    return _to_python(drift_report)


def check_drift_from_file(data_path: str, baselines_path: str,
                          feature_cols: list[str]) -> dict:
    df = pd.read_csv(data_path)
    baselines = load_json(baselines_path)
    return detect_drift(
        df, baselines, feature_cols, baselines_path=baselines_path
    )

if __name__ == "__main__":
    """
    Standalone entry point used by:
      - `python -m src.drift_detection`
      - DVC stage `drift_check` in dvc.yaml
      - MLflow project entry point `drift_check` in MLproject
 
    Reads the processed test set, engineers features, compares against
    the existing drift baselines on disk, and writes:
      - data/baselines/drift_report.json   (full detail; tracked as DVC `outs`)
      - data/baselines/drift_summary.json  (flat summary; tracked as DVC `metrics`)
 
    First-run behaviour: if no baselines exist yet (preprocessing has
    never run), this writes empty-but-valid versions of both files so
    DVC doesn't fail. Subsequent runs will produce a real comparison.
    """
    import json
    from src.utils import load_config, get_project_root, ensure_dir
    from src.data_preprocessing import engineer_features, get_feature_columns
 
    cfg = load_config()
    root = get_project_root()
    test_path = os.path.join(str(root), cfg["data"]["processed_dir"], "test.csv")
    baselines_path = os.path.join(str(root), cfg["data"]["baselines_path"])
    baselines_dir = os.path.dirname(baselines_path)
    report_path = os.path.join(str(root), baselines_dir, "drift_report.json")
    summary_path = os.path.join(str(root), baselines_dir, "drift_summary.json")
    ensure_dir(os.path.dirname(report_path))
 
 
    def _write_summary(report: dict) -> dict:
        """
        Extract a FLAT top-level summary from a detailed drift report.
 
        Why flat: `dvc metrics show` recursively expands nested dicts
        into one row per leaf, which makes a per-feature drift report
        unreadable. A flat summary surfaces only what you actually want
        to track over time: did drift happen, how many features, what
        proportion, what reference was used.
        """
        n_total = int(report.get("total_features_checked", 0)) or 0
        n_drifted = int(report.get("n_drifted", 0)) or 0
        summary = {
            "overall_drift": bool(report.get("overall_drift", False)),
            "n_drifted": n_drifted,
            "n_total_features": n_total,
            "drift_pct": round(n_drifted / n_total, 4) if n_total else 0.0,
            "reference_type": report.get("reference_type", "unknown"),
        }
        return summary
 
 
    # First-run safeguard: no baselines on disk yet
    if not os.path.exists(baselines_path):
        empty_report = {
            "overall_drift": False,
            "n_drifted": 0,
            "total_features_checked": 0,
            "drifted_features": [],
            "feature_details": {},
            "reference_type": "first_run_no_baseline",
        }
        with open(report_path, "w") as f:
            json.dump(empty_report, f, indent=2)
        with open(summary_path, "w") as f:
            json.dump(_write_summary(empty_report), f, indent=2)
        logger.info(
            f"No baselines at {baselines_path}; wrote empty report+summary."
        )
        sys.exit(0)
 
    # Test data must exist (produced by data_ingestion stage)
    if not os.path.exists(test_path):
        logger.error(f"test.csv not found at {test_path} — run data_ingestion first")
        sys.exit(1)
 
    df = pd.read_csv(test_path)
    df = engineer_features(df)
    feature_cols = get_feature_columns()
    baselines = load_json(baselines_path)
 
    report = detect_drift(
        df, baselines, feature_cols, baselines_path=baselines_path
    )
 
    # Full detailed report — DVC `outs` (cached, versioned)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
 
    # Flat summary — DVC `metrics` (Git-tracked, surfaces in `dvc metrics show`)
    with open(summary_path, "w") as f:
        json.dump(_write_summary(report), f, indent=2)
 
    logger.info(
        f"Drift report written to {report_path} | "
        f"summary written to {summary_path} | "
        f"overall_drift={report['overall_drift']} | "
        f"{report['n_drifted']}/{report['total_features_checked']} features drifted"
    )
    print(
        f"Drift check complete: {report['n_drifted']}/{report['total_features_checked']} "
        f"features drifted (overall_drift={report['overall_drift']})"
    )