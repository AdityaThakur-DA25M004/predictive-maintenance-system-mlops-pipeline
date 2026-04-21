"""
Unit tests for the drift detection module.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.drift_detection import ks_test, compute_psi


class TestKSTest:
    def test_identical_distributions_no_drift(self):
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000).tolist()
        cur = np.random.normal(0, 1, 1000).tolist()
        result = ks_test(ref, cur)
        assert result["drift_detected"] == False
        assert result["p_value"] > 0.05

    def test_shifted_distribution_detects_drift(self):
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000).tolist()
        cur = np.random.normal(3, 1, 1000).tolist()  # shifted by 3σ
        result = ks_test(ref, cur)
        assert result["drift_detected"] == True
        assert result["p_value"] < 0.05


class TestPSI:
    def test_identical_data_low_psi(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.1

    def test_shifted_data_high_psi(self):
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(5, 2, 1000)
        psi = compute_psi(ref, cur)
        assert psi > 0.25
