"""
Unit tests for the data preprocessing module.
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_preprocessing import engineer_features, get_feature_columns, fit_scaler, apply_scaler, compute_drift_baselines


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "Air temperature [K]": [300.0, 301.0, 302.0],
        "Process temperature [K]": [310.0, 312.0, 314.0],
        "Rotational speed [rpm]": [1500, 1600, 1400],
        "Torque [Nm]": [40.0, 45.0, 35.0],
        "Tool wear [min]": [10, 50, 100],
        "Type": ["L", "M", "H"],
    })


class TestFeatureEngineering:
    def test_adds_derived_features(self, raw_df):
        result = engineer_features(raw_df)
        assert "temp_diff" in result.columns
        assert "power" in result.columns
        assert "wear_degree" in result.columns
        assert "speed_torque_ratio" in result.columns
        assert "type_encoded" in result.columns

    def test_temp_diff_correct(self, raw_df):
        result = engineer_features(raw_df)
        expected = raw_df["Process temperature [K]"] - raw_df["Air temperature [K]"]
        pd.testing.assert_series_equal(result["temp_diff"], expected, check_names=False)

    def test_type_encoding(self, raw_df):
        result = engineer_features(raw_df)
        assert list(result["type_encoded"]) == [1, 2, 0]

    def test_no_mutation(self, raw_df):
        original = raw_df.copy()
        engineer_features(raw_df)
        pd.testing.assert_frame_equal(raw_df, original)


class TestScaler:
    def test_fit_and_apply(self, raw_df):
        df = engineer_features(raw_df)
        feature_cols = get_feature_columns()
        scaler = fit_scaler(df, feature_cols)
        scaled = apply_scaler(df, feature_cols, scaler)
        for col in feature_cols:
            assert abs(scaled[col].mean()) < 1e-10
            assert abs(scaled[col].std(ddof=0) - 1.0) < 0.1 or len(df) < 10


class TestDriftBaselines:
    def test_baseline_structure(self, raw_df):
        df = engineer_features(raw_df)
        feature_cols = get_feature_columns()
        baselines = compute_drift_baselines(df, feature_cols)
        for col in feature_cols:
            assert "mean" in baselines[col]
            assert "std" in baselines[col]
            assert "min" in baselines[col]
            assert "max" in baselines[col]
            assert "count" in baselines[col]
