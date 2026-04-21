"""
Unit tests for the data ingestion module.
"""

import os
import sys
import pytest
import pandas as pd
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_ingestion import load_raw_data, validate_schema, validate_data_quality, split_data, drop_leaky_columns


@pytest.fixture
def sample_data():
    """Create a minimal valid DataFrame for testing."""
    return pd.DataFrame({
        "UDI": range(1, 101),
        "Product ID": [f"P{i}" for i in range(100)],
        "Type": ["L"] * 60 + ["M"] * 30 + ["H"] * 10,
        "Air temperature [K]": [300.0 + i * 0.01 for i in range(100)],
        "Process temperature [K]": [310.0 + i * 0.01 for i in range(100)],
        "Rotational speed [rpm]": [1500] * 100,
        "Torque [Nm]": [40.0] * 100,
        "Tool wear [min]": list(range(100)),
        "Machine failure": [0] * 90 + [1] * 10,
        "TWF": [0] * 95 + [1] * 5,
        "HDF": [0] * 92 + [1] * 8,
        "PWF": [0] * 93 + [1] * 7,
        "OSF": [0] * 94 + [1] * 6,
        "RNF": [0] * 99 + [1] * 1,
    })


class TestLoadRawData:
    def test_load_existing_file(self, sample_data, tmp_path):
        filepath = tmp_path / "test.csv"
        sample_data.to_csv(filepath, index=False)
        df = load_raw_data(str(filepath))
        assert len(df) == 100
        assert len(df.columns) == 14

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_raw_data("/nonexistent/path.csv")


class TestValidateSchema:
    def test_valid_schema(self, sample_data):
        assert validate_schema(sample_data) is True

    def test_missing_columns(self, sample_data):
        df = sample_data.drop(columns=["Machine failure"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)


class TestDataQuality:
    def test_quality_pass(self, sample_data):
        report = validate_data_quality(sample_data)
        assert report["total_rows"] == 100
        assert report["total_missing"] == 0
        assert report["quality_status"] == "PASS"

    def test_detects_missing_values(self, sample_data):
        sample_data.loc[0, "Air temperature [K]"] = None
        report = validate_data_quality(sample_data)
        assert report["total_missing"] > 0


class TestSplitData:
    def test_split_sizes(self, sample_data):
        train, test = split_data(sample_data, test_size=0.2, random_state=42)
        assert len(train) == 80
        assert len(test) == 20

    def test_stratification(self, sample_data):
        train, test = split_data(sample_data, test_size=0.2, random_state=42)
        train_rate = train["Machine failure"].mean()
        test_rate = test["Machine failure"].mean()
        assert abs(train_rate - test_rate) < 0.05


class TestDropLeakyColumns:
    def test_removes_failure_mode_cols(self, sample_data):
        result = drop_leaky_columns(sample_data)
        for col in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
            assert col not in result.columns

    def test_removes_identifier_cols(self, sample_data):
        result = drop_leaky_columns(sample_data)
        assert "UDI" not in result.columns
        assert "Product ID" not in result.columns

    def test_keeps_features_and_target(self, sample_data):
        result = drop_leaky_columns(sample_data)
        assert "Machine failure" in result.columns
        assert "Air temperature [K]" in result.columns
        assert "Type" in result.columns
