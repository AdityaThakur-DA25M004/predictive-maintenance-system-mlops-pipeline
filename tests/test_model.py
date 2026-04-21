"""
Unit tests for the model training and evaluation module.
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_training import evaluate_model


@pytest.fixture
def trained_model():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series([0] * 180 + [1] * 20)
    model = RandomForestClassifier(n_estimators=10, random_state=42, class_weight="balanced")
    model.fit(X, y)
    return model, X, y


class TestEvaluateModel:
    def test_returns_all_metrics(self, trained_model):
        model, X, y = trained_model
        metrics = evaluate_model(model, X, y)
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "roc_auc" in metrics
        assert "true_positives" in metrics
        assert "true_negatives" in metrics

    def test_metrics_in_valid_range(self, trained_model):
        model, X, y = trained_model
        metrics = evaluate_model(model, X, y)
        for key in ["accuracy", "f1_score", "precision", "recall", "roc_auc"]:
            assert 0.0 <= metrics[key] <= 1.0

    def test_confusion_matrix_sums(self, trained_model):
        model, X, y = trained_model
        metrics = evaluate_model(model, X, y)
        total = (metrics["true_positives"] + metrics["true_negatives"] +
                 metrics["false_positives"] + metrics["false_negatives"])
        assert total == len(y)
