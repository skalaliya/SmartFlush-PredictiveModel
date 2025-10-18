"""
Structural tests for SmartFlush metrics module.
"""

import numpy as np

from src import metrics


def test_safe_flush_accuracy_basic():
    y_true = np.array([4.0, 4.0])
    y_pred = np.array([4.2, 3.6])
    score = metrics.safe_flush_accuracy(y_true, y_pred, threshold=0.95)
    assert 0.0 <= score <= 1.0


def test_water_efficiency_mae_basic():
    y_true = np.array([4.0, 4.0])
    y_pred = np.array([4.5, 3.5])
    mae = metrics.water_efficiency_mae(y_true, y_pred)
    assert mae == 0.5


def test_compare_against_competitor_structure():
    result = metrics.compare_against_competitor(0.9, 0.8, "accuracy")
    assert set(result.keys()) == {"ours", "competitor", "delta"}


def test_unimplemented_hooks_raise():
    try:
        metrics.evaluate_models({}, {})
    except NotImplementedError:
        assert True
