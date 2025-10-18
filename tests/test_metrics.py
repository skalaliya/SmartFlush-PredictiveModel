"""
Behavioural tests for SmartFlush metrics module.
"""

from pathlib import Path

import numpy as np

from src import metrics


def test_safe_flush_accuracy_basic():
    y_true = np.array([4.0, 4.0])
    y_pred = np.array([4.2, 3.6])
    score = metrics.safe_flush_accuracy(y_true, y_pred, threshold=0.95)
    assert 0.0 <= score <= 1.0


def test_evaluate_models_and_save_reports(tmp_path: Path):
    eval_data = {
        "y_test": np.array([1, 2, 2, 3]),
        "predictions": {
            "ridge": np.array([2, 3, 2, 3]),
            "mlr": np.array([1, 2, 3, 3]),
        },
    }
    config = {
        "evaluation": {
            "safe_flush_threshold": 0.95,
            "baseline_flush_volume": 6.0,
            "competitor": {"safe_accuracy": 0.56, "mae": 0.93},
        }
    }

    results = metrics.evaluate_models(eval_data, config)
    assert "summary_table" in results
    assert set(eval_data["predictions"].keys()).issubset(results.keys())

    metrics.save_reports(results, {"tables_dir": tmp_path})
    assert (tmp_path / "model_summary.csv").exists()
