"""
Custom metrics and reporting utilities for the SmartFlush predictive modeling project.

The functions in this module provide scaffolding for:
  * Safe flush accuracy (cleanliness-focused KPI).
  * Mean Absolute Error (water efficiency).
  * Confusion matrices and classification reports.
  * Comparisons against a baseline competitor model.
  * Environmental and economic impact calculations (hotel scenario).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from . import utils

LOGGER = logging.getLogger(__name__)


def safe_flush_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.95) -> float:
    """
    Compute the percentage of predictions that meet or exceed the safe flush threshold.

    Args:
        y_true: Ground-truth flush volumes (liters or class labels mapped to liters).
        y_pred: Predicted flush volumes in the same units as `y_true`.
        threshold: Minimum ratio (prediction / actual) required to deem a flush safe.

    Returns:
        Float between 0 and 1 representing safe flush accuracy.
    """
    if y_true.size == 0:
        return 0.0

    safe_events = y_pred >= threshold * y_true
    accuracy = float(np.mean(safe_events))
    LOGGER.debug("Safe flush accuracy computed: %.4f (threshold=%.2f)", accuracy, threshold)
    return accuracy


def water_efficiency_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Convenience wrapper for mean absolute error expressed in liters.
    """
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def compare_against_competitor(metric_value: float, competitor_value: float, metric_name: str) -> Dict[str, float]:
    """
    Compare SmartFlush metric to the baseline competitor.
    """
    delta = metric_value - competitor_value
    LOGGER.debug("Metric %s comparison: ours=%.4f competitor=%.4f delta=%.4f", metric_name, metric_value, competitor_value, delta)
    return {
        "ours": metric_value,
        "competitor": competitor_value,
        "delta": delta,
    }


def build_confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int]) -> pd.DataFrame:
    """
    Build a confusion matrix as a pandas DataFrame.
    """
    matrix = sk_metrics.confusion_matrix(y_true, y_pred)
    labels = np.unique(np.concatenate([np.asarray(list(y_true)), np.asarray(list(y_pred))]))
    return pd.DataFrame(matrix, index=labels, columns=labels)


def classification_report(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, Any]:
    """
    Produce a classification report dictionary (precision/recall/F1).
    """
    report = sk_metrics.classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return report


def evaluate_models(artifacts: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate trained models using custom metrics and return organised results.
    """
    LOGGER.info("Evaluating models using SmartFlush metrics.")
    evaluation_cfg = config.get("evaluation", {})
    safe_threshold = evaluation_cfg.get("safe_flush_threshold", 0.95)
    competitor_cfg = evaluation_cfg.get("competitor", {})
    baseline_flush_volume = evaluation_cfg.get("baseline_flush_volume", 6.0)

    if hasattr(artifacts, "evaluation_data"):
        eval_data = artifacts.evaluation_data
    else:
        eval_data = artifacts

    y_true = np.asarray(eval_data.get("y_test"))
    predictions_map = eval_data.get("predictions", {})

    results: Dict[str, Any] = {}
    summary_rows = []

    for model_name, y_pred in predictions_map.items():
        y_pred_arr = np.asarray(y_pred, dtype=float)

        metrics_dict = {}
        metrics_dict["safe_flush_accuracy"] = safe_flush_accuracy(y_true, y_pred_arr, safe_threshold)
        metrics_dict["mae"] = sk_metrics.mean_absolute_error(y_true, y_pred_arr)
        metrics_dict["water_mae"] = water_efficiency_mae(y_true, y_pred_arr)
        mse = sk_metrics.mean_squared_error(y_true, y_pred_arr)
        metrics_dict["rmse"] = float(np.sqrt(mse))
        metrics_dict["r2"] = sk_metrics.r2_score(y_true, y_pred_arr)

        discrete_pred = np.allclose(y_pred_arr, np.round(y_pred_arr))
        if discrete_pred:
            y_pred_discrete = np.round(y_pred_arr).astype(int)
            conf_matrix = build_confusion_matrix(y_true, y_pred_discrete)
            class_report = classification_report(y_true, y_pred_discrete)
        else:
            conf_matrix = None
            class_report = None

        water_savings = utils.compute_water_savings(y_pred_arr, baseline_flush_volume, config)

        metrics_dict.update(
            {
                "confusion_matrix": conf_matrix,
                "classification_report": class_report,
                "water_savings": water_savings,
            }
        )

        if competitor_cfg:
            competitor_accuracy = competitor_cfg.get("safe_accuracy", 0.0)
            metrics_dict["safe_flush_vs_competitor"] = compare_against_competitor(
                metrics_dict["safe_flush_accuracy"], competitor_accuracy, "safe_flush_accuracy"
            )
            competitor_mae = competitor_cfg.get("mae", 0.0)
            metrics_dict["mae_vs_competitor"] = compare_against_competitor(metrics_dict["mae"], competitor_mae, "mae")

        results[model_name] = metrics_dict
        summary_rows.append(
            {
                "model": model_name,
                "safe_flush_accuracy": metrics_dict["safe_flush_accuracy"],
                "mae": metrics_dict["mae"],
                "rmse": metrics_dict["rmse"],
                "water_savings_percent": water_savings["savings_percent"],
            }
        )

    results["summary_table"] = pd.DataFrame(summary_rows).sort_values("safe_flush_accuracy", ascending=False)
    return results


def save_reports(evaluation_results: Dict[str, Any], output_paths: Dict[str, Path]) -> None:
    """
    Persist evaluation outputs (tables, figures, reports) to disk.
    """
    LOGGER.info("Saving evaluation reports to %s", output_paths)
    tables_dir = output_paths.get("tables_dir", Path("results/tables"))
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = evaluation_results.get("summary_table")
    if isinstance(summary, pd.DataFrame):
        summary.to_csv(tables_dir / "model_summary.csv", index=False)

    for model_name, metrics_dict in evaluation_results.items():
        if not isinstance(metrics_dict, dict):
            continue
        conf_matrix = metrics_dict.get("confusion_matrix")
        if conf_matrix is not None:
            conf_matrix.to_csv(tables_dir / f"{model_name}_confusion_matrix.csv")
        class_report = metrics_dict.get("classification_report")
        if class_report is not None:
            pd.DataFrame(class_report).to_csv(tables_dir / f"{model_name}_classification_report.csv")
