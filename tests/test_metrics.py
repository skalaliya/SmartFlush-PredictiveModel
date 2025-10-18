"""
Tests for metrics module
"""

import pytest
import numpy as np

from src.metrics import (
    safe_flush_accuracy, water_savings, calculate_regression_metrics,
    evaluate_model, MetricsTracker
)


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing"""
    np.random.seed(42)
    y_true = np.array([3.5, 4.0, 3.8, 4.2, 3.9, 4.5, 3.7, 4.1])
    y_pred = np.array([3.6, 3.9, 3.9, 4.0, 4.0, 4.3, 3.8, 4.2])
    return y_true, y_pred


def test_safe_flush_accuracy(sample_predictions):
    """Test safe flush accuracy calculation"""
    y_true, y_pred = sample_predictions
    
    accuracy = safe_flush_accuracy(y_true, y_pred, threshold=0.95)
    
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_safe_flush_accuracy_perfect():
    """Test safe flush accuracy with perfect predictions"""
    y_true = np.array([4.0, 4.0, 4.0])
    y_pred = np.array([4.0, 4.0, 4.0])
    
    accuracy = safe_flush_accuracy(y_true, y_pred, threshold=0.95)
    
    assert accuracy == 1.0


def test_safe_flush_accuracy_zero():
    """Test safe flush accuracy with all unsafe predictions"""
    y_true = np.array([4.0, 4.0, 4.0])
    y_pred = np.array([3.0, 3.0, 3.0])
    
    accuracy = safe_flush_accuracy(y_true, y_pred, threshold=0.95)
    
    assert accuracy == 0.0


def test_water_savings(sample_predictions):
    """Test water savings calculation"""
    y_true, y_pred = sample_predictions
    
    metrics = water_savings(y_true, y_pred, baseline_volume=6.0)
    
    assert isinstance(metrics, dict)
    assert 'baseline_total_liters' in metrics
    assert 'predicted_total_liters' in metrics
    assert 'predicted_savings_liters' in metrics
    assert 'predicted_efficiency_percent' in metrics
    assert metrics['n_samples'] == len(y_true)


def test_calculate_regression_metrics(sample_predictions):
    """Test regression metrics calculation"""
    y_true, y_pred = sample_predictions
    
    metrics = calculate_regression_metrics(y_true, y_pred)
    
    assert isinstance(metrics, dict)
    assert 'mae' in metrics
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert 'mape' in metrics
    
    # Check that metrics are reasonable
    assert metrics['mae'] >= 0
    assert metrics['mse'] >= 0
    assert metrics['rmse'] >= 0
    assert metrics['rmse'] == pytest.approx(np.sqrt(metrics['mse']))


def test_evaluate_model(sample_predictions):
    """Test comprehensive model evaluation"""
    y_true, y_pred = sample_predictions
    
    config = {
        'metrics': {
            'safe_flush_threshold': 0.95,
            'water_savings_baseline': 6.0
        }
    }
    
    metrics = evaluate_model(y_true, y_pred, config)
    
    assert isinstance(metrics, dict)
    assert 'mae' in metrics
    assert 'safe_flush_accuracy' in metrics
    assert any('water_' in key for key in metrics.keys())


def test_metrics_tracker():
    """Test MetricsTracker class"""
    tracker = MetricsTracker()
    
    # Add metrics for multiple models
    tracker.add_metrics('model1', {'mae': 0.5, 'r2': 0.85})
    tracker.add_metrics('model2', {'mae': 0.3, 'r2': 0.90})
    
    # Test best model retrieval
    best_model, best_value = tracker.get_best_model('mae', maximize=False)
    assert best_model == 'model2'
    assert best_value == 0.3
    
    best_model, best_value = tracker.get_best_model('r2', maximize=True)
    assert best_model == 'model2'
    assert best_value == 0.90


def test_metrics_tracker_comparison():
    """Test metrics comparison"""
    tracker = MetricsTracker()
    
    tracker.add_metrics('model1', {'mae': 0.5, 'r2': 0.85, 'safe_flush_accuracy': 0.90})
    tracker.add_metrics('model2', {'mae': 0.3, 'r2': 0.90, 'safe_flush_accuracy': 0.95})
    
    comparison = tracker.compare_models()
    
    assert isinstance(comparison, dict)
    assert 'all_metrics' in comparison
    assert 'best_mae' in comparison
    assert 'best_r2' in comparison
