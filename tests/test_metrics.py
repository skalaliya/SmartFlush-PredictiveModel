"""
Unit tests for metrics module.
"""

import pytest
import numpy as np
import pandas as pd
from src.metrics import (
    calculate_safe_flush_accuracy,
    calculate_mae,
    calculate_accuracy,
    generate_confusion_matrix,
    evaluate_model,
    create_benchmark_table,
    calculate_improvement_over_baseline
)


class TestSafeFlushAccuracy:
    """Tests for safe flush accuracy calculation."""
    
    def test_safe_flush_accuracy_perfect(self):
        """Test with perfect safe predictions."""
        y_true = np.array([5, 6, 7, 8, 9])
        y_pred = np.array([5, 6, 7, 8, 9])
        
        accuracy = calculate_safe_flush_accuracy(y_true, y_pred)
        
        assert accuracy == 1.0
    
    def test_safe_flush_accuracy_over_predict(self):
        """Test with over-predictions (safe)."""
        y_true = np.array([5, 6, 7, 8, 9])
        y_pred = np.array([6, 7, 8, 9, 10])
        
        accuracy = calculate_safe_flush_accuracy(y_true, y_pred)
        
        assert accuracy == 1.0
    
    def test_safe_flush_accuracy_under_predict(self):
        """Test with under-predictions (unsafe)."""
        y_true = np.array([5, 6, 7, 8, 9])
        y_pred = np.array([4, 5, 6, 7, 8])
        
        accuracy = calculate_safe_flush_accuracy(y_true, y_pred)
        
        assert accuracy == 0.0
    
    def test_safe_flush_accuracy_mixed(self):
        """Test with mixed predictions."""
        y_true = np.array([5, 6, 7, 8, 9])
        y_pred = np.array([5, 7, 6, 9, 10])
        
        accuracy = calculate_safe_flush_accuracy(y_true, y_pred)
        
        assert 0 < accuracy < 1
    
    def test_safe_flush_accuracy_mismatched(self):
        """Test with mismatched lengths."""
        y_true = np.array([5, 6, 7])
        y_pred = np.array([5, 6])
        
        with pytest.raises(ValueError):
            calculate_safe_flush_accuracy(y_true, y_pred)


class TestMAE:
    """Tests for Mean Absolute Error calculation."""
    
    def test_mae_zero(self):
        """Test MAE with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        mae = calculate_mae(y_true, y_pred)
        
        assert mae == 0.0
    
    def test_mae_positive(self):
        """Test MAE with non-zero error."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])
        
        mae = calculate_mae(y_true, y_pred)
        
        assert mae == 1.0
    
    def test_mae_mixed_errors(self):
        """Test MAE with mixed errors."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 2, 2, 4, 6])
        
        mae = calculate_mae(y_true, y_pred)
        
        assert mae == pytest.approx(0.6)


class TestAccuracy:
    """Tests for classification accuracy."""
    
    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        acc = calculate_accuracy(y_true, y_pred)
        
        assert acc == 1.0
    
    def test_accuracy_half(self):
        """Test accuracy with 50% correct."""
        y_true = np.array([1, 2, 3, 4, 5, 6])
        y_pred = np.array([1, 2, 3, 5, 6, 7])
        
        acc = calculate_accuracy(y_true, y_pred)
        
        assert acc == 0.5


class TestConfusionMatrix:
    """Tests for confusion matrix generation."""
    
    def test_confusion_matrix_binary(self):
        """Test confusion matrix for binary classification."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        cm = generate_confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)
    
    def test_confusion_matrix_multiclass(self):
        """Test confusion matrix for multiclass classification."""
        y_true = np.array([1, 2, 3, 1, 2, 3])
        y_pred = np.array([1, 2, 3, 2, 2, 1])
        
        cm = generate_confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (3, 3)
        assert cm.sum() == len(y_true)


class TestEvaluateModel:
    """Tests for comprehensive model evaluation."""
    
    def test_evaluate_model(self):
        """Test complete model evaluation."""
        y_true = np.array([1, 2, 3, 4, 5] * 10)
        y_pred = np.array([1, 2, 3, 4, 5] * 10)
        
        metrics = evaluate_model(y_true, y_pred, 'TestModel')
        
        assert 'model' in metrics
        assert 'accuracy' in metrics
        assert 'safe_flush_accuracy' in metrics
        assert 'mae' in metrics
        assert metrics['model'] == 'TestModel'
        assert metrics['accuracy'] == 1.0
        assert metrics['mae'] == 0.0


class TestBenchmarkTable:
    """Tests for benchmark table creation."""
    
    def test_create_benchmark_table(self):
        """Test benchmark table creation."""
        results = [
            {'model': 'Model1', 'accuracy': 0.8, 'safe_flush_accuracy': 0.85, 'mae': 0.5},
            {'model': 'Model2', 'accuracy': 0.75, 'safe_flush_accuracy': 0.80, 'mae': 0.6}
        ]
        baseline = {'model': 'Baseline', 'accuracy': 0.7, 'safe_flush_accuracy': 0.75, 'mae': 0.7}
        
        table = create_benchmark_table(results, baseline)
        
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 3
        assert 'model' in table.columns
        assert 'accuracy' in table.columns
    
    def test_create_benchmark_table_no_baseline(self):
        """Test benchmark table without baseline."""
        results = [
            {'model': 'Model1', 'accuracy': 0.8, 'mae': 0.5}
        ]
        
        table = create_benchmark_table(results, None)
        
        assert len(table) == 1


class TestImprovementCalculation:
    """Tests for improvement calculation over baseline."""
    
    def test_calculate_improvement_positive(self):
        """Test positive improvement calculation."""
        model_metrics = {'accuracy': 0.8, 'safe_flush_accuracy': 0.85, 'mae': 0.4}
        baseline_metrics = {'accuracy': 0.7, 'safe_flush_accuracy': 0.75, 'mae': 0.5}
        
        improvements = calculate_improvement_over_baseline(model_metrics, baseline_metrics)
        
        assert 'accuracy_improvement_pct' in improvements
        assert 'safe_flush_accuracy_improvement_pct' in improvements
        assert 'mae_reduction_pct' in improvements
        assert improvements['accuracy_improvement_pct'] > 0
        assert improvements['mae_reduction_pct'] > 0
    
    def test_calculate_improvement_negative(self):
        """Test negative improvement calculation."""
        model_metrics = {'accuracy': 0.6, 'mae': 0.6}
        baseline_metrics = {'accuracy': 0.7, 'mae': 0.5}
        
        improvements = calculate_improvement_over_baseline(model_metrics, baseline_metrics)
        
        assert improvements['accuracy_improvement_pct'] < 0
        assert improvements['mae_reduction_pct'] < 0
