"""
Metrics Module
Custom metrics for flush-volume optimization including safe_flush_accuracy, MAE, and water savings
"""

import logging
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def safe_flush_accuracy(y_true: np.ndarray, 
                       y_pred: np.ndarray,
                       threshold: float = 0.95) -> float:
    """
    Calculate safe flush accuracy
    Safe flush is when predicted volume is at least threshold * actual volume
    
    Args:
        y_true: True flush volumes
        y_pred: Predicted flush volumes
        threshold: Safety threshold (default 0.95 means predicted should be >= 95% of actual)
        
    Returns:
        Accuracy as fraction of safe flushes
    """
    safe_flushes = y_pred >= (threshold * y_true)
    accuracy = np.mean(safe_flushes)
    
    logger.info(f"Safe flush accuracy (threshold={threshold}): {accuracy:.4f}")
    
    return accuracy


def water_savings(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 baseline_volume: float = 6.0) -> Dict[str, float]:
    """
    Calculate water savings metrics
    
    Args:
        y_true: True flush volumes needed
        y_pred: Predicted flush volumes
        baseline_volume: Baseline flush volume (liters per flush)
        
    Returns:
        Dictionary with water savings metrics
    """
    n_samples = len(y_true)
    
    # Total water usage
    baseline_total = baseline_volume * n_samples
    actual_total = np.sum(y_true)
    predicted_total = np.sum(y_pred)
    
    # Savings calculations
    baseline_savings = baseline_total - actual_total
    predicted_savings = baseline_total - predicted_total
    
    # Efficiency metrics
    baseline_efficiency = (baseline_savings / baseline_total) * 100 if baseline_total > 0 else 0
    predicted_efficiency = (predicted_savings / baseline_total) * 100 if baseline_total > 0 else 0
    
    # Over/under estimation
    waste = np.maximum(0, y_pred - y_true)
    shortage = np.maximum(0, y_true - y_pred)
    
    metrics = {
        'baseline_total_liters': baseline_total,
        'actual_total_liters': actual_total,
        'predicted_total_liters': predicted_total,
        'baseline_savings_liters': baseline_savings,
        'predicted_savings_liters': predicted_savings,
        'baseline_efficiency_percent': baseline_efficiency,
        'predicted_efficiency_percent': predicted_efficiency,
        'average_waste_per_flush': np.mean(waste),
        'average_shortage_per_flush': np.mean(shortage),
        'total_waste': np.sum(waste),
        'total_shortage': np.sum(shortage),
        'n_samples': n_samples
    }
    
    logger.info(f"Water savings - Predicted efficiency: {predicted_efficiency:.2f}%, "
               f"Total savings: {predicted_savings:.2f}L")
    
    return metrics


def calculate_regression_metrics(y_true: np.ndarray,
                                 y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate standard regression metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with regression metrics
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.inf
    }
    
    logger.info(f"Regression metrics - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
    
    return metrics


def evaluate_model(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  config: Optional[dict] = None) -> Dict[str, float]:
    """
    Comprehensive model evaluation with all metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        config: Configuration dictionary
        
    Returns:
        Dictionary with all evaluation metrics
    """
    logger.info("Evaluating model performance")
    
    # Get config parameters
    if config is None:
        config = {}
    
    metrics_config = config.get('metrics', {})
    safe_threshold = metrics_config.get('safe_flush_threshold', 0.95)
    baseline = metrics_config.get('water_savings_baseline', 6.0)
    
    # Calculate all metrics
    metrics = {}
    
    # Regression metrics
    metrics.update(calculate_regression_metrics(y_true, y_pred))
    
    # Safe flush accuracy
    metrics['safe_flush_accuracy'] = safe_flush_accuracy(y_true, y_pred, safe_threshold)
    
    # Water savings
    water_metrics = water_savings(y_true, y_pred, baseline)
    metrics.update({f'water_{k}': v for k, v in water_metrics.items()})
    
    logger.info("Model evaluation complete")
    
    return metrics


def print_metrics_report(metrics: Dict[str, float], model_name: str = "Model"):
    """
    Print formatted metrics report
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"  {model_name} - Performance Metrics")
    print(f"{'='*60}\n")
    
    # Regression metrics
    print("Regression Metrics:")
    print(f"  MAE:  {metrics.get('mae', 0):.4f}")
    print(f"  RMSE: {metrics.get('rmse', 0):.4f}")
    print(f"  R²:   {metrics.get('r2', 0):.4f}")
    print(f"  MAPE: {metrics.get('mape', 0):.2f}%")
    
    # Safe flush metrics
    print(f"\nSafe Flush Metrics:")
    print(f"  Safe Flush Accuracy: {metrics.get('safe_flush_accuracy', 0):.2%}")
    
    # Water savings metrics
    print(f"\nWater Savings Metrics:")
    print(f"  Predicted Efficiency: {metrics.get('water_predicted_efficiency_percent', 0):.2f}%")
    print(f"  Total Savings:        {metrics.get('water_predicted_savings_liters', 0):.2f}L")
    print(f"  Avg Waste/Flush:      {metrics.get('water_average_waste_per_flush', 0):.4f}L")
    print(f"  Avg Shortage/Flush:   {metrics.get('water_average_shortage_per_flush', 0):.4f}L")
    
    print(f"\n{'='*60}\n")


class MetricsTracker:
    """Track and store metrics across multiple models"""
    
    def __init__(self):
        """Initialize metrics tracker"""
        self.metrics = {}
        
    def add_metrics(self, model_name: str, metrics: Dict[str, float]):
        """
        Add metrics for a model
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
        """
        self.metrics[model_name] = metrics
        logger.info(f"Added metrics for {model_name}")
        
    def get_best_model(self, metric: str = 'mae', 
                      maximize: bool = False) -> tuple:
        """
        Get best model based on specified metric
        
        Args:
            metric: Metric to use for comparison
            maximize: Whether to maximize the metric (default: minimize)
            
        Returns:
            Tuple of (model_name, metric_value)
        """
        if not self.metrics:
            return None, None
            
        best_model = None
        best_value = float('-inf') if maximize else float('inf')
        
        for model_name, model_metrics in self.metrics.items():
            if metric not in model_metrics:
                continue
                
            value = model_metrics[metric]
            
            if maximize:
                if value > best_value:
                    best_value = value
                    best_model = model_name
            else:
                if value < best_value:
                    best_value = value
                    best_model = model_name
        
        logger.info(f"Best model by {metric}: {best_model} ({best_value:.4f})")
        
        return best_model, best_value
    
    def compare_models(self) -> dict:
        """
        Compare all tracked models
        
        Returns:
            Dictionary with comparison results
        """
        import pandas as pd
        
        if not self.metrics:
            return {}
        
        # Create comparison dataframe
        df = pd.DataFrame(self.metrics).T
        
        comparison = {
            'all_metrics': df,
            'best_mae': self.get_best_model('mae', maximize=False),
            'best_r2': self.get_best_model('r2', maximize=True),
            'best_safe_flush': self.get_best_model('safe_flush_accuracy', maximize=True),
            'best_water_savings': self.get_best_model('water_predicted_efficiency_percent', maximize=True)
        }
        
        return comparison
    
    def save_comparison(self, filepath: str):
        """
        Save metrics comparison to CSV
        
        Args:
            filepath: Path to save CSV file
        """
        import pandas as pd
        
        if not self.metrics:
            logger.warning("No metrics to save")
            return
            
        df = pd.DataFrame(self.metrics).T
        df.to_csv(filepath)
        logger.info(f"Metrics comparison saved to {filepath}")
