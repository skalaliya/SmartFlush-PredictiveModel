"""
Metrics module for SmartFlush Predictive Model.

This module provides:
- Safe flush accuracy calculation
- Mean Absolute Error (MAE)
- Confusion matrix generation
- Classification reports
- Benchmarking against competitor baseline
- Model performance comparison tables
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


def calculate_safe_flush_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: int = 0
) -> float:
    """
    Calculate safe flush accuracy.
    
    A prediction is considered "safe" if it's equal to or greater than the true value
    (plus optional tolerance), ensuring adequate flushing while potentially using more water.
    
    Args:
        y_true: True flush levels
        y_pred: Predicted flush levels
        tolerance: Tolerance for predictions below true value (default: 0)
        
    Returns:
        Safe flush accuracy as a float between 0 and 1
        
    Example:
        >>> safe_acc = calculate_safe_flush_accuracy(y_test, predictions)
        >>> print(f"Safe flush accuracy: {safe_acc:.2%}")
    """
    try:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        # A prediction is safe if: y_pred >= y_true - tolerance
        safe_predictions = y_pred >= (y_true - tolerance)
        safe_accuracy = np.mean(safe_predictions)
        
        logger.info(f"Safe flush accuracy (tolerance={tolerance}): {safe_accuracy:.4f}")
        return float(safe_accuracy)
        
    except Exception as e:
        logger.error(f"Error calculating safe flush accuracy: {e}")
        raise


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE value
        
    Example:
        >>> mae = calculate_mae(y_test, predictions)
    """
    try:
        mae_value = mean_absolute_error(y_true, y_pred)
        logger.info(f"MAE: {mae_value:.4f}")
        return float(mae_value)
        
    except Exception as e:
        logger.error(f"Error calculating MAE: {e}")
        raise


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy as a float between 0 and 1
        
    Example:
        >>> acc = calculate_accuracy(y_test, predictions)
    """
    try:
        accuracy = accuracy_score(y_true, y_pred)
        logger.info(f"Accuracy: {accuracy:.4f}")
        return float(accuracy)
        
    except Exception as e:
        logger.error(f"Error calculating accuracy: {e}")
        raise


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None
) -> np.ndarray:
    """
    Generate confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names (optional)
        
    Returns:
        Confusion matrix as numpy array
        
    Example:
        >>> cm = generate_confusion_matrix(y_test, predictions)
    """
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        logger.info(f"Confusion matrix shape: {cm.shape}")
        return cm
        
    except Exception as e:
        logger.error(f"Error generating confusion matrix: {e}")
        raise


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    labels: Optional[List] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Confusion Matrix'
) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the figure
        labels: List of label names
        figsize: Figure size
        title: Plot title
        
    Example:
        >>> plot_confusion_matrix(y_test, predictions, 'results/confusion_matrix.png')
    """
    try:
        logger.info("Plotting confusion matrix")
        
        cm = generate_confusion_matrix(y_true, y_pred, labels)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels if labels else 'auto',
            yticklabels=labels if labels else 'auto',
            ax=ax
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
        raise


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None
) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of target classes
        save_path: Path to save report as text file
        
    Returns:
        Classification report as string
        
    Example:
        >>> report = generate_classification_report(y_test, predictions)
        >>> print(report)
    """
    try:
        logger.info("Generating classification report")
        
        report = classification_report(y_true, y_pred, target_names=target_names)
        
        logger.info(f"Classification report:\n{report}")
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Classification report saved to {save_path}")
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating classification report: {e}")
        raise


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Model'
) -> Dict[str, float]:
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        y_true: True values/labels
        y_pred: Predicted values/labels
        model_name: Name of the model for logging
        
    Returns:
        Dictionary with all evaluation metrics
        
    Example:
        >>> metrics = evaluate_model(y_test, predictions, 'Ridge Regression')
    """
    try:
        logger.info(f"Evaluating model: {model_name}")
        
        metrics = {
            'model': model_name,
            'accuracy': calculate_accuracy(y_true, y_pred),
            'safe_flush_accuracy': calculate_safe_flush_accuracy(y_true, y_pred),
            'mae': calculate_mae(y_true, y_pred)
        }
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"Safe Flush Acc: {metrics['safe_flush_accuracy']:.4f}, "
                   f"MAE: {metrics['mae']:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise


def create_benchmark_table(
    model_results: List[Dict[str, float]],
    competitor_baseline: Optional[Dict[str, float]] = None,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Create benchmark comparison table for all models.
    
    Args:
        model_results: List of dictionaries with model metrics
        competitor_baseline: Dictionary with competitor baseline metrics
        save_path: Path to save table as CSV
        
    Returns:
        DataFrame with benchmark comparison
        
    Example:
        >>> results = [
        ...     evaluate_model(y_test, ridge_preds, 'Ridge'),
        ...     evaluate_model(y_test, lr_preds, 'LogReg')
        ... ]
        >>> baseline = {'model': 'Competitor', 'accuracy': 0.31, 'safe_flush_accuracy': 0.56, 'mae': 0.93}
        >>> table = create_benchmark_table(results, baseline)
    """
    try:
        logger.info("Creating benchmark table")
        
        # Combine all results
        all_results = model_results.copy()
        if competitor_baseline:
            all_results.insert(0, competitor_baseline)
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Format columns
        if 'accuracy' in df.columns:
            df['accuracy'] = df['accuracy'].round(4)
        if 'safe_flush_accuracy' in df.columns:
            df['safe_flush_accuracy'] = df['safe_flush_accuracy'].round(4)
        if 'mae' in df.columns:
            df['mae'] = df['mae'].round(4)
        
        logger.info(f"Benchmark table created with {len(df)} models")
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            logger.info(f"Benchmark table saved to {save_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating benchmark table: {e}")
        raise


def plot_model_comparison(
    benchmark_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot model comparison bar chart.
    
    Args:
        benchmark_df: DataFrame with model benchmark results
        save_path: Path to save the figure
        figsize: Figure size
        
    Example:
        >>> plot_model_comparison(benchmark_df, 'results/model_comparison.png')
    """
    try:
        logger.info("Plotting model comparison")
        
        # Prepare data
        metrics_to_plot = ['accuracy', 'safe_flush_accuracy', 'mae']
        available_metrics = [m for m in metrics_to_plot if m in benchmark_df.columns]
        
        if not available_metrics:
            logger.warning("No metrics available for plotting")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            # Sort by metric value
            sorted_df = benchmark_df.sort_values(metric, ascending=(metric != 'mae'))
            
            # Create bar plot
            bars = ax.bar(
                range(len(sorted_df)),
                sorted_df[metric],
                color=['red' if 'Competitor' in str(model) else 'steelblue' 
                       for model in sorted_df['model']],
                alpha=0.7
            )
            
            ax.set_xticks(range(len(sorted_df)))
            ax.set_xticklabels(sorted_df['model'], rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, sorted_df[metric])):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{val:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
        
        plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting model comparison: {e}")
        raise


def calculate_improvement_over_baseline(
    model_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate improvement percentages over baseline.
    
    Args:
        model_metrics: Dictionary with model metrics
        baseline_metrics: Dictionary with baseline metrics
        
    Returns:
        Dictionary with improvement percentages for each metric
        
    Example:
        >>> improvements = calculate_improvement_over_baseline(model_metrics, baseline)
    """
    try:
        improvements = {}
        
        for metric in ['accuracy', 'safe_flush_accuracy']:
            if metric in model_metrics and metric in baseline_metrics:
                baseline_val = baseline_metrics[metric]
                model_val = model_metrics[metric]
                
                if baseline_val > 0:
                    improvement = ((model_val - baseline_val) / baseline_val) * 100
                    improvements[f'{metric}_improvement_pct'] = improvement
        
        # For MAE, lower is better
        if 'mae' in model_metrics and 'mae' in baseline_metrics:
            baseline_mae = baseline_metrics['mae']
            model_mae = model_metrics['mae']
            
            if baseline_mae > 0:
                reduction = ((baseline_mae - model_mae) / baseline_mae) * 100
                improvements['mae_reduction_pct'] = reduction
        
        logger.info(f"Improvements calculated: {improvements}")
        return improvements
        
    except Exception as e:
        logger.error(f"Error calculating improvements: {e}")
        raise
