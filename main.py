"""
SmartFlush Predictive Model - Main Entry Point
Predictive flush-volume optimization from sensor data
"""

import logging
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.utils import load_config, setup_logging, create_output_directories
from src.data_loading import load_data_from_excel
from src.eda import perform_eda
from src.utils import scale_features, create_polynomial_features, calculate_vif, remove_high_vif_features
from src.models import train_models
from src.metrics import evaluate_model, print_metrics_report, MetricsTracker

logger = logging.getLogger(__name__)


def main(config_path: str = "config.yaml", data_path: str = None, target_column: str = "flush_volume"):
    """
    Main execution pipeline for SmartFlush predictive modeling
    
    Args:
        config_path: Path to configuration file
        data_path: Path to data file (overrides config)
        target_column: Name of target column
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    setup_logging(config)
    logger.info("="*60)
    logger.info("SmartFlush Predictive Model - Starting")
    logger.info("="*60)
    
    # Create output directories
    create_output_directories(config)
    
    # Load data
    data_config = config.get('data', {})
    if data_path is None:
        data_path = data_config.get('input_path', 'data/sensor_data.xlsx')
    
    logger.info(f"Loading data from {data_path}")
    
    try:
        X, y = load_data_from_excel(
            file_path=data_path,
            target_column=target_column,
            clean=True
        )
        logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please place your sensor data Excel file at the specified path")
        logger.info("Expected columns: sensor features + target column (flush_volume)")
        return
    
    # Exploratory Data Analysis
    logger.info("Performing EDA")
    output_config = config.get('output', {})
    eda_output_dir = output_config.get('plot_path', 'reports/figures/')
    
    eda_report = perform_eda(X, y, config, eda_output_dir)
    logger.info(f"EDA complete: {len(X.columns)} features analyzed")
    
    # Feature Engineering
    logger.info("Feature engineering started")
    
    # VIF analysis and feature removal
    features_config = config.get('features', {})
    vif_threshold = features_config.get('vif_threshold', 10.0)
    
    logger.info("Calculating VIF and removing high multicollinearity")
    X_vif, removed_features = remove_high_vif_features(X, vif_threshold)
    logger.info(f"Removed {len(removed_features)} features with high VIF: {removed_features}")
    
    # Polynomial features
    poly_degree = features_config.get('polynomial_degree', 2)
    if poly_degree > 1:
        logger.info(f"Creating polynomial features (degree={poly_degree})")
        X_poly, poly_transformer = create_polynomial_features(X_vif, degree=poly_degree)
    else:
        X_poly = X_vif
    
    # Feature scaling
    scaling_method = features_config.get('scaling_method', 'standard')
    logger.info(f"Scaling features using {scaling_method} method")
    X_scaled, scaler = scale_features(X_poly, method=scaling_method)
    
    logger.info(f"Feature engineering complete: {X_scaled.shape[1]} features")
    
    # Train-test split
    test_size = data_config.get('test_size', 0.2)
    random_state = data_config.get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train-test split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Create validation set from training data for ANN
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )
    
    # Train models
    logger.info("Training models")
    trainer = train_models(X_train_sub, y_train_sub, config, X_val, y_val)
    
    # Evaluate models
    logger.info("Evaluating models")
    metrics_tracker = MetricsTracker()
    
    for model_name in trainer.best_models.keys():
        logger.info(f"Evaluating {model_name}")
        
        # Predictions
        y_pred = trainer.predict(model_name, X_test.values)
        
        # Calculate metrics
        metrics = evaluate_model(y_test.values, y_pred, config)
        metrics_tracker.add_metrics(model_name, metrics)
        
        # Print report
        print_metrics_report(metrics, model_name.upper())
    
    # Compare models
    logger.info("Comparing models")
    comparison = metrics_tracker.compare_models()
    
    print("\n" + "="*60)
    print("  Model Comparison Summary")
    print("="*60)
    print(f"\nBest MAE:          {comparison['best_mae'][0]} ({comparison['best_mae'][1]:.4f})")
    print(f"Best R²:           {comparison['best_r2'][0]} ({comparison['best_r2'][1]:.4f})")
    print(f"Best Safe Flush:   {comparison['best_safe_flush'][0]} ({comparison['best_safe_flush'][1]:.2%})")
    print(f"Best Water Savings: {comparison['best_water_savings'][0]} ({comparison['best_water_savings'][1]:.2f}%)")
    print("="*60 + "\n")
    
    # Save results
    output_path = data_config.get('output_path', 'results/')
    metrics_tracker.save_comparison(f"{output_path}model_comparison.csv")
    
    # Save models
    if output_config.get('save_models', True):
        model_path = output_config.get('model_path', 'results/models/')
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        for model_name in trainer.best_models.keys():
            trainer.save_model(model_name, f"{model_path}{model_name}")
            logger.info(f"Saved {model_name} model")
    
    logger.info("="*60)
    logger.info("SmartFlush Predictive Model - Complete")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartFlush Predictive Model")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to data file (overrides config)")
    parser.add_argument("--target", type=str, default="flush_volume",
                       help="Name of target column")
    
    args = parser.parse_args()
    
    main(config_path=args.config, data_path=args.data, target_column=args.target)
