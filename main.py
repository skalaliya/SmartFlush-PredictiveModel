"""
Main orchestration script for SmartFlush Predictive Model.

This script coordinates the complete pipeline:
1. Load and combine data from Excel files
2. Perform exploratory data analysis (EDA)
3. Train and evaluate multiple models (Ridge, LogReg, SVC, ANN)
4. Optimize probability thresholds for applicable models
5. Calculate water savings and cost impact for hotel scenario
6. Save plots, tables, and reports
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List
import yaml
import numpy as np
import pandas as pd
import colorlog

# Import custom modules
from src.data_loading import (
    load_and_combine_data,
    handle_missing_values,
    prepare_data
)
from src.eda import perform_eda, set_plot_style
from src.models import (
    train_ridge_model,
    train_logistic_regression_model,
    train_svc_model,
    train_ann_model,
    predict_with_adjustment,
    get_probability_predictions,
    optimize_probability_threshold,
    plot_learning_curves
)
from src.metrics import (
    evaluate_model,
    create_benchmark_table,
    plot_model_comparison,
    plot_confusion_matrix,
    generate_classification_report
)
from src.utils import (
    calculate_water_savings,
    estimate_annual_impact
)


def setup_logging(config: Dict) -> None:
    """
    Set up logging configuration with colors.
    
    Args:
        config: Configuration dictionary with logging settings
    """
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_file = config.get('logging', {}).get('file', 'results/smartflush.log')
    
    # Create log directory if needed
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatter with colors for console
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # File formatter without colors
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def load_config(config_path: str = 'config.yaml') -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)


def main():
    """
    Main execution function for the SmartFlush pipeline.
    """
    print("=" * 80)
    print("SmartFlush Predictive Model Pipeline")
    print("=" * 80)
    
    # Load configuration
    config = load_config('config.yaml')
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Starting SmartFlush pipeline")
    
    # Set plotting style
    set_plot_style()
    
    # Create output directories
    results_dir = Path(config['outputs']['results_dir'])
    reports_dir = Path(config['outputs']['reports_dir'])
    plots_dir = Path(config['outputs']['plots_dir'])
    
    for directory in [results_dir, reports_dir, plots_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    try:
        # ========================================================================
        # STEP 1: Load and Combine Data
        # ========================================================================
        logger.info("=" * 60)
        logger.info("STEP 1: Loading and Combining Data")
        logger.info("=" * 60)
        
        data_files = []
        combined_data_path = config['data']['combined_data']
        additional_data_path = config['data']['additional_data']
        
        if Path(combined_data_path).exists():
            data_files.append(combined_data_path)
        if Path(additional_data_path).exists():
            data_files.append(additional_data_path)
        
        if not data_files:
            logger.warning("No data files found. Creating synthetic sample data for demonstration.")
            # Create sample data for testing the pipeline
            np.random.seed(42)
            n_samples = 1000
            df = pd.DataFrame({
                'photodiode_1': np.random.randn(n_samples),
                'photodiode_2': np.random.randn(n_samples),
                'waste_level': np.random.randint(1, 6, n_samples),
                'sensor_1': np.random.randn(n_samples),
                'sensor_2': np.random.randn(n_samples),
                'flush_level': np.random.randint(1, 12, n_samples)
            })
        else:
            df = load_and_combine_data(data_files, how='concat')
        
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Handle missing values
        df = handle_missing_values(df, strategy='mean')
        
        # ========================================================================
        # STEP 2: Exploratory Data Analysis (EDA)
        # ========================================================================
        logger.info("=" * 60)
        logger.info("STEP 2: Performing Exploratory Data Analysis")
        logger.info("=" * 60)
        
        # Determine target column (adjust based on actual data)
        target_col = 'flush_level' if 'flush_level' in df.columns else df.columns[-1]
        logger.info(f"Using target column: {target_col}")
        
        # Perform comprehensive EDA
        eda_output_dir = plots_dir / 'eda'
        eda_results = perform_eda(
            df,
            target_col=target_col,
            output_dir=eda_output_dir
        )
        
        logger.info("EDA complete")
        
        # ========================================================================
        # STEP 3: Data Preparation
        # ========================================================================
        logger.info("=" * 60)
        logger.info("STEP 3: Preparing Data for Modeling")
        logger.info("=" * 60)
        
        preprocessing_config = config['preprocessing']
        data_dict = prepare_data(
            df,
            target_col=target_col,
            test_size=preprocessing_config['test_size'],
            random_state=preprocessing_config['random_state'],
            vif_threshold=preprocessing_config['vif_threshold'],
            apply_polynomial=False,  # Can be enabled via config
            apply_standardization=True
        )
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # ========================================================================
        # STEP 4: Train Models
        # ========================================================================
        logger.info("=" * 60)
        logger.info("STEP 4: Training Models")
        logger.info("=" * 60)
        
        all_results = []
        
        # 4.1 Ridge Regression
        logger.info("-" * 60)
        logger.info("Training Ridge Regression")
        logger.info("-" * 60)
        try:
            ridge_config = config['models']['ridge']
            ridge_model, ridge_info = train_ridge_model(
                X_train, y_train,
                alphas=ridge_config['alphas'],
                add_one=ridge_config['add_one_to_predictions']
            )
            
            ridge_preds = predict_with_adjustment(
                ridge_model, X_test,
                add_one=ridge_config['add_one_to_predictions']
            )
            
            ridge_metrics = evaluate_model(y_test, ridge_preds, 'Ridge Regression')
            all_results.append(ridge_metrics)
        except Exception as e:
            logger.error(f"Error training Ridge: {e}")
        
        # 4.2 Logistic Regression
        logger.info("-" * 60)
        logger.info("Training Logistic Regression")
        logger.info("-" * 60)
        try:
            lr_config = config['models']['logistic_regression']
            lr_model, lr_info = train_logistic_regression_model(
                X_train, y_train,
                C_values=lr_config['C_values'],
                add_one=lr_config['add_one_to_predictions']
            )
            
            lr_preds = predict_with_adjustment(
                lr_model, X_test,
                add_one=lr_config['add_one_to_predictions'],
                round_predictions=False
            )
            
            lr_metrics = evaluate_model(y_test, lr_preds, 'Logistic Regression')
            all_results.append(lr_metrics)
            
            # MLR_2: Probability threshold optimization
            logger.info("Optimizing probability threshold for Logistic Regression (MLR_2)")
            best_threshold, threshold_results = optimize_probability_threshold(
                lr_model, X_test, y_test,
                threshold_range=(0.3, 0.9),
                threshold_step=0.05
            )
            logger.info(f"Best threshold: {best_threshold}")
            
        except Exception as e:
            logger.error(f"Error training Logistic Regression: {e}")
        
        # 4.3 SVC Linear
        logger.info("-" * 60)
        logger.info("Training SVC (Linear)")
        logger.info("-" * 60)
        try:
            svc_config = config['models']['svc']
            svc_linear, svc_linear_info = train_svc_model(
                X_train, y_train,
                kernel='linear',
                C_values=svc_config['C_values'],
                probability=svc_config['probability']
            )
            
            svc_linear_preds = svc_linear.predict(X_test)
            svc_linear_metrics = evaluate_model(y_test, svc_linear_preds, 'SVC (Linear)')
            all_results.append(svc_linear_metrics)
            
        except Exception as e:
            logger.error(f"Error training SVC Linear: {e}")
        
        # 4.4 SVC Polynomial
        logger.info("-" * 60)
        logger.info("Training SVC (Polynomial)")
        logger.info("-" * 60)
        try:
            svc_poly, svc_poly_info = train_svc_model(
                X_train, y_train,
                kernel='poly',
                C_values=svc_config['C_values'],
                probability=svc_config['probability']
            )
            
            svc_poly_preds = svc_poly.predict(X_test)
            svc_poly_metrics = evaluate_model(y_test, svc_poly_preds, 'SVC (Polynomial)')
            all_results.append(svc_poly_metrics)
            
        except Exception as e:
            logger.error(f"Error training SVC Poly: {e}")
        
        # 4.5 ANN (Artificial Neural Network)
        logger.info("-" * 60)
        logger.info("Training Artificial Neural Network")
        logger.info("-" * 60)
        try:
            ann_config = config['models']['ann']
            
            # Split training data for validation
            from sklearn.model_selection import train_test_split
            X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
                X_train, y_train,
                test_size=ann_config['validation_split'],
                random_state=42
            )
            
            ann_model, ann_history = train_ann_model(
                X_train_nn, y_train_nn,
                X_val_nn, y_val_nn,
                hidden_layers=ann_config['hidden_layers'],
                dropout_rate=ann_config['dropout_rate'],
                learning_rate=ann_config['learning_rate'],
                batch_size=ann_config['batch_size'],
                epochs=ann_config['epochs'],
                early_stopping_patience=ann_config['early_stopping_patience']
            )
            
            # Plot learning curves
            plot_learning_curves(ann_history, plots_dir / 'ann_learning_curves.png')
            
            # Predictions
            ann_preds_proba = ann_model.predict(X_test)
            ann_preds = np.argmax(ann_preds_proba, axis=1)
            
            ann_metrics = evaluate_model(y_test, ann_preds, 'ANN')
            all_results.append(ann_metrics)
            
        except Exception as e:
            logger.error(f"Error training ANN: {e}")
        
        # ========================================================================
        # STEP 5: Benchmarking and Comparison
        # ========================================================================
        logger.info("=" * 60)
        logger.info("STEP 5: Benchmarking Against Competitor")
        logger.info("=" * 60)
        
        competitor_baseline = config['metrics']['competitor_baseline']
        competitor_baseline['model'] = 'Competitor Baseline'
        
        benchmark_df = create_benchmark_table(
            all_results,
            competitor_baseline,
            save_path=reports_dir / 'benchmark_comparison.csv'
        )
        
        logger.info("\nBenchmark Results:")
        logger.info("\n" + benchmark_df.to_string())
        
        # Plot comparison
        plot_model_comparison(benchmark_df, plots_dir / 'model_comparison.png')
        
        # ========================================================================
        # STEP 6: Water Savings and Cost Impact Calculation
        # ========================================================================
        logger.info("=" * 60)
        logger.info("STEP 6: Calculating Water Savings and Cost Impact")
        logger.info("=" * 60)
        
        hotel_config = config['hotel']
        water_config = config['water']
        
        # Use best model's predictions (choose one with highest accuracy)
        if 'ridge_preds' in locals():
            best_preds = ridge_preds
            best_model_name = 'Ridge Regression'
        else:
            best_preds = y_test  # Fallback
            best_model_name = 'Baseline'
        
        # Calculate savings
        savings = calculate_water_savings(y_test, best_preds)
        
        logger.info(f"\nWater Usage Analysis ({best_model_name}):")
        logger.info(f"  Actual water usage: {savings['actual_usage']:.2f}L")
        logger.info(f"  Predicted water usage: {savings['predicted_usage']:.2f}L")
        logger.info(f"  Water saved: {savings['savings']:.2f}L ({savings['savings_percentage']:.2f}%)")
        
        # Estimate annual impact for hotel
        avg_savings_per_flush = savings['savings'] / len(y_test)
        annual_impact = estimate_annual_impact(
            avg_savings_per_flush,
            num_rooms=hotel_config['num_rooms'],
            flushes_per_day=hotel_config['flushes_per_day'],
            days_per_year=hotel_config['days_per_year'],
            cost_per_1000L=water_config['cost_per_1000L']
        )
        
        logger.info(f"\nAnnual Hotel Impact ({hotel_config['num_rooms']} rooms):")
        logger.info(f"  Total flushes per year: {annual_impact['annual_flushes']:,}")
        logger.info(f"  Annual water savings: {annual_impact['annual_water_savings']:.2f}L")
        logger.info(f"  Annual cost savings: €{annual_impact['annual_cost_savings']:.2f}")
        
        # Save impact report
        impact_report = {
            'model': best_model_name,
            'test_set_savings': savings,
            'annual_impact': annual_impact,
            'hotel_parameters': hotel_config
        }
        
        import json
        with open(reports_dir / 'impact_report.json', 'w') as f:
            json.dump(impact_report, f, indent=2)
        
        # ========================================================================
        # STEP 7: Generate Final Reports
        # ========================================================================
        logger.info("=" * 60)
        logger.info("STEP 7: Generating Final Reports")
        logger.info("=" * 60)
        
        # Confusion matrix for best model
        if 'ridge_preds' in locals():
            plot_confusion_matrix(
                y_test, ridge_preds,
                save_path=plots_dir / 'confusion_matrix_ridge.png',
                title='Ridge Regression - Confusion Matrix'
            )
            
            generate_classification_report(
                y_test, ridge_preds,
                save_path=reports_dir / 'classification_report_ridge.txt'
            )
        
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline Complete!")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to:")
        logger.info(f"  - Plots: {plots_dir}")
        logger.info(f"  - Reports: {reports_dir}")
        logger.info(f"  - Benchmark: {reports_dir / 'benchmark_comparison.csv'}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
