"""
Utility Functions Module
Includes VIF calculation, residual analysis, scaling, and polynomial features
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import yaml

logger = logging.getLogger(__name__)


def calculate_vif(X: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for features
    
    Args:
        X: Feature DataFrame
        threshold: VIF threshold for multicollinearity detection
        
    Returns:
        DataFrame with VIF values and flags for high multicollinearity
    """
    logger.info("Calculating VIF for features")
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data["High_Multicollinearity"] = vif_data["VIF"] > threshold
    
    logger.info(f"Features with VIF > {threshold}: {vif_data['High_Multicollinearity'].sum()}")
    
    return vif_data


def remove_high_vif_features(X: pd.DataFrame, 
                             threshold: float = 10.0,
                             max_iterations: int = 10) -> Tuple[pd.DataFrame, List[str]]:
    """
    Iteratively remove features with high VIF
    
    Args:
        X: Feature DataFrame
        threshold: VIF threshold
        max_iterations: Maximum iterations for removal
        
    Returns:
        Tuple of (cleaned DataFrame, list of removed features)
    """
    X_clean = X.copy()
    removed_features = []
    
    for iteration in range(max_iterations):
        vif_df = calculate_vif(X_clean, threshold)
        
        # Check if any features exceed threshold
        high_vif = vif_df[vif_df["High_Multicollinearity"]]
        
        if len(high_vif) == 0:
            logger.info(f"VIF cleanup complete after {iteration} iterations")
            break
            
        # Remove feature with highest VIF
        max_vif_feature = vif_df.loc[vif_df["VIF"].idxmax(), "Feature"]
        removed_features.append(max_vif_feature)
        X_clean = X_clean.drop(columns=[max_vif_feature])
        
        logger.info(f"Iteration {iteration + 1}: Removed '{max_vif_feature}' with VIF={vif_df['VIF'].max():.2f}")
    
    return X_clean, removed_features


def calculate_residuals(y_true: np.ndarray, 
                       y_pred: np.ndarray) -> dict:
    """
    Calculate residual statistics
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary with residual statistics
    """
    residuals = y_true - y_pred
    
    stats = {
        "mean": np.mean(residuals),
        "std": np.std(residuals),
        "min": np.min(residuals),
        "max": np.max(residuals),
        "median": np.median(residuals),
        "q25": np.percentile(residuals, 25),
        "q75": np.percentile(residuals, 75),
        "residuals": residuals
    }
    
    logger.info(f"Residual stats - Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    
    return stats


def get_scaler(method: str = "standard"):
    """
    Get sklearn scaler instance
    
    Args:
        method: Scaling method ('standard', 'minmax', 'robust')
        
    Returns:
        Scaler instance
    """
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler()
    }
    
    if method not in scalers:
        logger.warning(f"Unknown scaling method '{method}', using 'standard'")
        method = "standard"
        
    logger.info(f"Using {method} scaler")
    return scalers[method]


def scale_features(X: pd.DataFrame, 
                  method: str = "standard",
                  scaler=None) -> Tuple[pd.DataFrame, object]:
    """
    Scale features using specified method
    
    Args:
        X: Feature DataFrame
        method: Scaling method
        scaler: Pre-fitted scaler (if None, creates new one)
        
    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    if scaler is None:
        scaler = get_scaler(method)
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    logger.info(f"Features scaled using {method} method")
    
    return X_scaled_df, scaler


def create_polynomial_features(X: pd.DataFrame, 
                               degree: int = 2,
                               include_bias: bool = False,
                               interaction_only: bool = False) -> Tuple[pd.DataFrame, PolynomialFeatures]:
    """
    Create polynomial features
    
    Args:
        X: Feature DataFrame
        degree: Polynomial degree
        include_bias: Whether to include bias term
        interaction_only: Only interaction features
        
    Returns:
        Tuple of (polynomial features DataFrame, fitted transformer)
    """
    logger.info(f"Creating polynomial features with degree={degree}")
    
    poly = PolynomialFeatures(degree=degree, 
                              include_bias=include_bias,
                              interaction_only=interaction_only)
    
    X_poly = poly.fit_transform(X)
    
    # Create column names
    feature_names = poly.get_feature_names_out(X.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    
    logger.info(f"Created {X_poly_df.shape[1]} polynomial features from {X.shape[1]} original features")
    
    return X_poly_df, poly


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise


def setup_logging(config: dict):
    """
    Setup logging configuration
    
    Args:
        config: Configuration dictionary
    """
    import os
    from pathlib import Path
    
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file', 'logs/smartflush.log')
    
    # Create logs directory
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Logging configured")


def create_output_directories(config: dict):
    """
    Create output directories from config
    
    Args:
        config: Configuration dictionary
    """
    from pathlib import Path
    
    output_config = config.get('output', {})
    
    directories = [
        output_config.get('model_path', 'results/models/'),
        output_config.get('plot_path', 'reports/figures/'),
        output_config.get('report_path', 'reports/'),
        'data',
        'results',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    logger.info("Output directories created")
