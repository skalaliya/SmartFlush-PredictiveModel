"""
Data loading and preprocessing module for SmartFlush Predictive Model.

This module handles:
- Loading data from Excel files (Combined_Data.xlsx, mon_fichier.xlsx)
- Data combination and validation
- VIF-based multicollinearity handling
- Feature engineering (polynomial features, residuals)
- Data standardization using StandardScaler
- Train-test splitting
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from src.utils import calculate_vif, remove_high_vif_features

# Configure logging
logger = logging.getLogger(__name__)


def load_excel_file(
    file_path: Union[str, Path],
    sheet_name: Optional[Union[str, int]] = 0
) -> pd.DataFrame:
    """
    Load data from an Excel file.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Sheet name or index to load (default: 0 for first sheet)
        
    Returns:
        DataFrame containing the loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be read
        
    Example:
        >>> df = load_excel_file("data/Combined_Data.xlsx")
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        logger.debug(f"Columns: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading Excel file {file_path}: {e}")
        raise


def combine_dataframes(
    dataframes: List[pd.DataFrame],
    how: str = 'outer',
    on: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Combine multiple DataFrames into one.
    
    Args:
        dataframes: List of DataFrames to combine
        how: How to combine ('outer', 'inner', 'concat'). Default: 'outer'
        on: Columns to merge on if using merge (default: None)
        
    Returns:
        Combined DataFrame
        
    Raises:
        ValueError: If dataframes list is empty or incompatible
        
    Example:
        >>> combined = combine_dataframes([df1, df2], how='outer')
    """
    try:
        if not dataframes:
            raise ValueError("No dataframes provided to combine")
        
        if len(dataframes) == 1:
            return dataframes[0].copy()
        
        logger.info(f"Combining {len(dataframes)} dataframes using method: {how}")
        
        if how == 'concat':
            # Concatenate vertically (stack rows)
            combined = pd.concat(dataframes, axis=0, ignore_index=True)
        else:
            # Merge dataframes
            combined = dataframes[0]
            for df in dataframes[1:]:
                if on:
                    combined = pd.merge(combined, df, on=on, how=how)
                else:
                    combined = pd.merge(combined, df, how=how, left_index=True, right_index=True)
        
        logger.info(f"Combined DataFrame shape: {combined.shape}")
        return combined
        
    except Exception as e:
        logger.error(f"Error combining dataframes: {e}")
        raise


def load_and_combine_data(
    file_paths: List[Union[str, Path]],
    how: str = 'concat'
) -> pd.DataFrame:
    """
    Load multiple Excel files and combine them into a single DataFrame.
    
    Args:
        file_paths: List of paths to Excel files
        how: How to combine the data ('concat', 'outer', 'inner')
        
    Returns:
        Combined DataFrame
        
    Example:
        >>> data = load_and_combine_data(
        ...     ["data/Combined_Data.xlsx", "data/mon_fichier.xlsx"],
        ...     how='concat'
        ... )
    """
    try:
        logger.info(f"Loading and combining {len(file_paths)} Excel files")
        
        dataframes = []
        for file_path in file_paths:
            try:
                df = load_excel_file(file_path)
                dataframes.append(df)
            except FileNotFoundError:
                logger.warning(f"File not found, skipping: {file_path}")
                continue
        
        if not dataframes:
            raise ValueError("No data files could be loaded")
        
        combined = combine_dataframes(dataframes, how=how)
        return combined
        
    except Exception as e:
        logger.error(f"Error loading and combining data: {e}")
        raise


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mean',
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values ('mean', 'median', 'drop', 'forward_fill')
        threshold: If strategy='drop', drop columns with more than this fraction of missing values
        
    Returns:
        DataFrame with missing values handled
        
    Example:
        >>> df_clean = handle_missing_values(df, strategy='mean')
    """
    try:
        df = df.copy()
        missing_before = df.isnull().sum().sum()
        
        logger.info(f"Handling missing values. Total missing: {missing_before}")
        
        if strategy == 'drop':
            # Drop columns with too many missing values
            missing_fraction = df.isnull().sum() / len(df)
            cols_to_drop = missing_fraction[missing_fraction > threshold].index.tolist()
            if cols_to_drop:
                logger.info(f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
                df = df.drop(columns=cols_to_drop)
            
            # Drop rows with any remaining missing values
            df = df.dropna()
            
        elif strategy == 'mean':
            # Fill numeric columns with mean
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            
        elif strategy == 'median':
            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
        elif strategy == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values after handling: {missing_after}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error handling missing values: {e}")
        raise


def create_residual_features(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Create residual features by fitting simple linear models and extracting residuals.
    
    Residuals can capture non-linear patterns not explained by linear relationships.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        feature_cols: Columns to create residual features for
        
    Returns:
        DataFrame with original features plus residual features
        
    Example:
        >>> X_with_residuals = create_residual_features(X, y, ['feature1', 'feature2'])
    """
    try:
        X_enhanced = X.copy()
        
        logger.info(f"Creating residual features for {len(feature_cols)} columns")
        
        for col in feature_cols:
            if col not in X.columns:
                logger.warning(f"Column {col} not found in X, skipping")
                continue
            
            # Fit linear model
            lr = LinearRegression()
            lr.fit(X[[col]], y)
            
            # Calculate residuals
            predictions = lr.predict(X[[col]])
            residuals = y - predictions
            
            # Add as new feature
            residual_col_name = f"{col}_residual"
            X_enhanced[residual_col_name] = residuals
            
        logger.info(f"Enhanced DataFrame shape: {X_enhanced.shape}")
        return X_enhanced
        
    except Exception as e:
        logger.error(f"Error creating residual features: {e}")
        raise


def apply_vif_filtering(
    df: pd.DataFrame,
    feature_cols: List[str],
    vif_threshold: float = 10.0
) -> List[str]:
    """
    Apply VIF-based feature selection to handle multicollinearity.
    
    Args:
        df: DataFrame containing features
        feature_cols: List of feature column names to consider
        vif_threshold: VIF threshold for feature removal (default: 10.0)
        
    Returns:
        List of selected feature names after VIF filtering
        
    Example:
        >>> selected_features = apply_vif_filtering(df, all_features, vif_threshold=10.0)
    """
    try:
        logger.info(f"Applying VIF filtering with threshold={vif_threshold}")
        logger.info(f"Initial feature count: {len(feature_cols)}")
        
        # Remove features with high VIF
        selected_features = remove_high_vif_features(
            df,
            feature_cols,
            threshold=vif_threshold
        )
        
        removed_count = len(feature_cols) - len(selected_features)
        logger.info(f"Removed {removed_count} features due to high VIF")
        logger.info(f"Final feature count: {len(selected_features)}")
        
        return selected_features
        
    except Exception as e:
        logger.error(f"Error applying VIF filtering: {e}")
        raise


def create_polynomial_features(
    X: pd.DataFrame,
    degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = False
) -> Tuple[np.ndarray, List[str]]:
    """
    Create polynomial features for the input data.
    
    Args:
        X: Input feature DataFrame
        degree: Degree of polynomial features (default: 2)
        interaction_only: If True, only interaction features are created
        include_bias: If True, include a bias column
        
    Returns:
        Tuple of:
        - Transformed feature array
        - List of feature names
        
    Example:
        >>> X_poly, feature_names = create_polynomial_features(X, degree=2)
    """
    try:
        logger.info(f"Creating polynomial features with degree={degree}")
        
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(X.columns)
        
        logger.info(f"Polynomial features shape: {X_poly.shape}")
        logger.info(f"Original features: {X.shape[1]}, Polynomial features: {X_poly.shape[1]}")
        
        return X_poly, feature_names.tolist()
        
    except Exception as e:
        logger.error(f"Error creating polynomial features: {e}")
        raise


def standardize_features(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """
    Standardize features using StandardScaler (zero mean, unit variance).
    
    Args:
        X_train: Training feature array
        X_test: Test feature array (optional)
        
    Returns:
        Tuple of:
        - Standardized X_train
        - Standardized X_test (if provided)
        - Fitted StandardScaler object
        
    Example:
        >>> X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)
    """
    try:
        logger.info("Standardizing features using StandardScaler")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        logger.info(f"Training data scaled: mean={X_train_scaled.mean():.6f}, "
                   f"std={X_train_scaled.std():.6f}")
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            logger.info(f"Test data scaled: mean={X_test_scaled.mean():.6f}, "
                       f"std={X_test_scaled.std():.6f}")
        
        return X_train_scaled, X_test_scaled, scaler
        
    except Exception as e:
        logger.error(f"Error standardizing features: {e}")
        raise


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    vif_threshold: float = 10.0,
    apply_polynomial: bool = False,
    polynomial_degree: int = 2,
    apply_standardization: bool = True
) -> Dict:
    """
    Complete data preparation pipeline: feature selection, splitting, scaling.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        feature_cols: List of feature columns (if None, uses all except target)
        test_size: Fraction of data for test set (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        vif_threshold: VIF threshold for feature filtering (default: 10.0)
        apply_polynomial: Whether to create polynomial features
        polynomial_degree: Degree for polynomial features
        apply_standardization: Whether to standardize features
        
    Returns:
        Dictionary containing:
        - X_train, X_test: Feature arrays
        - y_train, y_test: Target arrays
        - feature_names: List of feature names
        - scaler: Fitted StandardScaler (if applied)
        
    Example:
        >>> data_dict = prepare_data(df, target_col='flush_level', vif_threshold=10.0)
        >>> X_train, y_train = data_dict['X_train'], data_dict['y_train']
    """
    try:
        logger.info("Starting data preparation pipeline")
        
        # Validate target column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Select features
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
            # Only use numeric columns
            feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Initial features: {len(feature_cols)}")
        
        # Apply VIF filtering if threshold is set
        if vif_threshold is not None and vif_threshold > 0:
            feature_cols = apply_vif_filtering(df, feature_cols, vif_threshold)
        
        # Extract features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values in features
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        feature_names = feature_cols
        
        # Apply polynomial features if requested
        if apply_polynomial:
            X_train_poly, feature_names = create_polynomial_features(
                X_train, degree=polynomial_degree
            )
            X_test_poly, _ = create_polynomial_features(
                X_test, degree=polynomial_degree
            )
            X_train = X_train_poly
            X_test = X_test_poly
        
        # Standardize features
        scaler = None
        if apply_standardization:
            X_train, X_test, scaler = standardize_features(X_train, X_test)
        
        result = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train.values if hasattr(y_train, 'values') else y_train,
            'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
            'feature_names': feature_names,
            'scaler': scaler
        }
        
        logger.info("Data preparation complete")
        return result
        
    except Exception as e:
        logger.error(f"Error in data preparation pipeline: {e}")
        raise
