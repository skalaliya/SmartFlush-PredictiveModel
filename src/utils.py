"""
Utility functions for SmartFlush Predictive Model.

This module provides helper functions for:
- VIF (Variance Inflation Factor) calculation for multicollinearity detection
- Chi-squared test helpers for categorical feature analysis
- Water savings calculations based on flush volume mappings
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configure logging
logger = logging.getLogger(__name__)


def calculate_vif(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for features to detect multicollinearity.
    
    VIF measures how much the variance of a regression coefficient is inflated due to
    multicollinearity. VIF > 10 typically indicates high multicollinearity.
    
    Args:
        df: DataFrame containing features
        features: List of feature column names. If None, uses all numeric columns.
        
    Returns:
        DataFrame with columns ['Feature', 'VIF'] sorted by VIF in descending order
        
    Raises:
        ValueError: If df is empty or features don't exist in df
        
    Example:
        >>> vif_results = calculate_vif(data, ['feature1', 'feature2', 'feature3'])
        >>> high_vif = vif_results[vif_results['VIF'] > 10]
    """
    try:
        if df.empty:
            raise ValueError("Input DataFrame is empty")
            
        # Select features
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Validate features exist
        missing_features = set(features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")
        
        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["Feature"] = features
        
        # Add small epsilon to avoid division issues
        X = df[features].copy()
        X = X.fillna(X.mean())  # Handle missing values
        
        vif_values = []
        for i, feature in enumerate(features):
            try:
                vif = variance_inflation_factor(X.values, i)
                vif_values.append(vif)
            except Exception as e:
                logger.warning(f"Could not calculate VIF for {feature}: {e}")
                vif_values.append(np.nan)
        
        vif_data["VIF"] = vif_values
        vif_data = vif_data.sort_values('VIF', ascending=False)
        
        logger.info(f"Calculated VIF for {len(features)} features")
        return vif_data
        
    except Exception as e:
        logger.error(f"Error calculating VIF: {e}")
        raise


def remove_high_vif_features(
    df: pd.DataFrame,
    features: List[str],
    threshold: float = 10.0,
    max_iterations: int = 10
) -> List[str]:
    """
    Iteratively remove features with high VIF until all remaining features have VIF below threshold.
    
    Args:
        df: DataFrame containing features
        features: List of feature column names to consider
        threshold: VIF threshold above which features are removed (default: 10.0)
        max_iterations: Maximum number of iterations to prevent infinite loops
        
    Returns:
        List of feature names with VIF below threshold
        
    Example:
        >>> selected_features = remove_high_vif_features(data, all_features, threshold=10.0)
    """
    try:
        remaining_features = features.copy()
        
        for iteration in range(max_iterations):
            if len(remaining_features) <= 1:
                logger.warning("Only one or fewer features remaining after VIF filtering")
                break
                
            vif_df = calculate_vif(df, remaining_features)
            max_vif = vif_df['VIF'].max()
            
            if max_vif < threshold:
                logger.info(f"All features have VIF < {threshold} after {iteration} iterations")
                break
                
            # Remove feature with highest VIF
            feature_to_remove = vif_df.iloc[0]['Feature']
            remaining_features.remove(feature_to_remove)
            logger.info(f"Removed {feature_to_remove} with VIF={max_vif:.2f}")
        
        logger.info(f"Final feature count: {len(remaining_features)}/{len(features)}")
        return remaining_features
        
    except Exception as e:
        logger.error(f"Error removing high VIF features: {e}")
        raise


def chi_squared_test(
    df: pd.DataFrame,
    feature: str,
    target: str
) -> Tuple[float, float, pd.DataFrame]:
    """
    Perform Chi-squared test of independence between categorical feature and target.
    
    Args:
        df: DataFrame containing the data
        feature: Name of the categorical feature column
        target: Name of the target column
        
    Returns:
        Tuple containing:
        - chi2_stat: Chi-squared test statistic
        - p_value: P-value of the test
        - contingency_table: Contingency table (crosstab)
        
    Raises:
        ValueError: If feature or target not in df, or if insufficient data
        
    Example:
        >>> chi2, p_val, table = chi_squared_test(data, 'category', 'flush_level')
        >>> if p_val < 0.05:
        >>>     print(f"Feature {feature} is significantly associated with target")
    """
    try:
        # Validate inputs
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame")
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found in DataFrame")
        
        # Create contingency table
        contingency_table = pd.crosstab(df[feature], df[target])
        
        if contingency_table.size == 0:
            raise ValueError("Contingency table is empty")
        
        # Perform chi-squared test
        chi2_stat, p_value, dof, expected_freq = chi2_contingency(contingency_table)
        
        logger.info(f"Chi-squared test: {feature} vs {target}")
        logger.info(f"  Chi2={chi2_stat:.4f}, p-value={p_value:.4e}, dof={dof}")
        
        return chi2_stat, p_value, contingency_table
        
    except Exception as e:
        logger.error(f"Error performing chi-squared test: {e}")
        raise


def get_flush_volume_map() -> Dict[int, float]:
    """
    Get the mapping from flush level to water volume in liters.
    
    Returns:
        Dictionary mapping flush level (1-11) to volume in liters
        
    Example:
        >>> volume_map = get_flush_volume_map()
        >>> print(f"Flush level 5 uses {volume_map[5]} liters")
    """
    return {
        1: 1.5,
        2: 1.9,
        3: 2.3,
        4: 2.7,
        5: 3.1,
        6: 3.5,
        7: 3.9,
        8: 4.3,
        9: 4.7,
        10: 5.3,
        11: 6.1
    }


def calculate_water_usage(flush_levels: np.ndarray) -> float:
    """
    Calculate total water usage in liters based on flush levels.
    
    Args:
        flush_levels: Array of flush level predictions (1-11)
        
    Returns:
        Total water usage in liters
        
    Raises:
        ValueError: If flush levels are out of valid range
        
    Example:
        >>> predictions = np.array([3, 5, 2, 7, 4])
        >>> total_water = calculate_water_usage(predictions)
    """
    try:
        volume_map = get_flush_volume_map()
        
        # Validate flush levels
        if np.any((flush_levels < 1) | (flush_levels > 11)):
            raise ValueError("Flush levels must be between 1 and 11")
        
        # Calculate total usage
        total_usage = sum(volume_map[int(level)] for level in flush_levels)
        
        logger.debug(f"Calculated water usage: {total_usage:.2f}L for {len(flush_levels)} flushes")
        return total_usage
        
    except Exception as e:
        logger.error(f"Error calculating water usage: {e}")
        raise


def calculate_water_savings(
    actual_levels: np.ndarray,
    predicted_levels: np.ndarray
) -> Dict[str, float]:
    """
    Calculate water savings by comparing actual vs predicted flush levels.
    
    Args:
        actual_levels: Array of actual flush levels
        predicted_levels: Array of predicted flush levels
        
    Returns:
        Dictionary containing:
        - actual_usage: Total water used with actual levels (liters)
        - predicted_usage: Total water used with predicted levels (liters)
        - savings: Water saved (liters)
        - savings_percentage: Percentage of water saved
        
    Example:
        >>> savings = calculate_water_savings(y_test, predictions)
        >>> print(f"Water saved: {savings['savings']:.2f}L ({savings['savings_percentage']:.1f}%)")
    """
    try:
        if len(actual_levels) != len(predicted_levels):
            raise ValueError("Actual and predicted arrays must have same length")
        
        actual_usage = calculate_water_usage(actual_levels)
        predicted_usage = calculate_water_usage(predicted_levels)
        savings = actual_usage - predicted_usage
        savings_percentage = (savings / actual_usage * 100) if actual_usage > 0 else 0.0
        
        result = {
            'actual_usage': actual_usage,
            'predicted_usage': predicted_usage,
            'savings': savings,
            'savings_percentage': savings_percentage
        }
        
        logger.info(f"Water savings: {savings:.2f}L ({savings_percentage:.1f}%)")
        return result
        
    except Exception as e:
        logger.error(f"Error calculating water savings: {e}")
        raise


def calculate_cost_savings(
    water_savings_liters: float,
    cost_per_1000L: float = 4.0
) -> float:
    """
    Calculate cost savings based on water savings and cost per 1000 liters.
    
    Args:
        water_savings_liters: Amount of water saved in liters
        cost_per_1000L: Cost per 1000 liters in currency units (default: 4.0 euros)
        
    Returns:
        Cost savings in currency units
        
    Example:
        >>> cost_saved = calculate_cost_savings(10000, cost_per_1000L=4.0)
        >>> print(f"Cost saved: €{cost_saved:.2f}")
    """
    try:
        cost_savings = (water_savings_liters / 1000.0) * cost_per_1000L
        logger.info(f"Cost savings: {cost_savings:.2f} currency units")
        return cost_savings
        
    except Exception as e:
        logger.error(f"Error calculating cost savings: {e}")
        raise


def estimate_annual_impact(
    water_savings_per_flush: float,
    num_rooms: int = 100,
    flushes_per_day: int = 5,
    days_per_year: int = 365,
    cost_per_1000L: float = 4.0
) -> Dict[str, float]:
    """
    Estimate annual water and cost savings for a hotel scenario.
    
    Args:
        water_savings_per_flush: Average water saved per flush (liters)
        num_rooms: Number of hotel rooms
        flushes_per_day: Flushes per room per day
        days_per_year: Days in operational year
        cost_per_1000L: Cost per 1000 liters
        
    Returns:
        Dictionary containing:
        - annual_flushes: Total flushes per year
        - annual_water_savings: Total water saved per year (liters)
        - annual_cost_savings: Total cost saved per year
        
    Example:
        >>> impact = estimate_annual_impact(0.5, num_rooms=100, flushes_per_day=5)
        >>> print(f"Annual savings: {impact['annual_water_savings']}L, €{impact['annual_cost_savings']}")
    """
    try:
        annual_flushes = num_rooms * flushes_per_day * days_per_year
        annual_water_savings = water_savings_per_flush * annual_flushes
        annual_cost_savings = calculate_cost_savings(annual_water_savings, cost_per_1000L)
        
        result = {
            'annual_flushes': annual_flushes,
            'annual_water_savings': annual_water_savings,
            'annual_cost_savings': annual_cost_savings
        }
        
        logger.info(f"Annual impact: {annual_flushes} flushes, "
                   f"{annual_water_savings:.2f}L saved, "
                   f"{annual_cost_savings:.2f} currency units saved")
        
        return result
        
    except Exception as e:
        logger.error(f"Error estimating annual impact: {e}")
        raise
