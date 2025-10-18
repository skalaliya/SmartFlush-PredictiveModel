"""
Tests for utility functions
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils import (
    calculate_vif, remove_high_vif_features, calculate_residuals,
    get_scaler, scale_features, create_polynomial_features
)


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    return X


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing"""
    np.random.seed(42)
    y_true = np.random.randn(100) * 2 + 5
    y_pred = y_true + np.random.randn(100) * 0.5
    return y_true, y_pred


def test_calculate_vif(sample_data):
    """Test VIF calculation"""
    vif_df = calculate_vif(sample_data, threshold=10.0)
    
    assert isinstance(vif_df, pd.DataFrame)
    assert 'Feature' in vif_df.columns
    assert 'VIF' in vif_df.columns
    assert 'High_Multicollinearity' in vif_df.columns
    assert len(vif_df) == 3


def test_remove_high_vif_features(sample_data):
    """Test removing high VIF features"""
    # Add a highly correlated feature
    sample_data['feature4'] = sample_data['feature1'] * 2 + 0.1
    
    X_clean, removed = remove_high_vif_features(sample_data, threshold=5.0)
    
    assert isinstance(X_clean, pd.DataFrame)
    assert isinstance(removed, list)
    assert len(X_clean.columns) <= len(sample_data.columns)


def test_calculate_residuals(sample_predictions):
    """Test residual calculation"""
    y_true, y_pred = sample_predictions
    
    stats = calculate_residuals(y_true, y_pred)
    
    assert 'mean' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats
    assert 'residuals' in stats
    assert len(stats['residuals']) == len(y_true)


def test_get_scaler():
    """Test scaler retrieval"""
    scaler = get_scaler('standard')
    assert isinstance(scaler, StandardScaler)
    
    scaler = get_scaler('minmax')
    assert scaler is not None
    
    scaler = get_scaler('robust')
    assert scaler is not None


def test_scale_features(sample_data):
    """Test feature scaling"""
    X_scaled, scaler = scale_features(sample_data, method='standard')
    
    assert isinstance(X_scaled, pd.DataFrame)
    assert X_scaled.shape == sample_data.shape
    assert list(X_scaled.columns) == list(sample_data.columns)
    
    # Check that features are scaled (mean ~0, std ~1)
    assert abs(X_scaled['feature1'].mean()) < 0.1
    assert abs(X_scaled['feature1'].std() - 1.0) < 0.1


def test_create_polynomial_features(sample_data):
    """Test polynomial feature creation"""
    X_poly, poly_transformer = create_polynomial_features(sample_data, degree=2)
    
    assert isinstance(X_poly, pd.DataFrame)
    assert X_poly.shape[1] > sample_data.shape[1]
    assert X_poly.shape[0] == sample_data.shape[0]
