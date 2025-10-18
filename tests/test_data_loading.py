"""
Unit tests for data_loading module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.data_loading import (
    combine_dataframes,
    handle_missing_values,
    create_residual_features,
    apply_vif_filtering,
    create_polynomial_features,
    standardize_features,
    prepare_data
)


class TestCombineDataframes:
    """Tests for combining dataframes."""
    
    def test_combine_dataframes_concat(self):
        """Test concatenating dataframes."""
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        
        result = combine_dataframes([df1, df2], how='concat')
        
        assert len(result) == 4
        assert list(result.columns) == ['A', 'B']
    
    def test_combine_dataframes_single(self):
        """Test combining single dataframe."""
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        
        result = combine_dataframes([df], how='concat')
        
        assert len(result) == 2
        pd.testing.assert_frame_equal(result, df)
    
    def test_combine_dataframes_empty_list(self):
        """Test combining empty list."""
        with pytest.raises(ValueError):
            combine_dataframes([], how='concat')


class TestHandleMissingValues:
    """Tests for missing value handling."""
    
    def test_handle_missing_values_mean(self):
        """Test filling missing values with mean."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0],
            'B': [5.0, np.nan, 7.0, 8.0]
        })
        
        result = handle_missing_values(df, strategy='mean')
        
        assert result.isnull().sum().sum() == 0
        assert result['A'].iloc[2] == pytest.approx((1 + 2 + 4) / 3)
    
    def test_handle_missing_values_drop(self):
        """Test dropping rows with missing values."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0],
            'B': [5.0, 6.0, 7.0, 8.0]
        })
        
        result = handle_missing_values(df, strategy='drop')
        
        assert len(result) < len(df)
        assert result.isnull().sum().sum() == 0


class TestFeatureEngineering:
    """Tests for feature engineering functions."""
    
    def test_create_polynomial_features(self):
        """Test polynomial feature creation."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        X_poly, feature_names = create_polynomial_features(X, degree=2)
        
        assert X_poly.shape[0] == 3
        assert X_poly.shape[1] > X.shape[1]
        assert len(feature_names) == X_poly.shape[1]
    
    def test_standardize_features(self):
        """Test feature standardization."""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_test = np.array([[7, 8], [9, 10]])
        
        X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)
        
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        assert abs(X_train_scaled.mean()) < 1e-10
        assert abs(X_train_scaled.std() - 1.0) < 1e-10
    
    def test_create_residual_features(self):
        """Test residual feature creation."""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        y = pd.Series([1, 2, 3, 4, 5])
        
        X_enhanced = create_residual_features(X, y, ['feature1'])
        
        assert 'feature1_residual' in X_enhanced.columns
        assert X_enhanced.shape[0] == X.shape[0]
        assert X_enhanced.shape[1] == X.shape[1] + 1


class TestDataPreparation:
    """Tests for complete data preparation pipeline."""
    
    def test_prepare_data_basic(self):
        """Test basic data preparation."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.randint(1, 12, 100)
        })
        
        result = prepare_data(
            df,
            target_col='target',
            test_size=0.2,
            random_state=42,
            vif_threshold=None
        )
        
        assert 'X_train' in result
        assert 'X_test' in result
        assert 'y_train' in result
        assert 'y_test' in result
        assert len(result['y_train']) == 80
        assert len(result['y_test']) == 20
    
    def test_prepare_data_with_vif(self):
        """Test data preparation with VIF filtering."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.randint(1, 12, 100)
        })
        
        result = prepare_data(
            df,
            target_col='target',
            test_size=0.2,
            vif_threshold=10.0
        )
        
        assert result['X_train'].shape[1] <= 3
