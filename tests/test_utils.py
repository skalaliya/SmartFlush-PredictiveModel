"""
Unit tests for utils module.
"""

import pytest
import numpy as np
import pandas as pd
from src.utils import (
    calculate_vif,
    remove_high_vif_features,
    chi_squared_test,
    get_flush_volume_map,
    calculate_water_usage,
    calculate_water_savings,
    calculate_cost_savings,
    estimate_annual_impact
)


class TestVIFCalculation:
    """Tests for VIF calculation functions."""
    
    def test_calculate_vif_basic(self):
        """Test basic VIF calculation."""
        # Create simple dataframe
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        
        vif_df = calculate_vif(df)
        
        assert isinstance(vif_df, pd.DataFrame)
        assert 'Feature' in vif_df.columns
        assert 'VIF' in vif_df.columns
        assert len(vif_df) == 3
    
    def test_calculate_vif_empty_df(self):
        """Test VIF calculation with empty dataframe."""
        df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            calculate_vif(df)
    
    def test_remove_high_vif_features(self):
        """Test removing features with high VIF."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        
        selected = remove_high_vif_features(df, df.columns.tolist(), threshold=10.0)
        
        assert isinstance(selected, list)
        assert len(selected) <= 3


class TestChiSquaredTest:
    """Tests for Chi-squared test functions."""
    
    def test_chi_squared_test_basic(self):
        """Test basic Chi-squared test."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A', 'B'] * 10,
            'target': [1, 2, 1, 2, 1, 2] * 10
        })
        
        chi2, p_value, table = chi_squared_test(df, 'category', 'target')
        
        assert isinstance(chi2, float)
        assert isinstance(p_value, float)
        assert isinstance(table, pd.DataFrame)
        assert chi2 >= 0
        assert 0 <= p_value <= 1
    
    def test_chi_squared_test_invalid_feature(self):
        """Test Chi-squared test with invalid feature."""
        df = pd.DataFrame({'feature': [1, 2, 3], 'target': [1, 2, 3]})
        
        with pytest.raises(ValueError):
            chi_squared_test(df, 'nonexistent', 'target')


class TestWaterCalculations:
    """Tests for water usage and savings calculations."""
    
    def test_get_flush_volume_map(self):
        """Test getting flush volume map."""
        volume_map = get_flush_volume_map()
        
        assert isinstance(volume_map, dict)
        assert len(volume_map) == 11
        assert all(1 <= k <= 11 for k in volume_map.keys())
        assert all(v > 0 for v in volume_map.values())
    
    def test_calculate_water_usage(self):
        """Test water usage calculation."""
        flush_levels = np.array([1, 2, 3, 4, 5])
        
        total_usage = calculate_water_usage(flush_levels)
        
        assert isinstance(total_usage, float)
        assert total_usage > 0
    
    def test_calculate_water_usage_invalid_levels(self):
        """Test water usage with invalid flush levels."""
        flush_levels = np.array([0, 12, 15])
        
        with pytest.raises(ValueError):
            calculate_water_usage(flush_levels)
    
    def test_calculate_water_savings(self):
        """Test water savings calculation."""
        actual = np.array([5, 6, 7, 8, 9])
        predicted = np.array([4, 5, 6, 7, 8])
        
        savings = calculate_water_savings(actual, predicted)
        
        assert isinstance(savings, dict)
        assert 'actual_usage' in savings
        assert 'predicted_usage' in savings
        assert 'savings' in savings
        assert 'savings_percentage' in savings
    
    def test_calculate_cost_savings(self):
        """Test cost savings calculation."""
        water_savings = 1000.0  # liters
        cost = calculate_cost_savings(water_savings, cost_per_1000L=4.0)
        
        assert isinstance(cost, float)
        assert cost == 4.0
    
    def test_estimate_annual_impact(self):
        """Test annual impact estimation."""
        impact = estimate_annual_impact(
            water_savings_per_flush=0.5,
            num_rooms=100,
            flushes_per_day=5,
            days_per_year=365,
            cost_per_1000L=4.0
        )
        
        assert isinstance(impact, dict)
        assert 'annual_flushes' in impact
        assert 'annual_water_savings' in impact
        assert 'annual_cost_savings' in impact
        assert impact['annual_flushes'] == 100 * 5 * 365


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_calculate_water_savings_mismatched_lengths(self):
        """Test water savings with mismatched array lengths."""
        actual = np.array([1, 2, 3])
        predicted = np.array([1, 2])
        
        with pytest.raises(ValueError):
            calculate_water_savings(actual, predicted)
