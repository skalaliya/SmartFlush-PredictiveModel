"""
Unit tests for eda module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from src.eda import (
    calculate_pearson_correlation,
    generate_summary_statistics,
    set_plot_style
)


class TestCorrelationAnalysis:
    """Tests for correlation analysis."""
    
    def test_calculate_pearson_correlation(self):
        """Test Pearson correlation calculation."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'feature3': [1, 3, 5, 7, 9]
        })
        
        corr_matrix = calculate_pearson_correlation(df)
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (3, 3)
        assert np.allclose(np.diag(corr_matrix), 1.0)
    
    def test_calculate_pearson_correlation_subset(self):
        """Test correlation with feature subset."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'feature3': [1, 3, 5, 7, 9],
            'target': [0, 1, 0, 1, 0]
        })
        
        corr_matrix = calculate_pearson_correlation(df, ['feature1', 'feature2'])
        
        assert corr_matrix.shape == (2, 2)


class TestSummaryStatistics:
    """Tests for summary statistics generation."""
    
    def test_generate_summary_statistics(self):
        """Test summary statistics generation."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['x', 'y', 'x', 'y', 'x']
        })
        
        summary = generate_summary_statistics(df)
        
        assert isinstance(summary, pd.DataFrame)
        assert 'missing_count' in summary.columns
        assert 'missing_percentage' in summary.columns
        assert 'dtype' in summary.columns
        assert len(summary) == 3
    
    def test_generate_summary_statistics_with_missing(self):
        """Test summary statistics with missing values."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, np.nan, 30, np.nan, 50]
        })
        
        summary = generate_summary_statistics(df)
        
        assert summary.loc['A', 'missing_count'] == 1
        assert summary.loc['B', 'missing_count'] == 2
        assert summary.loc['B', 'missing_percentage'] == 40.0


class TestPlotting:
    """Tests for plotting functions."""
    
    def test_set_plot_style(self):
        """Test setting plot style."""
        # Should not raise any errors
        set_plot_style()
        set_plot_style(style='seaborn-v0_8-darkgrid', context='notebook')


class TestEDAIntegration:
    """Integration tests for EDA functions."""
    
    @pytest.mark.integration
    def test_perform_eda_basic(self):
        """Test basic EDA performance."""
        from src.eda import perform_eda
        
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.randint(1, 12, 100)
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = perform_eda(
                df,
                target_col='target',
                output_dir=Path(tmpdir)
            )
            
            assert 'summary_stats' in results
            assert 'correlation_matrix' in results
            assert isinstance(results['summary_stats'], pd.DataFrame)
            assert isinstance(results['correlation_matrix'], pd.DataFrame)
