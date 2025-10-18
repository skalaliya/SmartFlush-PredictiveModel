"""
Tests for EDA module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.eda import EDAAnalyzer, perform_eda


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'feature4': np.random.randint(0, 3, 100)
    })
    y = pd.Series(X['feature1'] * 2 + X['feature2'] * 1.5 + np.random.randn(100) * 0.5, name='target')
    return X, y


@pytest.fixture
def config():
    """Create sample configuration"""
    return {
        'eda': {
            'chi2_alpha': 0.05,
            'pearson_threshold': 0.7,
            'plot_figsize': [12, 8]
        }
    }


def test_eda_analyzer_init(sample_data, config):
    """Test EDAAnalyzer initialization"""
    X, y = sample_data
    analyzer = EDAAnalyzer(X, y, config)
    
    assert analyzer.X is X
    assert analyzer.y is y
    assert analyzer.config == config


def test_pearson_correlation(sample_data, config):
    """Test Pearson correlation calculation"""
    X, y = sample_data
    analyzer = EDAAnalyzer(X, y, config)
    
    corr_matrix, high_corr = analyzer.pearson_correlation(threshold=0.7)
    
    assert isinstance(corr_matrix, pd.DataFrame)
    assert isinstance(high_corr, list)
    assert corr_matrix.shape == (len(X.columns), len(X.columns))


def test_target_correlation(sample_data, config):
    """Test target correlation calculation"""
    X, y = sample_data
    analyzer = EDAAnalyzer(X, y, config)
    
    correlations = analyzer.target_correlation()
    
    assert isinstance(correlations, pd.Series)
    assert len(correlations) == len(X.columns)


def test_generate_summary_report(sample_data, config):
    """Test summary report generation"""
    X, y = sample_data
    analyzer = EDAAnalyzer(X, y, config)
    
    report = analyzer.generate_summary_report()
    
    assert isinstance(report, dict)
    assert 'n_samples' in report
    assert 'n_features' in report
    assert 'feature_names' in report
    assert report['n_samples'] == 100
    assert report['n_features'] == 4


def test_plot_correlation_heatmap(sample_data, config):
    """Test correlation heatmap generation"""
    X, y = sample_data
    analyzer = EDAAnalyzer(X, y, config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "heatmap.png"
        analyzer.plot_correlation_heatmap(save_path=str(save_path))
        
        # Check that file was created
        assert save_path.exists()


def test_plot_target_correlation(sample_data, config):
    """Test target correlation plot generation"""
    X, y = sample_data
    analyzer = EDAAnalyzer(X, y, config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "target_corr.png"
        analyzer.plot_target_correlation(save_path=str(save_path))
        
        # Check that file was created
        assert save_path.exists()


def test_plot_distributions(sample_data, config):
    """Test distribution plots generation"""
    X, y = sample_data
    analyzer = EDAAnalyzer(X, y, config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "distributions.png"
        analyzer.plot_distributions(save_path=str(save_path))
        
        # Check that file was created
        assert save_path.exists()


def test_perform_eda(sample_data, config):
    """Test perform_eda convenience function"""
    X, y = sample_data
    
    with tempfile.TemporaryDirectory() as tmpdir:
        report = perform_eda(X, y, config, output_dir=tmpdir)
        
        assert isinstance(report, dict)
        assert 'n_samples' in report
        
        # Check that plots were created
        assert (Path(tmpdir) / "correlation_heatmap.png").exists()
        assert (Path(tmpdir) / "target_correlation.png").exists()
        assert (Path(tmpdir) / "distributions.png").exists()
