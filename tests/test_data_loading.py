"""
Tests for data loading module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.data_loading import DataLoader, load_data_from_excel


@pytest.fixture
def sample_excel_file():
    """Create a sample Excel file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2.5, 3.5, 4.5, 5.5, 6.5],
            'feature3': [10, 20, 30, 40, 50],
            'flush_volume': [3.5, 4.0, 3.8, 4.2, 3.9]
        })
        df.to_excel(tmp.name, index=False, engine='openpyxl')
        yield tmp.name
    Path(tmp.name).unlink()


def test_data_loader_init():
    """Test DataLoader initialization"""
    loader = DataLoader("test.xlsx")
    assert loader.file_path == Path("test.xlsx")
    assert loader.data is None


def test_load_excel_success(sample_excel_file):
    """Test successful Excel loading"""
    loader = DataLoader(sample_excel_file)
    data = loader.load_excel()
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 5
    assert 'feature1' in data.columns
    assert 'flush_volume' in data.columns


def test_load_excel_file_not_found():
    """Test loading non-existent file"""
    loader = DataLoader("nonexistent.xlsx")
    
    with pytest.raises(FileNotFoundError):
        loader.load_excel()


def test_get_data_info(sample_excel_file):
    """Test getting data information"""
    loader = DataLoader(sample_excel_file)
    loader.load_excel()
    
    info = loader.get_data_info()
    
    assert 'shape' in info
    assert 'columns' in info
    assert 'missing_values' in info
    assert info['shape'] == (5, 4)


def test_clean_data(sample_excel_file):
    """Test data cleaning"""
    loader = DataLoader(sample_excel_file)
    loader.load_excel()
    
    cleaned = loader.clean_data(drop_na=True, drop_duplicates=True)
    
    assert isinstance(cleaned, pd.DataFrame)
    assert len(cleaned) <= 5


def test_split_features_target(sample_excel_file):
    """Test splitting features and target"""
    loader = DataLoader(sample_excel_file)
    loader.load_excel()
    
    X, y = loader.split_features_target('flush_volume')
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert 'flush_volume' not in X.columns
    assert len(X) == len(y)


def test_load_data_from_excel(sample_excel_file):
    """Test convenience function"""
    X, y = load_data_from_excel(sample_excel_file, 'flush_volume')
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
