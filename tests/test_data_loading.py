"""
Lightweight structural tests for the SmartFlush data_loading module.

These tests confirm that key interfaces are present so the implementation work
can grow against a stable contract.
"""

from pathlib import Path

import pandas as pd

from src import data_loading


def test_load_data_returns_dataframe(tmp_path: Path):
    sample_file = tmp_path / "sample.xlsx"
    pd.DataFrame({"a": [1], "b": [2]}).to_excel(sample_file, index=False)

    result = data_loading.load_data([sample_file])
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == 2


def test_preprocessing_interface_exists():
    assert hasattr(data_loading, "preprocess_data")
    try:
        data_loading.preprocess_data(pd.DataFrame(), None, {})
    except NotImplementedError:
        assert True
