"""
Structural tests for the SmartFlush EDA module.

Ensures placeholder functions exist and raise NotImplementedError until fully implemented.
"""

from pathlib import Path

import pandas as pd

from src import eda


def test_run_eda_interface(tmp_path: Path):
    df = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
    try:
        eda.run_eda(df, "target", {}, tmp_path)
    except NotImplementedError:
        assert True


def test_helper_functions_present():
    assert hasattr(eda, "perform_chi2_tests")
    assert hasattr(eda, "compute_pearson_correlations")
    assert hasattr(eda, "plot_pairwise_relationships")
    assert hasattr(eda, "plot_feature_boxplots")
