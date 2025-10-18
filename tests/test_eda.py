"""
Behavioural tests for the SmartFlush EDA module.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src import eda


def test_run_eda_generates_outputs(tmp_path: Path):
    rng = np.random.default_rng(0)
    dataset = pd.DataFrame(
        {
            "sensor_a": rng.normal(size=150),
            "sensor_b": rng.normal(size=150),
            "case_of_flush": rng.choice(["light", "medium", "heavy"], size=150),
            "flush_volume_class": rng.integers(1, 4, size=150),
        }
    )

    config = {
        "data": {"categorical_features": ["case_of_flush"]},
        "eda": {
            "chi2_alpha": 0.05,
            "pearson_method": "pearson",
            "pairplot_sample_size": 100,
            "boxplot_targets": ["sensor_a"],
        },
    }

    results = eda.run_eda(dataset, "flush_volume_class", config, tmp_path)

    assert "correlation" in results
    assert (tmp_path / "correlation_matrix.csv").exists()
    assert (tmp_path / "correlation_heatmap.png").exists()
    assert (tmp_path / "pairplot.png").exists()
    assert (tmp_path / "chi_square_tests.csv").exists()
    assert (tmp_path / "boxplots/sensor_a_vs_flush_volume_class.png").exists()


def test_perform_chi2_tests_flags_significance():
    data = pd.DataFrame(
        {
            "cat_feature": ["a", "a", "b", "b", "c", "c"],
            "target": [0, 0, 1, 1, 1, 0],
        }
    )
    results = eda.perform_chi2_tests(data, ["cat_feature"], "target", alpha=0.1)
    assert not results.empty
    assert "significant" in results.columns
