"""
Behavioural tests for the SmartFlush data_loading module.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src import data_loading


def test_load_data_combines_multiple_sheets(tmp_path: Path):
    sample_file = tmp_path / "sample.xlsx"
    with pd.ExcelWriter(sample_file) as writer:
        pd.DataFrame({"feature": [1, 2], "target": [0, 1]}).to_excel(writer, sheet_name="sheet_a", index=False)
        pd.DataFrame({"feature": [3, 4], "target": [1, 0]}).to_excel(writer, sheet_name="sheet_b", index=False)

    combined = data_loading.load_data([sample_file])
    assert combined.shape[0] == 4
    assert "__source_sheet" in combined.columns
    assert "__source_file" in combined.columns


def test_preprocess_data_returns_artifacts():
    rng = np.random.default_rng(42)
    dataset = pd.DataFrame(
        {
            "sensor_a": rng.normal(0, 1, size=120),
            "sensor_b": rng.normal(0, 1, size=120),
            "sensor_c": rng.normal(0, 1, size=120),
            "case_of_flush": rng.choice(["light", "medium", "heavy"], size=120),
            "flush_volume_class": rng.integers(1, 4, size=120),
        }
    )
    dataset["sensor_c"] = dataset["sensor_a"] * 0.8 + rng.normal(0, 0.2, size=120)  # induce collinearity

    config = {
        "data": {
            "categorical_features": ["case_of_flush"],
            "test_size": 0.25,
            "random_state": 7,
            "stratify": True,
        },
        "preprocessing": {
            "imputation_strategy": "median",
            "vif_threshold": 5.0,
            "include_residual_features": True,
            "polynomial_degree": 2,
            "validation_split": 0.2,
        },
    }

    artifacts = data_loading.preprocess_data(dataset, "flush_volume_class", config)

    assert artifacts.features is not None
    assert artifacts.target is not None
    assert set(artifacts.splits.keys()) >= {"X_train", "X_test", "y_train", "y_test"}
    assert artifacts.features.shape[1] >= len(dataset.columns)
    assert "removed_vif_features" in artifacts.metadata
