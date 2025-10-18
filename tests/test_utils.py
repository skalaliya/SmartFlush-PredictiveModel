"""
Structural tests for SmartFlush utility functions.
"""

from pathlib import Path

import pandas as pd

from src import utils


def test_load_config_roundtrip(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("sample: value\n", encoding="utf-8")
    loaded = utils.load_config(config_path)
    assert loaded["sample"] == "value"


def test_configure_logging_creates_file(tmp_path: Path):
    utils.configure_logging({}, tmp_path)
    assert (tmp_path / "smartflush.log").exists()


def test_volume_mapping_contains_expected_keys():
    for key in range(1, 12):
        assert key in utils.VOLUME_MAPPING


def test_calculate_vif_returns_dataframe():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0], "c": [1.0, 1.5, 2.0, 2.5]})
    result = utils.calculate_vif(frame, threshold=5.0)
    assert list(result.columns) == ["feature", "vif", "high_multicollinearity"]


def test_perform_chi2_test_returns_stats():
    data = pd.Series(["x", "x", "y", "y"], name="feature")
    target = pd.Series([0, 0, 1, 1], name="target")
    result = utils.perform_chi2_test(data, target)
    assert set(result.keys()) == {"feature", "chi2", "p_value", "dof"}
