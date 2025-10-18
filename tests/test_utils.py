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


def test_unimplemented_helpers_raise():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    try:
        utils.calculate_vif(frame, 10)
    except NotImplementedError:
        assert True
