"""
Shared utility functions for the SmartFlush predictive modeling project.

Exposes helpers for:
  * Configuration loading (`load_config`).
  * Logging setup (`configure_logging`).
  * Directory management (`ensure_directories`).
  * Statistical helpers (VIF, chi-square) and business utilities (water savings).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)

VOLUME_MAPPING: Dict[int, float] = {
    1: 1.5,
    2: 1.9,
    3: 2.2,
    4: 2.6,
    5: 3.0,
    6: 3.4,
    7: 3.8,
    8: 4.3,
    9: 4.7,
    10: 5.4,
    11: 6.1,
}


def load_config(path: Path) -> Dict[str, Any]:
    """
    Load the YAML configuration file.
    """
    LOGGER.debug("Loading configuration from %s", path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def configure_logging(logging_config: Dict[str, Any], logs_dir: Path) -> None:
    """
    Configure application-wide logging behaviour.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / logging_config.get("file_name", "smartflush.log")
    log_level = logging_config.get("level", "INFO").upper()
    log_format = logging_config.get("format", "%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    LOGGER.debug("Logging configured. Output file at %s", log_file)


def ensure_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Materialise required project directories and return their resolved paths.
    """
    paths_cfg = config.get("paths", {})
    outputs_cfg = config.get("outputs", {})

    directories = {
        "data_dir": Path(paths_cfg.get("data_dir", "data")),
        "notebooks_dir": Path(paths_cfg.get("notebooks_dir", "notebooks")),
        "reports_dir": Path(outputs_cfg.get("reports_dir", "reports")),
        "figures_dir": Path(outputs_cfg.get("figures_dir", "results/figures")),
        "tables_dir": Path(outputs_cfg.get("tables_dir", "results/tables")),
        "models_dir": Path(outputs_cfg.get("models_dir", "results/models")),
        "results_dir": Path(paths_cfg.get("results_dir", "results")),
        "logs_dir": Path(paths_cfg.get("logs_dir", "logs")),
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    return directories


def calculate_vif(features: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for the provided features.
    """
    LOGGER.debug("Calculating VIF for threshold=%s", threshold)
    raise NotImplementedError("Implement VIF calculation using statsmodels.")


def perform_chi2_test(feature: pd.Series, target: pd.Series) -> Dict[str, float]:
    """
    Conduct a chi-square independence test for a single feature.
    """
    LOGGER.debug("Performing chi-square test for feature=%s", feature.name)
    raise NotImplementedError("Implement chi-square test logic.")


def map_flush_class_to_volume(flush_class: int) -> float:
    """
    Map SmartFlush class identifiers to their corresponding volume in litres.
    """
    return VOLUME_MAPPING.get(flush_class, float(flush_class))


def compute_water_savings(predicted_classes: Iterable[int], baseline_volume: float, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute water savings benchmarks given predicted flush classes.
    """
    LOGGER.debug("Computing water savings for baseline volume %.2f", baseline_volume)
    raise NotImplementedError("Implement water savings computation leveraging volume mapping.")


@dataclass
class ImpactScenario:
    """Represents scenario assumptions for environmental/economic impact calculations."""

    rooms: int
    flushes_per_day: int
    water_cost_eur_per_1000l: float


def estimate_hotel_impact(predicted_volume_liters: np.ndarray, scenario: ImpactScenario) -> Dict[str, float]:
    """
    Estimate water and cost savings for a hospitality scenario.
    """
    LOGGER.debug("Estimating hotel impact for scenario=%s", scenario)
    raise NotImplementedError("Implement environmental/economic impact estimation.")
