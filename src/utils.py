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
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
    """Load the YAML configuration file."""
    LOGGER.debug("Loading configuration from %s", path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def configure_logging(logging_config: Dict[str, Any], logs_dir: Path) -> None:
    """Configure application-wide logging behaviour."""
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
    """Materialise required project directories and return their resolved paths."""
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


def _ensure_numeric_dataframe(features: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric-only DataFrame with safe column names."""
    numeric_df = features.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.fillna(0.0)
    return numeric_df


def calculate_vif(features: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for the provided features.

    Returns a DataFrame with columns [feature, vif, high_multicollinearity].
    """
    if features.empty:
        return pd.DataFrame(columns=["feature", "vif", "high_multicollinearity"])

    processed = _ensure_numeric_dataframe(features)
    if processed.shape[1] < 2:
        # Need at least two features to compute VIF; return zeros.
        return pd.DataFrame(
            {
                "feature": processed.columns,
                "vif": np.zeros(processed.shape[1]),
                "high_multicollinearity": [False] * processed.shape[1],
            }
        )

    vif_values = []
    for idx, column in enumerate(processed.columns):
        try:
            value = variance_inflation_factor(processed.values, idx)
        except Exception:  # statsmodels may raise if singular
            value = np.inf
        vif_values.append(value)

    vif_df = pd.DataFrame(
        {
            "feature": processed.columns,
            "vif": vif_values,
        }
    )
    vif_df["high_multicollinearity"] = vif_df["vif"] > threshold
    LOGGER.debug("Calculated VIF values:\n%s", vif_df.sort_values("vif", ascending=False).head())
    return vif_df


def perform_chi2_test(feature: pd.Series, target: pd.Series) -> Dict[str, float]:
    """Conduct a chi-square independence test for a single feature."""
    contingency = pd.crosstab(feature, target)
    chi2, p_value, dof, _ = chi2_contingency(contingency)
    result = {
        "feature": feature.name,
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
    }
    LOGGER.debug("Chi-square result for %s: %s", feature.name, result)
    return result


def map_flush_class_to_volume(flush_class: int) -> float:
    """Map SmartFlush class identifiers to their corresponding volume in litres."""
    return VOLUME_MAPPING.get(int(flush_class), float(flush_class))


def compute_water_savings(predicted_classes: Iterable[int], baseline_volume: float, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute water savings benchmarks given predicted flush classes.

    Args:
        predicted_classes: Iterable of predicted classes.
        baseline_volume: Liters per flush of the legacy system.
        config: Project configuration providing volume mapping overrides.
    """
    mapping = config.get("evaluation", {}).get("volume_mapping", VOLUME_MAPPING)
    volumes = np.array([float(mapping.get(int(cls), mapping.get(int(cls), baseline_volume))) for cls in predicted_classes])

    total_predicted = volumes.sum()
    total_baseline = baseline_volume * len(volumes)
    savings_liters = total_baseline - total_predicted
    savings_pct = (savings_liters / total_baseline) * 100 if total_baseline else 0.0

    LOGGER.debug(
        "Water savings computed: predicted=%s baseline=%s savings=%sL (%s%%)",
        total_predicted,
        total_baseline,
        savings_liters,
        savings_pct,
    )

    return {
        "total_predicted_liters": total_predicted,
        "baseline_total_liters": total_baseline,
        "savings_liters": savings_liters,
        "savings_percent": savings_pct,
    }


@dataclass
class ImpactScenario:
    """Represents scenario assumptions for environmental/economic impact calculations."""

    rooms: int
    flushes_per_day: int
    water_cost_eur_per_1000l: float
    occupancy_rate: float = 1.0  # 0-1 multiplier


def estimate_hotel_impact(predicted_volume_liters: np.ndarray, scenario: ImpactScenario) -> Dict[str, float]:
    """
    Estimate water and cost savings for a hospitality scenario.

    Args:
        predicted_volume_liters: Numpy array containing per-flush predicted volumes.
        scenario: ImpactScenario configuration.
    """
    mean_volume = float(np.mean(predicted_volume_liters)) if predicted_volume_liters.size else 0.0
    total_flushes_day = scenario.rooms * scenario.flushes_per_day * scenario.occupancy_rate
    total_volume_day = mean_volume * total_flushes_day
    total_volume_year = total_volume_day * 365

    cost_per_liter = scenario.water_cost_eur_per_1000l / 1000.0
    cost_day = total_volume_day * cost_per_liter
    cost_year = total_volume_year * cost_per_liter

    LOGGER.debug(
        "Hotel impact: volume_day=%.2fL volume_year=%.2fL cost_day=%.2f€ cost_year=%.2f€",
        total_volume_day,
        total_volume_year,
        cost_day,
        cost_year,
    )

    return {
        "mean_flush_volume_l": mean_volume,
        "flushes_per_day": total_flushes_day,
        "daily_volume_l": total_volume_day,
        "annual_volume_l": total_volume_year,
        "daily_cost_eur": cost_day,
        "annual_cost_eur": cost_year,
    }
