"""
SmartFlush predictive modeling package.

Provides modular building blocks for the commercial SmartFlush system:
  * Data ingestion & preprocessing (`src.data_loading`)
  * Exploratory data analysis routines (`src.eda`)
  * Model orchestration (`src.models`)
  * Custom metrics and reporting (`src.metrics`)
  * Shared helpers (`src.utils`)
"""

__version__ = "0.1.0"

from . import data_loading, eda, metrics, models, utils  # noqa: E402

__all__ = [
    "data_loading",
    "eda",
    "metrics",
    "models",
    "utils",
]
