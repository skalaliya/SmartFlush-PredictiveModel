"""
Exploratory data analysis utilities for the SmartFlush predictive modeling project.

This module exposes scaffolded functions that will eventually drive:
  * Chi-square association testing for categorical features.
  * Pearson correlation studies for photodiode readings and other continuous sensors.
  * Pairplot visualisations to understand multivariate relationships.
  * Boxplots relating feature distributions to flush case targets.

The implementations are intentionally left as placeholders so the modelling team can
layer in business-specific logic while retaining consistent interfaces.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import utils

LOGGER = logging.getLogger(__name__)


def run_eda(dataset: pd.DataFrame, target_column: Optional[str], config: Dict[str, Any], output_directory: Path) -> Dict[str, Any]:
    """
    Execute the end-to-end EDA workflow.

    Args:
        dataset: Combined feature/target dataset.
        target_column: Target variable used to highlight relationships.
        config: Project configuration dictionary.
        output_directory: Directory for saving plots and tables.

    Returns:
        Dictionary mapping report names to generated artefacts.
    """
    LOGGER.info("Starting EDA run for dataset shape=%s", getattr(dataset, "shape", "unknown"))
    output_directory.mkdir(parents=True, exist_ok=True)

    eda_cfg = config.get("eda", {})
    alpha = eda_cfg.get("chi2_alpha", 0.05)
    correlation_method = eda_cfg.get("pearson_method", "pearson")
    pairplot_sample = eda_cfg.get("pairplot_sample_size", 2000)
    boxplot_targets = eda_cfg.get("boxplot_targets", [])

    results: Dict[str, Any] = {}

    if target_column and target_column in dataset.columns:
        categorical_features = [
            col for col in dataset.select_dtypes(include=["object", "category"]).columns if col != target_column
        ]
        # Include configured categorical overrides
        categorical_features.extend(
            [col for col in config.get("data", {}).get("categorical_features", []) if col in dataset.columns]
        )
        categorical_features = sorted(set(categorical_features))

        if categorical_features:
            chi_df = perform_chi2_tests(dataset, categorical_features, target_column, alpha)
            chi_path = output_directory / "chi_square_tests.csv"
            chi_df.to_csv(chi_path, index=False)
            LOGGER.info("Saved chi-square results to %s", chi_path)
            results["chi_square"] = chi_df
    else:
        categorical_features = []

    # Pearson correlations
    numeric_df = dataset.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr_df = compute_pearson_correlations(numeric_df, method=correlation_method)
        corr_path = output_directory / "correlation_matrix.csv"
        corr_df.to_csv(corr_path)
        results["correlation"] = corr_df

        heatmap_path = output_directory / "correlation_heatmap.png"
        plt.figure(figsize=tuple(eda_cfg.get("heatmap_figsize", [12, 8])))
        sns.heatmap(corr_df, cmap="coolwarm", center=0, square=True)
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        LOGGER.info("Saved correlation heatmap to %s", heatmap_path)

    # Pairplot
    if numeric_df.shape[1] >= 2:
        pairplot_path = output_directory / "pairplot.png"
        plot_pairwise_relationships(dataset, target_column, pairplot_path, pairplot_sample)

    # Boxplots
    if target_column and target_column in dataset.columns:
        box_output = output_directory / "boxplots"
        box_output.mkdir(parents=True, exist_ok=True)

        if not boxplot_targets:
            features = numeric_df.columns.tolist()
        else:
            features = [col for col in boxplot_targets if col in dataset.columns]

        plot_feature_boxplots(dataset, features, target_column, box_output)

    return results


def perform_chi2_tests(dataset: pd.DataFrame, categorical_features: Iterable[str], target_column: str, alpha: float) -> pd.DataFrame:
    """
    Perform chi-square tests to evaluate associations between categorical features and the target.

    Args:
        dataset: Data containing categorical features and the target.
        categorical_features: Iterable of column names to inspect.
        target_column: Target column name.
        alpha: Significance threshold for hypothesis testing.

    Returns:
        DataFrame summarising chi-square statistics and p-values.
    """
    LOGGER.debug("Running chi-square tests for features=%s with alpha=%s", list(categorical_features), alpha)
    records = []
    for feature in categorical_features:
        if feature not in dataset.columns:
            LOGGER.warning("Skipping chi-square test for missing feature '%s'", feature)
            continue
        try:
            result = utils.perform_chi2_test(dataset[feature], dataset[target_column])
            result["significant"] = result["p_value"] < alpha
            records.append(result)
        except ValueError as exc:
            LOGGER.warning("Chi-square test failed for feature '%s': %s", feature, exc)
    return pd.DataFrame(records)


def compute_pearson_correlations(dataset: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Compute Pearson (or alternative) correlation matrix for continuous features.

    Args:
        dataset: Numerical features DataFrame.
        method: Correlation method accepted by pandas (default 'pearson').

    Returns:
        Correlation matrix DataFrame.
    """
    LOGGER.debug("Computing %s correlation matrix for dataset.", method)
    return dataset.corr(method=method)


def plot_pairwise_relationships(dataset: pd.DataFrame, hue: Optional[str], output_path: Path, sample_size: Optional[int] = None) -> None:
    """
    Generate pairplots to visualise feature relationships.

    Args:
        dataset: Data to plot.
        hue: Optional categorical column to colour the plots.
        output_path: File path for the saved figure.
        sample_size: Optional row cap to speed up rendering.
    """
    LOGGER.debug("Preparing pairplot at %s (hue=%s, sample_size=%s)", output_path, hue, sample_size)
    if sample_size is not None and len(dataset) > sample_size:
        dataset = dataset.sample(sample_size, random_state=42)

    plot_data = dataset.copy()
    numeric_cols = plot_data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        LOGGER.info("Pairplot skipped: fewer than two numeric columns available.")
        return

    sns.pairplot(plot_data, vars=numeric_cols, hue=hue, corner=True, diag_kind="kde")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    LOGGER.info("Saved pairplot to %s", output_path)


def plot_feature_boxplots(dataset: pd.DataFrame, features: Iterable[str], target_column: str, output_directory: Path) -> None:
    """
    Create boxplots showing feature distributions across target classes.

    Args:
        dataset: Data containing the features and target.
        features: Features to visualise.
        target_column: Target column name.
        output_directory: Directory to save generated plots.
    """
    LOGGER.debug("Generating boxplots for features=%s to %s", list(features), output_directory)
    if target_column not in dataset.columns:
        LOGGER.warning("Boxplot skipped: target column '%s' missing.", target_column)
        return

    for feature in features:
        if feature not in dataset.columns or pd.api.types.is_numeric_dtype(dataset[feature]) is False:
            continue
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=dataset[target_column], y=dataset[feature])
        plt.title(f"{feature} vs {target_column}")
        plt.xlabel(target_column)
        plt.ylabel(feature)
        plt.tight_layout()
        plot_path = output_directory / f"{feature}_vs_{target_column}.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        LOGGER.debug("Saved boxplot for feature %s to %s", feature, plot_path)
