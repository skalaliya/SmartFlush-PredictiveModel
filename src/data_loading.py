"""
Data ingestion and preprocessing utilities for the SmartFlush project.

This module exposes the public functions that the pipeline expects:
- load_data: collect and concatenate Excel sources.
- preprocess_data: orchestrate VIF checks, scaling, polynomial features, and residual features.
- handle_multicollinearity / apply_standard_scaler / build_polynomial_features: reusable helpers.

Each function is scaffolded with logging, type hints, and docstrings so the implementation can be
completed in subsequent development iterations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from . import utils

LOGGER = logging.getLogger(__name__)


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with normalised column names (lowercase, underscores)."""
    normalised = frame.copy()
    normalised.columns = [str(col).strip().lower().replace(" ", "_") for col in normalised.columns]
    return normalised


def _resolve_sheet_data(path: Path, sheet_selector: Optional[Union[str, int, Sequence[Union[str, int]], Dict[str, Sequence[Union[str, int]]]]]) -> List[pd.DataFrame]:
    """Read Excel file using flexible sheet selection."""
    try:
        if sheet_selector is None:
            content = pd.read_excel(path, sheet_name=None)
        elif isinstance(sheet_selector, (str, int)):
            content = pd.read_excel(path, sheet_name=sheet_selector)
        elif isinstance(sheet_selector, Sequence) and not isinstance(sheet_selector, (bytes, str)):
            content = {name: pd.read_excel(path, sheet_name=name) for name in sheet_selector}
        elif isinstance(sheet_selector, dict):
            names = sheet_selector.get(path.name, sheet_selector.get(str(path), None))
            if names is None:
                content = pd.read_excel(path, sheet_name=None)
            else:
                content = {name: pd.read_excel(path, sheet_name=name) for name in names}
        else:
            content = pd.read_excel(path, sheet_name=sheet_selector)
    except ValueError:
        LOGGER.warning("Sheet selection failed for %s. Falling back to default sheet.", path)
        content = pd.read_excel(path, sheet_name=0)

    frames: List[pd.DataFrame]
    if isinstance(content, dict):
        frames = [frame.assign(__source_sheet=name) for name, frame in content.items()]
    else:
        frames = [content.assign(__source_sheet=str(sheet_selector) if sheet_selector is not None else "sheet0")]
    return frames


@dataclass
class PreprocessingArtifacts:
    """Container for preprocessed outputs."""

    features: Optional[pd.DataFrame]
    target: Optional[pd.Series]
    transformers: Dict[str, Any]
    metadata: Dict[str, Any]
    splits: Dict[str, Any] = field(default_factory=dict)


def load_data(file_paths: Sequence[Path], sheet_name: Optional[Union[str, int, Sequence[Union[str, int]], Dict[str, Sequence[Union[str, int]]]]] = None) -> pd.DataFrame:
    """
    Load and concatenate Excel datasets.

    Args:
        file_paths: Ordered collection of Excel file paths to ingest.
        sheet_name: Optional sheet selector shared by all files or mapping per file.

    Returns:
        Combined pandas DataFrame. Empty DataFrame if no sources are located.
    """
    frames: List[pd.DataFrame] = []
    for path in file_paths:
        if not path.exists():
            LOGGER.warning("Skip missing data file: %s", path)
            continue

        LOGGER.info("Loading dataset from %s", path)
        for frame in _resolve_sheet_data(path, sheet_name):
            cleaned = _normalise_columns(frame)
            cleaned["__source_file"] = path.name
            frames.append(cleaned)

    if not frames:
        LOGGER.warning("No datasets loaded. Returning empty DataFrame.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    LOGGER.info("Combined dataset shape: rows=%s, cols=%s", *combined.shape)
    return combined


def preprocess_data(dataset: pd.DataFrame, target_column: Optional[str], config: Dict[str, Any]) -> PreprocessingArtifacts:
    """
    Apply SmartFlush preprocessing steps to the raw dataset.
    """
    if dataset.empty:
        raise ValueError("Dataset is empty. Ensure data files are available.")

    if target_column is None:
        raise ValueError("Target column must be provided for preprocessing.")

    target_column = target_column.lower()
    if target_column not in dataset.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")

    LOGGER.info("Preprocessing dataset with target column '%s'", target_column)
    data_cfg = config.get("data", {})
    prep_cfg = config.get("preprocessing", {})

    features = dataset.drop(columns=[target_column]).copy()
    target = dataset[target_column].copy()

    categorical_overrides = data_cfg.get("categorical_features", [])
    categorical_features = set(col.lower() for col in categorical_overrides if col in features.columns)
    inferred_categoricals = set(features.select_dtypes(include=["object", "category"]).columns)
    categorical_cols = sorted(categorical_features.union(inferred_categoricals))

    numeric_cols = sorted(col for col in features.columns if col not in categorical_cols and pd.api.types.is_numeric_dtype(features[col]))
    LOGGER.debug("Numeric columns: %s", numeric_cols)
    LOGGER.debug("Categorical columns: %s", categorical_cols)

    # Impute numeric data
    numeric_imputer = SimpleImputer(strategy=prep_cfg.get("imputation_strategy", "median"))
    numeric_df = pd.DataFrame(numeric_imputer.fit_transform(features[numeric_cols]), columns=numeric_cols, index=features.index)

    # VIF-based removal
    vif_threshold = prep_cfg.get("vif_threshold", 8.0)
    numeric_vif_df, removed_features = handle_multicollinearity(numeric_df, vif_threshold)

    residual_features = pd.DataFrame(index=features.index)
    if prep_cfg.get("include_residual_features", False) and numeric_vif_df.shape[1] >= 2:
        LOGGER.info("Generating residual-based features to mitigate multicollinearity.")
        residual_features = _build_residual_features(numeric_vif_df)
        residual_features.columns = [f"{col}_residual" for col in residual_features.columns]

    # Categorical preprocessing
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    if categorical_cols:
        categorical_imputed = categorical_imputer.fit_transform(features[categorical_cols])
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # fallback for older scikit-learn
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        categorical_encoded = encoder.fit_transform(categorical_imputed)
        categorical_feature_names = encoder.get_feature_names_out(categorical_cols)
        categorical_df = pd.DataFrame(categorical_encoded, columns=categorical_feature_names, index=features.index)
    else:
        categorical_df = pd.DataFrame(index=features.index)
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_combined = pd.concat([numeric_vif_df, residual_features], axis=1)

    scaled_numeric, scaler = apply_standard_scaler(numeric_combined, prep_cfg)

    degree = int(prep_cfg.get("polynomial_degree", 1))
    include_interactions = bool(prep_cfg.get("interaction_only", False))
    if degree > 1 and not scaled_numeric.empty:
        LOGGER.info("Applying polynomial features (degree=%s, interaction_only=%s).", degree, include_interactions)
        poly_numeric, poly_transformer = build_polynomial_features(scaled_numeric, degree, include_interactions)
    else:
        poly_numeric = scaled_numeric
        poly_transformer = None

    final_features = pd.concat([poly_numeric, categorical_df], axis=1)
    LOGGER.info("Feature matrix shape after preprocessing: %s", final_features.shape)

    test_size = data_cfg.get("test_size", 0.2)
    random_state = data_cfg.get("random_state", 42)
    stratify = target if data_cfg.get("stratify", False) else None

    X_train, X_test, y_train, y_test = train_test_split(final_features, target, test_size=test_size, random_state=random_state, stratify=stratify)

    val_split = prep_cfg.get("validation_split", 0.1)
    if val_split and val_split > 0.0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_split,
            random_state=random_state,
            stratify=y_train if stratify is not None else None,
        )
    else:
        X_val = pd.DataFrame()
        y_val = pd.Series(dtype=y_train.dtype if not y_train.empty else float)

    transformers = {
        "numeric_imputer": numeric_imputer,
        "categorical_imputer": categorical_imputer,
        "one_hot_encoder": encoder,
        "scaler": scaler,
        "polynomial": poly_transformer,
    }

    metadata = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "removed_vif_features": removed_features,
        "final_feature_names": list(final_features.columns),
    }

    splits = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

    return PreprocessingArtifacts(
        features=final_features,
        target=target,
        transformers=transformers,
        metadata=metadata,
        splits=splits,
    )


def _build_residual_features(features: pd.DataFrame) -> pd.DataFrame:
    """Create residual features by regressing each column on the others."""
    residuals = {}
    for column in features.columns:
        predictors = features.drop(columns=[column])
        if predictors.empty:
            continue
        model = LinearRegression()
        model.fit(predictors, features[column])
        predicted = model.predict(predictors)
        residuals[column] = features[column] - predicted
    return pd.DataFrame(residuals, index=features.index)


def handle_multicollinearity(features: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    """Iteratively remove features that exceed the specified VIF threshold."""
    working = features.copy()
    removed: List[str] = []

    while True:
        vif_df = utils.calculate_vif(working, threshold)
        high_vif = vif_df[vif_df["high_multicollinearity"]]
        if high_vif.empty:
            break
        drop_feature = high_vif.sort_values("vif", ascending=False).iloc[0]["feature"]
        LOGGER.info("Removing feature '%s' due to high VIF (%.2f).", drop_feature, high_vif.iloc[0]["vif"])
        working = working.drop(columns=[drop_feature])
        removed.append(drop_feature)
        if working.shape[1] <= 1:
            break

    return working, removed


def apply_standard_scaler(features: pd.DataFrame, scaler_config: Dict[str, Any]) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale features using StandardScaler (or compatible scaling strategy)."""
    if features.empty:
        return features.copy(), StandardScaler()

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled, columns=features.columns, index=features.index)
    LOGGER.debug("Applied standard scaling to numeric features.")
    return scaled_df, scaler


def build_polynomial_features(features: pd.DataFrame, degree: int, include_interactions: bool) -> Tuple[pd.DataFrame, PolynomialFeatures]:
    """Generate polynomial (and optional interaction) features."""
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=include_interactions)
    transformed = poly.fit_transform(features)
    feature_names = poly.get_feature_names_out(features.columns)
    transformed_df = pd.DataFrame(transformed, columns=feature_names, index=features.index)
    LOGGER.debug("Generated polynomial features with shape %s.", transformed_df.shape)
    return transformed_df, poly
