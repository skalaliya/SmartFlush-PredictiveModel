"""
Model orchestration for the SmartFlush predictive modeling project.

This module provides scaffolding for the required model families:
  * Ridge Regression with +1 prediction adjustment.
  * Multinomial Logistic Regression with deterministic and probability-based variants (MLR, MLR_2).
  * Support Vector Classifiers with linear and polynomial kernels (SVC, SVC_2).
  * Artificial Neural Network (ANN) leveraging TensorFlow/Keras, ReLU activations, dropout, and learning curves.

The code intentionally exposes template methods that raise `NotImplementedError`. This guides future
development while guaranteeing a consistent interface that upstream orchestration (e.g. `main.py`)
and downstream tests can depend on from day one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tensorflow import keras
from tensorflow.keras import callbacks, layers, optimizers

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelArtifacts:
    """Container for trained models, evaluation splits, and metadata."""

    models: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_data: Dict[str, Any] = field(default_factory=dict)


class ModelingEngine:
    """
    High-level coordinator for SmartFlush model training pipelines.

    The engine consumes preprocessed datasets and configuration dictionaries to train and persist
    model variants. Concrete algorithmic logic should be implemented within the dedicated helper
    methods that currently raise `NotImplementedError`.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_cfg = config.get("models", {})
        self.grid_cfg = config.get("grid_search", {})
        LOGGER.debug("Initialised ModelingEngine with models=%s", list(self.models_cfg.keys()))

    def train_all(self, data_bundle: Dict[str, Any]) -> ModelArtifacts:
        """
        Train all enabled models using the provided data bundle.

        Args:
            data_bundle: Dictionary containing training, validation, and testing splits along with
                any fitted preprocessing artefacts required for inference.

        Returns:
            ModelArtifacts populated with trained estimators and metadata.
        """
        LOGGER.info("Training all enabled models.")
        if hasattr(data_bundle, "splits"):
            splits = data_bundle.splits
            metadata = getattr(data_bundle, "metadata", {})
        else:
            splits = data_bundle.get("splits", {})
            metadata = data_bundle.get("metadata", {})
        if not splits:
            raise ValueError("Data bundle must contain training/testing splits.")

        X_train = splits.get("X_train")
        y_train = splits.get("y_train")
        X_val = splits.get("X_val")
        y_val = splits.get("y_val")
        X_test = splits.get("X_test")
        y_test = splits.get("y_test")

        artifacts = ModelArtifacts(
            metadata={
                "feature_names": metadata.get("final_feature_names", []),
                "removed_vif_features": metadata.get("removed_vif_features", []),
            },
            evaluation_data={"X_test": X_test, "y_test": y_test, "predictions": {}},
        )

        if self.models_cfg.get("ridge", {}).get("enabled", True):
            artifacts.models["ridge"] = self.train_ridge(X_train, y_train)

        if self.models_cfg.get("multinomial_logistic", {}).get("enabled", True):
            artifacts.models["mlr"] = self.train_multinomial_logistic(X_train, y_train)

        mlr_prob_cfg = self.models_cfg.get("multinomial_logistic_prob", {})
        if mlr_prob_cfg.get("enabled", True):
            threshold_values = mlr_prob_cfg.get("thresholds", [0.5])
            prob_models = self.train_multinomial_logistic_prob(X_train, y_train, threshold_values)
            artifacts.models.update(prob_models)

        if self.models_cfg.get("svc", {}).get("enabled", True):
            artifacts.models["svc"] = self.train_svc(X_train, y_train)

        svc_prob_cfg = self.models_cfg.get("svc_prob", {})
        if svc_prob_cfg.get("enabled", True):
            svc_thresholds = svc_prob_cfg.get("thresholds", [0.5])
            svc_prob_models = self.train_svc_prob(X_train, y_train, svc_thresholds)
            artifacts.models.update(svc_prob_models)

        if self.models_cfg.get("ann", {}).get("enabled", True):
            artifacts.models["ann"] = self.train_ann(
                X_train,
                y_train,
                validation_data={"X_val": X_val, "y_val": y_val} if X_val is not None and not X_val.empty else None,
            )

        # Generate predictions for evaluation
        for name, model in artifacts.models.items():
            try:
                preds = self.predict(name, model, X_test)
                artifacts.evaluation_data["predictions"][name] = preds
            except Exception as exc:  # capture to continue evaluation for other models
                LOGGER.warning("Prediction failed for %s: %s", name, exc)

        return artifacts

    def train_ridge(self, features: pd.DataFrame, target: pd.Series) -> Any:
        """Train Ridge Regression pipeline with GridSearchCV and +1 prediction increment."""
        LOGGER.debug("Training Ridge Regression model.")
        model_cfg = self.models_cfg.get("ridge", {})
        alphas = model_cfg.get("grid", {}).get("alpha", [0.1, 1.0, 10.0])

        pipeline = Pipeline(steps=[("regressor", Ridge())])

        param_grid = {"regressor__alpha": alphas}

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=self.grid_cfg.get("cv_folds", 5),
            scoring=self.grid_cfg.get("scoring", "neg_mean_absolute_error"),
            n_jobs=self.grid_cfg.get("n_jobs", -1),
        )
        grid.fit(features, target)
        LOGGER.info("Best Ridge parameters: %s", grid.best_params_)
        return grid.best_estimator_

    def train_multinomial_logistic(self, features: pd.DataFrame, target: pd.Series) -> Any:
        """Train multinomial logistic regression with +1 prediction increment."""
        LOGGER.debug("Training Multinomial Logistic Regression model.")
        cfg = self.models_cfg.get("multinomial_logistic", {})
        param_grid = cfg.get(
            "grid",
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__max_iter": [200],
            },
        )

        pipeline = Pipeline(
            steps=[
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=500,
                    ),
                )
            ]
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=self.grid_cfg.get("cv_folds", 5),
            scoring="accuracy",
            n_jobs=self.grid_cfg.get("n_jobs", -1),
        )
        grid.fit(features, target)
        LOGGER.info("Best Multinomial Logistic params: %s", grid.best_params_)
        return grid.best_estimator_

    def train_multinomial_logistic_prob(self, features: pd.DataFrame, target: pd.Series, thresholds: Iterable[float]) -> Dict[str, Any]:
        """
        Train probability-based MLR variant with adjustable thresholds.

        Args:
            features: Training features.
            target: Training target.
            thresholds: Iterable of probability thresholds to evaluate.

        Returns:
            Mapping of threshold identifiers to fitted models or calibration metadata.
        """
        LOGGER.debug("Training probability-driven MLR variant with thresholds=%s", list(thresholds))
        base_model = self.train_multinomial_logistic(features, target)
        return {
            f"mlr_prob_{threshold:.2f}": {"model": base_model, "threshold": threshold, "type": "mlr_prob"}
            for threshold in thresholds
        }

    def train_svc(self, features: pd.DataFrame, target: pd.Series) -> Any:
        """Train Support Vector Classifier with linear and polynomial kernels."""
        LOGGER.debug("Training Support Vector Classifier.")
        cfg = self.models_cfg.get("svc", {})
        kernels = cfg.get("kernels", ["linear", "poly"])
        param_grid = cfg.get(
            "grid",
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__degree": [2, 3],
            },
        )

        pipeline = Pipeline(
            steps=[
                (
                    "classifier",
                    SVC(kernel="rbf", probability=True, gamma="scale"),
                )
            ]
        )

        # ensure kernel is part of grid if provided separately
        if "classifier__kernel" not in param_grid:
            param_grid["classifier__kernel"] = kernels

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=self.grid_cfg.get("cv_folds", 5),
            scoring="accuracy",
            n_jobs=self.grid_cfg.get("n_jobs", -1),
        )
        grid.fit(features, target)
        LOGGER.info("Best SVC parameters: %s", grid.best_params_)
        return grid.best_estimator_

    def train_svc_prob(self, features: pd.DataFrame, target: pd.Series, thresholds: Iterable[float]) -> Dict[str, Any]:
        """Train probability-calibrated SVC variant."""
        LOGGER.debug("Training probability-calibrated SVC variant.")
        base_model = self.train_svc(features, target)
        return {
            f"svc_prob_{threshold:.2f}": {"model": base_model, "threshold": threshold, "type": "svc_prob"}
            for threshold in thresholds
        }

    def train_ann(self, features: pd.DataFrame, target: pd.Series, validation_data: Optional[Dict[str, Any]] = None) -> Any:
        """
        Train Keras-based ANN with ReLU activations, dropout, and sigmoid probability head.
        Should record learning curves for overfitting diagnostics.
        """
        LOGGER.debug("Training Artificial Neural Network model.")
        cfg = self.models_cfg.get("ann", {})
        architecture = cfg.get("architecture", [128, 64, 32])
        dropout_rate = cfg.get("dropout_rate", 0.2)
        learning_rate = cfg.get("learning_rate", 0.001)
        epochs = cfg.get("epochs", 100)
        batch_size = cfg.get("batch_size", 32)

        num_classes = len(np.unique(target))

        model = keras.Sequential()
        model.add(layers.Input(shape=(features.shape[1],)))

        for units in architecture:
            model.add(layers.Dense(units, activation="relu"))
            model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(num_classes, activation="softmax"))

        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        callbacks_list = [
            callbacks.EarlyStopping(monitor="val_loss", patience=cfg.get("patience", 15), restore_best_weights=True)
        ]

        x_train = np.asarray(features, dtype=np.float32)
        y_train = np.asarray(target, dtype=np.int32)

        validation = None
        if validation_data:
            X_val = validation_data.get("X_val")
            y_val = validation_data.get("y_val")
            if X_val is not None and not X_val.empty:
                validation = (np.asarray(X_val, dtype=np.float32), np.asarray(y_val, dtype=np.int32))

        history = model.fit(
            x_train,
            y_train,
            validation_data=validation,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks_list,
        )

        return {"model": model, "history": history.history}

    def predict(self, model_name: str, model: Any, features: pd.DataFrame) -> Any:
        """
        Generate predictions from a trained model, applying the required +1 increment logic when necessary.
        """
        LOGGER.debug("Generating predictions for model=%s", model_name)
        if features is None or features.empty:
            raise ValueError("Feature matrix for prediction is empty.")

        if isinstance(model, Pipeline):
            estimator = model
        elif isinstance(model, dict) and "model" in model:
            estimator = model["model"]
        else:
            estimator = model

        X = np.asarray(features, dtype=np.float32) if isinstance(features, pd.DataFrame) else features

        if model_name.startswith("ridge"):
            preds = estimator.predict(features)
            return preds + 1.0

        if model_name.startswith("mlr_prob_"):
            threshold = model.get("threshold", 0.5)
            proba = estimator.predict_proba(features)
            classes = estimator.classes_
            best_idx = np.argmax(proba, axis=1)
            best_prob = proba[np.arange(len(best_idx)), best_idx]
            adjusted = classes[best_idx].astype(float)
            adjusted[best_prob < threshold] += 1
            return adjusted + 1

        if model_name == "mlr":
            preds = estimator.predict(features).astype(float)
            return preds + 1

        if model_name.startswith("svc_prob_"):
            threshold = model.get("threshold", 0.5)
            proba = estimator.predict_proba(features)
            classes = estimator.classes_
            best_idx = np.argmax(proba, axis=1)
            best_prob = proba[np.arange(len(best_idx)), best_idx]
            adjusted = classes[best_idx].astype(float)
            adjusted[best_prob < threshold] += 1
            return adjusted + 1

        if model_name == "svc":
            preds = estimator.predict(features).astype(float)
            return preds + 1

        if model_name == "ann":
            ann_model = model["model"]
            preds = ann_model.predict(X, verbose=0)
            classes = np.argmax(preds, axis=1).astype(float)
            return classes + 1

        raise ValueError(f"Unknown model name '{model_name}' for prediction.")

    def persist_artifacts(self, artifacts: ModelArtifacts, output_paths: Dict[str, Any]) -> None:
        """
        Save trained models, learning curves, and supporting metadata to disk.

        Args:
            artifacts: ModelArtifacts containing objects to persist.
            output_paths: Directory configuration for persistence.
        """
        LOGGER.debug("Persisting artifacts to %s", output_paths)
        models_dir = Path(output_paths.get("models_dir", "results/models"))
        models_dir.mkdir(parents=True, exist_ok=True)

        for name, model in artifacts.models.items():
            file_stub = models_dir / name
            if name == "ann":
                ann_model = model["model"]
                ann_model.save(file_stub.with_suffix(".keras"))
                history_path = file_stub.with_suffix(".history.json")
                history_path.write_text(json.dumps(model.get("history", {}), indent=2))
            elif isinstance(model, Pipeline):
                joblib.dump(model, file_stub.with_suffix(".joblib"))
            elif isinstance(model, dict) and "model" in model:
                joblib.dump(model["model"], file_stub.with_suffix(".joblib"))
            else:
                joblib.dump(model, file_stub.with_suffix(".joblib"))
