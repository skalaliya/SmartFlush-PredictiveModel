"""
Machine Learning Models module for SmartFlush Predictive Model.

This module implements:
- Ridge Regression with GridSearchCV and +1 adjustment
- Multinomial Logistic Regression with GridSearchCV and +1 adjustment
- Probability threshold variant of LogReg (MLR_2)
- Support Vector Classification (SVC) with linear and polynomial kernels
- Probability threshold variant of SVC (SVC_2)
- Artificial Neural Network (ANN) with Keras:
  - Dense layers with ReLU activation
  - 20% Dropout regularization
  - Adam optimizer
  - SparseCategoricalCrossentropy loss
  - Sigmoid probabilities for threshold optimization
  - Learning curve plotting
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, History

# Configure logging
logger = logging.getLogger(__name__)


def create_ridge_pipeline(
    alphas: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    cv: int = 5
) -> GridSearchCV:
    """
    Create Ridge Regression pipeline with GridSearchCV.
    
    Args:
        alphas: List of alpha values for regularization
        cv: Number of cross-validation folds
        
    Returns:
        GridSearchCV object with Ridge pipeline
        
    Example:
        >>> ridge_model = create_ridge_pipeline()
        >>> ridge_model.fit(X_train, y_train)
    """
    try:
        logger.info(f"Creating Ridge pipeline with {len(alphas)} alpha values")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])
        
        param_grid = {
            'ridge__alpha': alphas
        }
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        logger.info("Ridge pipeline created")
        return grid_search
        
    except Exception as e:
        logger.error(f"Error creating Ridge pipeline: {e}")
        raise


def train_ridge_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alphas: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    add_one: bool = True
) -> Tuple[Any, Dict]:
    """
    Train Ridge Regression model with GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training targets
        alphas: Alpha values for grid search
        add_one: Whether to add 1 to predictions (default: True)
        
    Returns:
        Tuple of (trained model, training info dict)
        
    Example:
        >>> model, info = train_ridge_model(X_train, y_train)
    """
    try:
        logger.info("Training Ridge Regression model")
        
        ridge = create_ridge_pipeline(alphas)
        ridge.fit(X_train, y_train)
        
        info = {
            'best_params': ridge.best_params_,
            'best_score': ridge.best_score_,
            'add_one': add_one
        }
        
        logger.info(f"Ridge trained. Best params: {info['best_params']}, "
                   f"Best CV score: {info['best_score']:.4f}")
        
        return ridge, info
        
    except Exception as e:
        logger.error(f"Error training Ridge model: {e}")
        raise


def create_logistic_regression_pipeline(
    C_values: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    max_iter: int = 1000,
    cv: int = 5
) -> GridSearchCV:
    """
    Create Multinomial Logistic Regression pipeline with GridSearchCV.
    
    Args:
        C_values: List of regularization parameter values
        max_iter: Maximum iterations for convergence
        cv: Number of cross-validation folds
        
    Returns:
        GridSearchCV object with LogisticRegression pipeline
        
    Example:
        >>> lr_model = create_logistic_regression_pipeline()
        >>> lr_model.fit(X_train, y_train)
    """
    try:
        logger.info(f"Creating Logistic Regression pipeline with {len(C_values)} C values")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=max_iter
            ))
        ])
        
        param_grid = {
            'logistic__C': C_values
        }
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        logger.info("Logistic Regression pipeline created")
        return grid_search
        
    except Exception as e:
        logger.error(f"Error creating Logistic Regression pipeline: {e}")
        raise


def train_logistic_regression_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C_values: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    add_one: bool = True
) -> Tuple[Any, Dict]:
    """
    Train Multinomial Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        C_values: C values for grid search
        add_one: Whether to add 1 to predictions
        
    Returns:
        Tuple of (trained model, training info dict)
        
    Example:
        >>> model, info = train_logistic_regression_model(X_train, y_train)
    """
    try:
        logger.info("Training Logistic Regression model")
        
        lr = create_logistic_regression_pipeline(C_values)
        lr.fit(X_train, y_train)
        
        info = {
            'best_params': lr.best_params_,
            'best_score': lr.best_score_,
            'add_one': add_one
        }
        
        logger.info(f"Logistic Regression trained. Best params: {info['best_params']}, "
                   f"Best CV score: {info['best_score']:.4f}")
        
        return lr, info
        
    except Exception as e:
        logger.error(f"Error training Logistic Regression model: {e}")
        raise


def create_svc_pipeline(
    kernel: str = 'linear',
    C_values: List[float] = [0.1, 1.0, 10.0],
    probability: bool = True,
    cv: int = 5
) -> GridSearchCV:
    """
    Create SVC pipeline with GridSearchCV.
    
    Args:
        kernel: Kernel type ('linear', 'poly', 'rbf')
        C_values: List of C values for regularization
        probability: Enable probability estimates
        cv: Number of cross-validation folds
        
    Returns:
        GridSearchCV object with SVC pipeline
        
    Example:
        >>> svc_model = create_svc_pipeline(kernel='linear')
        >>> svc_model.fit(X_train, y_train)
    """
    try:
        logger.info(f"Creating SVC pipeline with {kernel} kernel")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel=kernel, probability=probability))
        ])
        
        param_grid = {
            'svc__C': C_values
        }
        
        if kernel == 'poly':
            param_grid['svc__degree'] = [2, 3]
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        logger.info("SVC pipeline created")
        return grid_search
        
    except Exception as e:
        logger.error(f"Error creating SVC pipeline: {e}")
        raise


def train_svc_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = 'linear',
    C_values: List[float] = [0.1, 1.0, 10.0],
    probability: bool = True
) -> Tuple[Any, Dict]:
    """
    Train Support Vector Classification model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        kernel: Kernel type ('linear', 'poly')
        C_values: C values for grid search
        probability: Enable probability estimates
        
    Returns:
        Tuple of (trained model, training info dict)
        
    Example:
        >>> model, info = train_svc_model(X_train, y_train, kernel='linear')
    """
    try:
        logger.info(f"Training SVC model with {kernel} kernel")
        
        svc = create_svc_pipeline(kernel, C_values, probability)
        svc.fit(X_train, y_train)
        
        info = {
            'kernel': kernel,
            'best_params': svc.best_params_,
            'best_score': svc.best_score_,
            'probability': probability
        }
        
        logger.info(f"SVC trained. Best params: {info['best_params']}, "
                   f"Best CV score: {info['best_score']:.4f}")
        
        return svc, info
        
    except Exception as e:
        logger.error(f"Error training SVC model: {e}")
        raise


def optimize_probability_threshold(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    threshold_range: Tuple[float, float] = (0.3, 0.9),
    threshold_step: float = 0.05,
    metric: str = 'accuracy'
) -> Tuple[float, Dict]:
    """
    Optimize probability threshold for classification.
    
    Args:
        model: Trained model with predict_proba method
        X_val: Validation features
        y_val: Validation targets
        threshold_range: (min, max) threshold values to try
        threshold_step: Step size for threshold search
        metric: Metric to optimize ('accuracy', 'safe_flush')
        
    Returns:
        Tuple of (best threshold, results dict)
        
    Example:
        >>> best_thresh, results = optimize_probability_threshold(model, X_val, y_val)
    """
    try:
        logger.info(f"Optimizing probability threshold for {metric}")
        
        # Get probability predictions
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_val)
        else:
            logger.warning("Model doesn't support predict_proba, using default predictions")
            return 0.5, {}
        
        # Try different thresholds
        thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)
        results = []
        
        for threshold in thresholds:
            # Apply threshold: choose class with highest probability above threshold
            y_pred = np.argmax(y_proba, axis=1)
            
            # Calculate metric
            if metric == 'accuracy':
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val, y_pred)
            elif metric == 'safe_flush':
                # Safe flush: prediction >= true value
                score = np.mean(y_pred >= y_val)
            else:
                score = 0.0
            
            results.append({
                'threshold': threshold,
                'score': score
            })
        
        # Find best threshold
        results_df = pd.DataFrame(results)
        best_idx = results_df['score'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_score = results_df.loc[best_idx, 'score']
        
        logger.info(f"Best threshold: {best_threshold:.2f} with score: {best_score:.4f}")
        
        return float(best_threshold), {'results': results_df, 'best_score': best_score}
        
    except Exception as e:
        logger.error(f"Error optimizing threshold: {e}")
        raise


def create_ann_model(
    input_dim: int,
    num_classes: int,
    hidden_layers: List[int] = [128, 64, 32],
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Create Artificial Neural Network model with Keras.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate for regularization (default: 0.2)
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled Keras model
        
    Example:
        >>> model = create_ann_model(input_dim=10, num_classes=11)
    """
    try:
        logger.info(f"Creating ANN model with {len(hidden_layers)} hidden layers")
        
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers with ReLU and Dropout
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
            model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer
        model.add(layers.Dense(num_classes, activation='softmax', name='output'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"ANN model created with {model.count_params()} parameters")
        return model
        
    except Exception as e:
        logger.error(f"Error creating ANN model: {e}")
        raise


def train_ann_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_layers: List[int] = [128, 64, 32],
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    early_stopping_patience: int = 10
) -> Tuple[keras.Model, History]:
    """
    Train Artificial Neural Network model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        early_stopping_patience: Patience for early stopping
        
    Returns:
        Tuple of (trained model, training history)
        
    Example:
        >>> model, history = train_ann_model(X_train, y_train, X_val, y_val)
    """
    try:
        logger.info("Training ANN model")
        
        # Ensure targets are integers
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        
        # Get number of classes
        num_classes = len(np.unique(y_train))
        input_dim = X_train.shape[1]
        
        # Create model
        model = create_ann_model(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=1
        )
        
        logger.info(f"ANN training complete. Final val_accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        return model, history
        
    except Exception as e:
        logger.error(f"Error training ANN model: {e}")
        raise


def plot_learning_curves(
    history: History,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Plot training and validation learning curves.
    
    Args:
        history: Keras training history object
        save_path: Path to save the figure
        figsize: Figure size
        
    Example:
        >>> plot_learning_curves(history, 'results/learning_curves.png')
    """
    try:
        logger.info("Plotting learning curves")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss curves
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy curves
        axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.suptitle('Learning Curves', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curves saved to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting learning curves: {e}")
        raise


def predict_with_adjustment(
    model: Any,
    X: np.ndarray,
    add_one: bool = True,
    round_predictions: bool = True
) -> np.ndarray:
    """
    Make predictions with optional +1 adjustment and rounding.
    
    Args:
        model: Trained model
        X: Features to predict on
        add_one: Whether to add 1 to predictions
        round_predictions: Whether to round predictions to integers
        
    Returns:
        Predictions array
        
    Example:
        >>> predictions = predict_with_adjustment(ridge_model, X_test, add_one=True)
    """
    try:
        predictions = model.predict(X)
        
        if add_one:
            predictions = predictions + 1
            logger.debug("Added 1 to predictions")
        
        if round_predictions:
            predictions = np.round(predictions).astype(int)
            logger.debug("Rounded predictions to integers")
        
        # Clip to valid range (assuming flush levels 1-11)
        predictions = np.clip(predictions, 1, 11)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise


def get_probability_predictions(
    model: Any,
    X: np.ndarray
) -> np.ndarray:
    """
    Get probability predictions from a model.
    
    Args:
        model: Trained model with predict_proba method
        X: Features to predict on
        
    Returns:
        Probability predictions array
        
    Example:
        >>> probas = get_probability_predictions(lr_model, X_test)
    """
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            logger.debug(f"Got probability predictions with shape {probabilities.shape}")
            return probabilities
        else:
            logger.warning("Model doesn't support predict_proba")
            return None
        
    except Exception as e:
        logger.error(f"Error getting probability predictions: {e}")
        raise
