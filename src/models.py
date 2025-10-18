"""
Models Module
Machine learning models for flush-volume prediction using sklearn pipelines and GridSearchCV
Includes Ridge, MLR, SVC, and ANN/Keras models
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import joblib

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and manage machine learning models"""
    
    def __init__(self, config: dict = None):
        """
        Initialize ModelTrainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        self.best_models = {}
        self.grid_searches = {}
        
    def create_ridge_pipeline(self) -> Tuple[Pipeline, dict]:
        """
        Create Ridge regression pipeline with parameter grid
        
        Returns:
            Tuple of (pipeline, parameter grid)
        """
        logger.info("Creating Ridge regression pipeline")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])
        
        model_config = self.config.get('models', {}).get('ridge', {})
        alpha_range = model_config.get('alpha_range', [0.1, 1.0, 10.0, 100.0])
        
        param_grid = {
            'ridge__alpha': alpha_range,
            'ridge__fit_intercept': [True, False],
            'ridge__solver': ['auto', 'svd', 'lsqr']
        }
        
        return pipeline, param_grid
    
    def create_mlr_pipeline(self) -> Pipeline:
        """
        Create Multiple Linear Regression pipeline
        
        Returns:
            MLR pipeline
        """
        logger.info("Creating MLR pipeline")
        
        model_config = self.config.get('models', {}).get('mlr', {})
        fit_intercept = model_config.get('fit_intercept', True)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlr', LinearRegression(fit_intercept=fit_intercept))
        ])
        
        return pipeline
    
    def create_svc_pipeline(self) -> Tuple[Pipeline, dict]:
        """
        Create Support Vector Regression pipeline with parameter grid
        
        Returns:
            Tuple of (pipeline, parameter grid)
        """
        logger.info("Creating SVR pipeline")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR())
        ])
        
        model_config = self.config.get('models', {}).get('svc', {})
        C_range = model_config.get('C_range', [0.1, 1.0, 10.0])
        kernel_options = model_config.get('kernel_options', ['linear', 'rbf'])
        
        param_grid = {
            'svr__C': C_range,
            'svr__kernel': kernel_options,
            'svr__gamma': ['scale', 'auto'],
            'svr__epsilon': [0.1, 0.2]
        }
        
        return pipeline, param_grid
    
    def create_ann_model(self, input_dim: int) -> keras.Model:
        """
        Create Artificial Neural Network using Keras
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        logger.info("Creating ANN model")
        
        model_config = self.config.get('models', {}).get('ann', {})
        hidden_layers = model_config.get('hidden_layers', [64, 32, 16])
        dropout_rate = model_config.get('dropout_rate', 0.2)
        learning_rate = model_config.get('learning_rate', 0.001)
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
        model.add(layers.Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        logger.info(f"ANN architecture: {hidden_layers} hidden units")
        
        return model
    
    def train_ridge(self, X_train: np.ndarray, y_train: np.ndarray) -> GridSearchCV:
        """
        Train Ridge regression with GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Fitted GridSearchCV object
        """
        logger.info("Training Ridge regression")
        
        pipeline, param_grid = self.create_ridge_pipeline()
        
        grid_config = self.config.get('grid_search', {})
        cv_folds = grid_config.get('cv_folds', 5)
        scoring = grid_config.get('scoring', 'neg_mean_absolute_error')
        n_jobs = grid_config.get('n_jobs', -1)
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best Ridge params: {grid_search.best_params_}")
        logger.info(f"Best Ridge score: {-grid_search.best_score_:.4f}")
        
        self.grid_searches['ridge'] = grid_search
        self.best_models['ridge'] = grid_search.best_estimator_
        
        return grid_search
    
    def train_mlr(self, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        """
        Train Multiple Linear Regression
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Fitted MLR pipeline
        """
        logger.info("Training MLR")
        
        pipeline = self.create_mlr_pipeline()
        pipeline.fit(X_train, y_train)
        
        self.best_models['mlr'] = pipeline
        
        return pipeline
    
    def train_svr(self, X_train: np.ndarray, y_train: np.ndarray) -> GridSearchCV:
        """
        Train Support Vector Regression with GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Fitted GridSearchCV object
        """
        logger.info("Training SVR")
        
        pipeline, param_grid = self.create_svc_pipeline()
        
        grid_config = self.config.get('grid_search', {})
        cv_folds = grid_config.get('cv_folds', 5)
        scoring = grid_config.get('scoring', 'neg_mean_absolute_error')
        n_jobs = grid_config.get('n_jobs', -1)
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best SVR params: {grid_search.best_params_}")
        logger.info(f"Best SVR score: {-grid_search.best_score_:.4f}")
        
        self.grid_searches['svr'] = grid_search
        self.best_models['svr'] = grid_search.best_estimator_
        
        return grid_search
    
    def train_ann(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: Optional[np.ndarray] = None,
                 y_val: Optional[np.ndarray] = None) -> keras.Model:
        """
        Train Artificial Neural Network
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Trained Keras model
        """
        logger.info("Training ANN")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # Create model
        model = self.create_ann_model(X_train.shape[1])
        
        # Callbacks
        model_config = self.config.get('models', {}).get('ann', {})
        epochs = model_config.get('epochs', 100)
        batch_size = model_config.get('batch_size', 32)
        
        early_stopping = EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Store model and scaler
        self.best_models['ann'] = {'model': model, 'scaler': scaler}
        
        logger.info("ANN training complete")
        
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: Optional[np.ndarray] = None,
                        y_val: Optional[np.ndarray] = None):
        """
        Train all enabled models
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        logger.info("Training all models")
        
        models_config = self.config.get('models', {})
        
        # Train Ridge
        if models_config.get('ridge', {}).get('enabled', True):
            self.train_ridge(X_train, y_train)
        
        # Train MLR
        if models_config.get('mlr', {}).get('enabled', True):
            self.train_mlr(X_train, y_train)
        
        # Train SVR
        if models_config.get('svc', {}).get('enabled', True):
            self.train_svr(X_train, y_train)
        
        # Train ANN
        if models_config.get('ann', {}).get('enabled', True):
            self.train_ann(X_train, y_train, X_val, y_val)
        
        logger.info("All models trained")
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with specified model
        
        Args:
            model_name: Name of model to use
            X: Features to predict on
            
        Returns:
            Predictions array
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        if model_name == 'ann':
            model_dict = self.best_models['ann']
            X_scaled = model_dict['scaler'].transform(X)
            predictions = model_dict['model'].predict(X_scaled).flatten()
        else:
            predictions = self.best_models[model_name].predict(X)
        
        return predictions
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save trained model to disk
        
        Args:
            model_name: Name of model to save
            filepath: Path to save model
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        if model_name == 'ann':
            # Save Keras model and scaler separately
            model_dict = self.best_models['ann']
            model_dict['model'].save(f"{filepath}_model.h5")
            joblib.dump(model_dict['scaler'], f"{filepath}_scaler.pkl")
            logger.info(f"ANN model saved to {filepath}_model.h5 and {filepath}_scaler.pkl")
        else:
            joblib.dump(self.best_models[model_name], filepath)
            logger.info(f"Model '{model_name}' saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """
        Load trained model from disk
        
        Args:
            model_name: Name to assign to loaded model
            filepath: Path to load model from
        """
        if model_name == 'ann':
            # Load Keras model and scaler
            model = keras.models.load_model(f"{filepath}_model.h5")
            scaler = joblib.load(f"{filepath}_scaler.pkl")
            self.best_models['ann'] = {'model': model, 'scaler': scaler}
            logger.info(f"ANN model loaded from {filepath}")
        else:
            self.best_models[model_name] = joblib.load(filepath)
            logger.info(f"Model '{model_name}' loaded from {filepath}")


def train_models(X_train: np.ndarray, 
                y_train: np.ndarray,
                config: dict,
                X_val: Optional[np.ndarray] = None,
                y_val: Optional[np.ndarray] = None) -> ModelTrainer:
    """
    Convenience function to train all models
    
    Args:
        X_train: Training features
        y_train: Training target
        config: Configuration dictionary
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        Trained ModelTrainer instance
    """
    trainer = ModelTrainer(config)
    trainer.train_all_models(X_train, y_train, X_val, y_val)
    return trainer
