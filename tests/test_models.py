"""
Tests for models module
"""

import pytest
import numpy as np
from sklearn.pipeline import Pipeline

from src.models import ModelTrainer, train_models


@pytest.fixture
def sample_training_data():
    """Create sample training data"""
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = X_train[:, 0] * 2 + X_train[:, 1] * 1.5 + np.random.randn(100) * 0.5
    X_test = np.random.randn(20, 5)
    y_test = X_test[:, 0] * 2 + X_test[:, 1] * 1.5 + np.random.randn(20) * 0.5
    return X_train, y_train, X_test, y_test


@pytest.fixture
def config():
    """Create sample configuration"""
    return {
        'models': {
            'ridge': {
                'enabled': True,
                'alpha_range': [0.1, 1.0, 10.0]
            },
            'mlr': {
                'enabled': True,
                'fit_intercept': True
            },
            'svc': {
                'enabled': False  # Disable for faster tests
            },
            'ann': {
                'enabled': False  # Disable for faster tests
            }
        },
        'grid_search': {
            'cv_folds': 3,
            'scoring': 'neg_mean_absolute_error',
            'n_jobs': 1
        }
    }


def test_model_trainer_init(config):
    """Test ModelTrainer initialization"""
    trainer = ModelTrainer(config)
    
    assert trainer.config == config
    assert isinstance(trainer.models, dict)
    assert isinstance(trainer.best_models, dict)


def test_create_ridge_pipeline(config):
    """Test Ridge pipeline creation"""
    trainer = ModelTrainer(config)
    
    pipeline, param_grid = trainer.create_ridge_pipeline()
    
    assert isinstance(pipeline, Pipeline)
    assert isinstance(param_grid, dict)
    assert 'ridge__alpha' in param_grid


def test_create_mlr_pipeline(config):
    """Test MLR pipeline creation"""
    trainer = ModelTrainer(config)
    
    pipeline = trainer.create_mlr_pipeline()
    
    assert isinstance(pipeline, Pipeline)
    assert 'scaler' in pipeline.named_steps
    assert 'mlr' in pipeline.named_steps


def test_create_svc_pipeline(config):
    """Test SVC pipeline creation"""
    trainer = ModelTrainer(config)
    
    pipeline, param_grid = trainer.create_svc_pipeline()
    
    assert isinstance(pipeline, Pipeline)
    assert isinstance(param_grid, dict)
    assert 'svr__C' in param_grid


def test_train_ridge(config, sample_training_data):
    """Test Ridge training"""
    X_train, y_train, _, _ = sample_training_data
    
    trainer = ModelTrainer(config)
    grid_search = trainer.train_ridge(X_train, y_train)
    
    assert 'ridge' in trainer.best_models
    assert 'ridge' in trainer.grid_searches
    assert grid_search.best_estimator_ is not None


def test_train_mlr(config, sample_training_data):
    """Test MLR training"""
    X_train, y_train, _, _ = sample_training_data
    
    trainer = ModelTrainer(config)
    pipeline = trainer.train_mlr(X_train, y_train)
    
    assert 'mlr' in trainer.best_models
    assert isinstance(pipeline, Pipeline)


def test_predict(config, sample_training_data):
    """Test model prediction"""
    X_train, y_train, X_test, _ = sample_training_data
    
    trainer = ModelTrainer(config)
    trainer.train_mlr(X_train, y_train)
    
    predictions = trainer.predict('mlr', X_test)
    
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_test)


def test_train_models_function(config, sample_training_data):
    """Test train_models convenience function"""
    X_train, y_train, _, _ = sample_training_data
    
    trainer = train_models(X_train, y_train, config)
    
    assert isinstance(trainer, ModelTrainer)
    assert len(trainer.best_models) > 0
