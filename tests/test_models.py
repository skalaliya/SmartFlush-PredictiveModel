"""
Unit tests for models module.
"""

import pytest
import numpy as np
from src.models import (
    create_ridge_pipeline,
    create_logistic_regression_pipeline,
    create_svc_pipeline,
    create_ann_model,
    predict_with_adjustment,
    get_probability_predictions
)


class TestRidgePipeline:
    """Tests for Ridge Regression pipeline."""
    
    def test_create_ridge_pipeline(self):
        """Test Ridge pipeline creation."""
        pipeline = create_ridge_pipeline(alphas=[0.1, 1.0, 10.0])
        
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')
    
    @pytest.mark.slow
    def test_ridge_pipeline_fit(self):
        """Test fitting Ridge pipeline."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        pipeline = create_ridge_pipeline(alphas=[0.1, 1.0], cv=2)
        pipeline.fit(X, y)
        
        assert hasattr(pipeline, 'best_params_')
        assert hasattr(pipeline, 'best_score_')


class TestLogisticRegressionPipeline:
    """Tests for Logistic Regression pipeline."""
    
    def test_create_logistic_regression_pipeline(self):
        """Test Logistic Regression pipeline creation."""
        pipeline = create_logistic_regression_pipeline(C_values=[0.1, 1.0, 10.0])
        
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')
    
    @pytest.mark.slow
    def test_logistic_regression_pipeline_fit(self):
        """Test fitting Logistic Regression pipeline."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        
        pipeline = create_logistic_regression_pipeline(C_values=[0.1, 1.0], cv=2)
        pipeline.fit(X, y)
        
        assert hasattr(pipeline, 'best_params_')
        assert hasattr(pipeline, 'best_score_')


class TestSVCPipeline:
    """Tests for SVC pipeline."""
    
    def test_create_svc_pipeline_linear(self):
        """Test SVC pipeline creation with linear kernel."""
        pipeline = create_svc_pipeline(kernel='linear', C_values=[0.1, 1.0])
        
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')
    
    def test_create_svc_pipeline_poly(self):
        """Test SVC pipeline creation with polynomial kernel."""
        pipeline = create_svc_pipeline(kernel='poly', C_values=[0.1, 1.0])
        
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')


class TestANNModel:
    """Tests for ANN model creation."""
    
    def test_create_ann_model_basic(self):
        """Test basic ANN model creation."""
        model = create_ann_model(
            input_dim=10,
            num_classes=11,
            hidden_layers=[64, 32],
            dropout_rate=0.2
        )
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert len(model.layers) > 0
    
    def test_create_ann_model_different_architectures(self):
        """Test ANN model with different architectures."""
        architectures = [
            [128, 64, 32],
            [64, 32],
            [256, 128]
        ]
        
        for hidden_layers in architectures:
            model = create_ann_model(
                input_dim=10,
                num_classes=11,
                hidden_layers=hidden_layers
            )
            assert model is not None


class TestPredictions:
    """Tests for prediction functions."""
    
    def test_predict_with_adjustment_no_adjustment(self):
        """Test predictions without adjustment."""
        from sklearn.linear_model import Ridge
        
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        y_train = np.random.randn(50)
        X_test = np.random.randn(10, 3)
        
        model = Ridge()
        model.fit(X_train, y_train)
        
        predictions = predict_with_adjustment(model, X_test, add_one=False, round_predictions=False)
        
        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_with_adjustment_with_adjustment(self):
        """Test predictions with +1 adjustment."""
        from sklearn.linear_model import Ridge
        
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        y_train = np.random.randn(50)
        X_test = np.random.randn(10, 3)
        
        model = Ridge()
        model.fit(X_train, y_train)
        
        predictions_no_adj = predict_with_adjustment(model, X_test, add_one=False, round_predictions=False)
        predictions_adj = predict_with_adjustment(model, X_test, add_one=True, round_predictions=False)
        
        assert np.allclose(predictions_adj, predictions_no_adj + 1)
    
    def test_predict_with_rounding(self):
        """Test predictions with rounding."""
        from sklearn.linear_model import Ridge
        
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        y_train = np.random.randn(50)
        X_test = np.random.randn(10, 3)
        
        model = Ridge()
        model.fit(X_train, y_train)
        
        predictions = predict_with_adjustment(model, X_test, add_one=False, round_predictions=True)
        
        assert all(p == int(p) for p in predictions)
    
    def test_get_probability_predictions(self):
        """Test getting probability predictions."""
        from sklearn.linear_model import LogisticRegression
        
        np.random.seed(42)
        X_train = np.random.randn(50, 3)
        y_train = np.random.randint(0, 3, 50)
        X_test = np.random.randn(10, 3)
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        probas = get_probability_predictions(model, X_test)
        
        assert probas is not None
        assert probas.shape[0] == 10
        assert probas.shape[1] == 3
        assert np.allclose(probas.sum(axis=1), 1.0)


class TestModelIntegration:
    """Integration tests for model training."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_ridge_full_pipeline(self):
        """Test complete Ridge training pipeline."""
        from src.models import train_ridge_model
        
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        
        model, info = train_ridge_model(X_train, y_train, alphas=[0.1, 1.0], add_one=True)
        
        assert model is not None
        assert 'best_params' in info
        assert 'best_score' in info
        assert 'add_one' in info
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_logistic_regression_full_pipeline(self):
        """Test complete Logistic Regression training pipeline."""
        from src.models import train_logistic_regression_model
        
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 3, 100)
        
        model, info = train_logistic_regression_model(X_train, y_train, C_values=[0.1, 1.0], add_one=False)
        
        assert model is not None
        assert 'best_params' in info
        assert 'best_score' in info
