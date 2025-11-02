"""Test Random Forest model functions."""

import pandas as pd
import numpy as np
from sledhead_imu.models.random_forest import (
    train_random_forest,
    predict_random_forest,
    predict_proba_random_forest,
    evaluate_random_forest
)


def test_train_random_forest():
    """Test Random Forest training."""
    np.random.seed(42)
    
    # Create sample data
    X_train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    y_train = pd.Series(np.random.randint(0, 3, 100))
    
    X_val = pd.DataFrame({
        'feature1': np.random.randn(20),
        'feature2': np.random.randn(20),
        'feature3': np.random.randn(20)
    })
    y_val = pd.Series(np.random.randint(0, 3, 20))
    
    # Train model
    model = train_random_forest(X_train, y_train, X_val, y_val)
    
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
    

def test_predict_random_forest():
    """Test Random Forest predictions."""
    np.random.seed(42)
    
    X_train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
    })
    y_train = pd.Series(np.random.randint(0, 2, 100))
    
    X_val = pd.DataFrame({
        'feature1': np.random.randn(20),
        'feature2': np.random.randn(20),
    })
    y_val = pd.Series(np.random.randint(0, 2, 20))
    
    # Train and predict
    model = train_random_forest(X_train, y_train, X_val, y_val)
    predictions = predict_random_forest(model, X_val)
    
    assert predictions is not None
    assert len(predictions) == len(X_val)
    assert predictions.dtype in [np.int64, np.int32]


def test_predict_proba_random_forest():
    """Test Random Forest probability predictions."""
    np.random.seed(42)
    
    X_train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
    })
    y_train = pd.Series(np.random.randint(0, 2, 100))
    
    X_val = pd.DataFrame({
        'feature1': np.random.randn(20),
        'feature2': np.random.randn(20),
    })
    y_val = pd.Series(np.random.randint(0, 2, 20))
    
    # Train and predict probabilities
    model = train_random_forest(X_train, y_train, X_val, y_val)
    proba = predict_proba_random_forest(model, X_val)
    
    assert proba is not None
    assert proba.shape[0] == len(X_val)
    assert proba.shape[1] >= 2  # At least 2 classes
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_evaluate_random_forest():
    """Test Random Forest evaluation."""
    np.random.seed(42)
    
    X_train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
    })
    y_train = pd.Series(np.random.randint(0, 2, 100))
    
    X_val = pd.DataFrame({
        'feature1': np.random.randn(20),
        'feature2': np.random.randn(20),
    })
    y_val = pd.Series(np.random.randint(0, 2, 20))
    
    # Train and evaluate
    model = train_random_forest(X_train, y_train, X_val, y_val)
    results = evaluate_random_forest(model, X_val, y_val)
    
    assert 'accuracy' in results
    assert 'confusion_matrix' in results
    assert 'feature_importance' in results
    assert 'classification_report' in results
    assert 'predictions' in results
    assert 'probabilities' in results
    
    assert results['accuracy'] >= 0.0 and results['accuracy'] <= 1.0
    assert isinstance(results['feature_importance'], pd.DataFrame)
    assert 'feature' in results['feature_importance'].columns
    assert 'importance' in results['feature_importance'].columns
