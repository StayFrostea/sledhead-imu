"""Random Forest classifier for head impact prediction."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Any


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Dict[str, Any] = None
) -> RandomForestClassifier:
    """Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Model configuration dict
        
    Returns:
        Trained Random Forest model
    """
    if config is None:
        config = {}
    
    # Default hyperparameters for Random Forest
    params = {
        'n_estimators': config.get('n_estimators', 100),
        'max_depth': config.get('max_depth', 20),
        'min_samples_split': config.get('min_samples_split', 2),
        'min_samples_leaf': config.get('min_samples_leaf', 1),
        'max_features': config.get('max_features', 'sqrt'),
        'class_weight': config.get('class_weight', 'balanced'),
        'random_state': config.get('random_state', 42),
        'n_jobs': config.get('n_jobs', -1),
        'verbose': config.get('verbose', 0)
    }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    return model


def predict_random_forest(
    model: RandomForestClassifier,
    X: pd.DataFrame
) -> np.ndarray:
    """Make predictions with Random Forest.
    
    Args:
        model: Trained Random Forest model
        X: Features
        
    Returns:
        Predictions
    """
    return model.predict(X)


def predict_proba_random_forest(
    model: RandomForestClassifier,
    X: pd.DataFrame
) -> np.ndarray:
    """Get prediction probabilities with Random Forest.
    
    Args:
        model: Trained Random Forest model
        X: Features
        
    Returns:
        Probability predictions (n_samples, n_classes)
    """
    return model.predict_proba(X)


def evaluate_random_forest(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y_true: pd.Series
) -> Dict[str, Any]:
    """Evaluate Random Forest model.
    
    Args:
        model: Trained Random Forest model
        X: Features
        y_true: True labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    results = {
        'classification_report': report,
        'confusion_matrix': cm,
        'feature_importance': feature_importance,
        'predictions': y_pred,
        'probabilities': y_proba,
        'accuracy': report['accuracy']
    }
    
    return results
