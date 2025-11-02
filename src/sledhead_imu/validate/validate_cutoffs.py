"""Validate model performance and define cutoffs."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple


def validate_model_performance(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate model performance metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of performance metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def validate_per_class_performance(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Calculate per-class performance metrics for classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        DataFrame with per-class metrics
    """
    unique_classes = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    
    per_class_metrics = []
    for cls in unique_classes:
        metrics = {
            'class': int(cls),
            'precision': precision_score(y_true == cls, y_pred == cls, zero_division=0),
            'recall': recall_score(y_true == cls, y_pred == cls, zero_division=0),
            'f1': f1_score(y_true == cls, y_pred == cls, zero_division=0),
            'support': int(np.sum(y_true == cls))
        }
        per_class_metrics.append(metrics)
    
    return pd.DataFrame(per_class_metrics)


def get_confusion_matrix_report(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    """Get confusion matrix and classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Tuple of (confusion_matrix, classification_report_df)
    """
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Convert to DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    return cm, report_df


def define_bench_cutoffs(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Define benchmark cutoffs based on validation performance.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        Optimal threshold for bench decisions
    """
    # Simple threshold optimization
    thresholds = np.linspace(0.1, 0.9, 9)
    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred_thresh, average="weighted")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
