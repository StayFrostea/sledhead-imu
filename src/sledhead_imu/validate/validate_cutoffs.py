"""Validate model performance and define cutoffs."""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict


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
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }


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
