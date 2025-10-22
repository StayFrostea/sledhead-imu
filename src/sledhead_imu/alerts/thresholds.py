"""Threshold management for alerts."""

import pandas as pd
import numpy as np
from typing import Dict


def define_alert_thresholds(validation_results: Dict[str, float]) -> Dict[str, float]:
    """Define alert thresholds based on validation results.

    Args:
        validation_results: Model validation metrics

    Returns:
        Dictionary of alert thresholds
    """
    # Simple threshold definition based on performance
    base_threshold = 0.5

    return {
        "low_alert": base_threshold * 0.7,
        "medium_alert": base_threshold,
        "high_alert": base_threshold * 1.3,
        "critical_alert": base_threshold * 1.5,
    }


def apply_thresholds(predictions: np.ndarray, thresholds: Dict[str, float]) -> pd.DataFrame:
    """Apply thresholds to predictions.

    Args:
        predictions: Model predictions
        thresholds: Alert thresholds

    Returns:
        DataFrame with alert levels
    """
    results = pd.DataFrame({"prediction": predictions})

    for level, threshold in thresholds.items():
        results[f"{level}_flag"] = predictions >= threshold

    return results
