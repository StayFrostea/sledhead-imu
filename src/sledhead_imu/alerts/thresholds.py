"""Threshold management for alerts."""

import pandas as pd
import numpy as np
from typing import Dict


def define_alert_thresholds_from_severity(severity_mapping: Dict[int, str] = None) -> Dict[int, str]:
    """Define alert levels from severity scores (0-5).
    
    Args:
        severity_mapping: Optional custom mapping of severity -> alert level
        
    Returns:
        Dictionary mapping severity scores to alert levels
    """
    if severity_mapping is None:
        # Default mapping for Random Forest severity outputs
        severity_mapping = {
            0: "none",      # No symptoms
            1: "none",      # Minor symptoms
            2: "low",       # Monitor
            3: "medium",    # Caution
            4: "high",      # Bench recommendation
            5: "critical"   # Immediate intervention
        }
    
    return severity_mapping


def apply_severity_thresholds(predictions: np.ndarray, severity_mapping: Dict[int, str] = None) -> pd.DataFrame:
    """Apply severity-to-alert mapping to predictions.

    Args:
        predictions: Severity predictions (0-5)
        severity_mapping: Mapping of severity scores to alert levels

    Returns:
        DataFrame with predictions and alert levels
    """
    if severity_mapping is None:
        severity_mapping = define_alert_thresholds_from_severity()
    
    results = pd.DataFrame({"severity": predictions})
    
    # Map severity to alert level
    results["alert_level"] = results["severity"].map(severity_mapping)
    results["alert_level"] = results["alert_level"].fillna("none")  # Handle unknown severities
    
    # Create boolean flags for each alert level
    alert_levels = ["none", "low", "medium", "high", "critical"]
    for level in alert_levels:
        results[f"{level}_alert"] = results["alert_level"] == level
    
    return results


def apply_thresholds(predictions: np.ndarray, thresholds: Dict[str, float]) -> pd.DataFrame:
    """Legacy function for backward compatibility.

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
