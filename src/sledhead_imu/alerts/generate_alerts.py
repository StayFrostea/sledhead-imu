"""Generate alerts based on predictions and thresholds."""

import pandas as pd
import numpy as np
from typing import Dict, List
from .thresholds import apply_severity_thresholds


def generate_alerts_from_severity(
    severity_predictions: np.ndarray, 
    athlete_ids: List[str],
    severity_mapping: Dict[int, str] = None
) -> pd.DataFrame:
    """Generate alerts from severity predictions.
    
    Args:
        severity_predictions: Severity predictions (0-5)
        athlete_ids: List of athlete IDs
        severity_mapping: Mapping of severity to alert level
        
    Returns:
        DataFrame with alerts
    """
    # Apply severity thresholds
    alerts_df = apply_severity_thresholds(severity_predictions, severity_mapping)
    
    # Add athlete IDs
    alerts_df['athlete_id'] = athlete_ids
    
    # Reorder columns
    col_order = ['athlete_id', 'severity', 'alert_level', 'none_alert', 
                 'low_alert', 'medium_alert', 'high_alert', 'critical_alert']
    alerts_df = alerts_df[col_order]
    
    return alerts_df


def generate_alerts(
    predictions: np.ndarray, athlete_ids: List[str], thresholds: Dict[str, float]
) -> pd.DataFrame:
    """Legacy function for backward compatibility.

    Args:
        predictions: Model predictions
        athlete_ids: List of athlete IDs
        thresholds: Alert thresholds

    Returns:
        DataFrame with alerts
    """
    alerts = pd.DataFrame(
        {"athlete_id": athlete_ids, "prediction": predictions, "alert_level": "none"}
    )

    # Assign alert levels based on thresholds
    for level, threshold in sorted(thresholds.items(), key=lambda x: x[1], reverse=True):
        mask = (predictions >= threshold) & (alerts["alert_level"] == "none")
        alerts.loc[mask, "alert_level"] = level

    return alerts


def summarize_alerts(alerts_df: pd.DataFrame) -> Dict[str, int]:
    """Summarize alert statistics.

    Args:
        alerts_df: DataFrame with alerts

    Returns:
        Dictionary of alert counts by level
    """
    return alerts_df["alert_level"].value_counts().to_dict()


def get_critical_alerts(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """Get only critical/high alerts for immediate attention.
    
    Args:
        alerts_df: DataFrame with alerts
        
    Returns:
        DataFrame filtered to only critical and high alerts
    """
    critical_levels = ['critical', 'high']
    return alerts_df[alerts_df['alert_level'].isin(critical_levels)]
