"""Additional metrics and evaluation utilities."""

import pandas as pd
import numpy as np
from typing import Dict


def calculate_exposure_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate exposure-related metrics.

    Args:
        df: Data with exposure calculations

    Returns:
        Dictionary of exposure metrics
    """
    return {
        "total_exposure": df["exposure_s"].sum(),
        "max_daily_exposure": df.groupby("athlete_id")["exposure_s"].max().mean(),
        "avg_exposure_per_run": df["exposure_s"].mean(),
        "runs_above_threshold": (df["exposure_s"] > 0).sum(),
    }


def calculate_alert_metrics(predictions: np.ndarray, thresholds: np.ndarray) -> Dict[str, float]:
    """Calculate alert-related metrics.

    Args:
        predictions: Model predictions
        thresholds: Alert thresholds

    Returns:
        Dictionary of alert metrics
    """
    alerts = predictions >= thresholds
    return {
        "alert_rate": alerts.mean(),
        "total_alerts": alerts.sum(),
        "max_alert_level": predictions.max(),
    }
