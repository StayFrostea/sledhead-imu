"""Generate alerts based on predictions and thresholds."""

import pandas as pd
import numpy as np
from typing import Dict, List


def generate_alerts(
    predictions: np.ndarray, athlete_ids: List[str], thresholds: Dict[str, float]
) -> pd.DataFrame:
    """Generate alerts for athletes based on predictions.

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
