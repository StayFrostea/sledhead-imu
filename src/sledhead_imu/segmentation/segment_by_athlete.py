"""Segment data by athlete."""

import pandas as pd
from typing import Dict


def segment_by_athlete(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Segment data by athlete ID.

    Args:
        df: IMU data with athlete_id column

    Returns:
        Dictionary mapping athlete_id to their data segments
    """
    if "athlete_id" not in df.columns:
        raise ValueError("DataFrame must contain 'athlete_id' column")

    segments = {}
    for athlete_id in df["athlete_id"].unique():
        segments[str(athlete_id)] = df[df["athlete_id"] == athlete_id].copy()

    return segments
