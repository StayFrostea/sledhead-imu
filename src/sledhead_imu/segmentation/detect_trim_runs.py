"""Detect and trim run segments."""

import pandas as pd
import numpy as np
from typing import Tuple, List


def detect_run_segments(df: pd.DataFrame, threshold: float = 1.5) -> List[Tuple[int, int]]:
    """Detect run segments based on activity threshold.

    Args:
        df: IMU data with g_mag column
        threshold: Activity threshold for run detection

    Returns:
        List of (start_idx, end_idx) tuples for run segments
    """
    # Find periods above threshold
    above_threshold = df["g_mag"] > threshold

    # Find start and end points
    starts = np.where(above_threshold & ~above_threshold.shift(1, fill_value=False))[0]
    ends = np.where(~above_threshold & above_threshold.shift(1, fill_value=False))[0]

    # Handle edge cases
    if len(starts) > len(ends):
        ends = np.append(ends, len(df))
    if len(ends) > len(starts):
        starts = np.append(0, starts)

    segments = [(int(start), int(end)) for start, end in zip(starts, ends)]
    return segments


def trim_run_segment(df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    """Trim a run segment from the data.

    Args:
        df: Full IMU data
        start_idx: Start index of segment
        end_idx: End index of segment

    Returns:
        Trimmed segment
    """
    return df.iloc[start_idx:end_idx].copy()
