"""Time synchronization utilities."""

import pandas as pd
from typing import Tuple


def sync_timestamps(
    df1: pd.DataFrame, df2: pd.DataFrame, tolerance: pd.Timedelta = pd.Timedelta("1s")
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Synchronize timestamps between two DataFrames.

    Args:
        df1: First DataFrame with timestamp column
        df2: Second DataFrame with timestamp column
        tolerance: Maximum time difference for sync

    Returns:
        Tuple of synchronized DataFrames
    """
    # Align timestamps to nearest values within tolerance
    df1_sync = df1.copy()
    df2_sync = df2.copy()

    # Simple nearest neighbor matching
    for idx, row in df1_sync.iterrows():
        time_diff = abs(df2_sync["timestamp"] - row["timestamp"])
        closest_idx = time_diff.idxmin()
        if time_diff[closest_idx] <= tolerance:
            df1_sync.loc[idx, "sync_idx"] = closest_idx

    return df1_sync, df2_sync
