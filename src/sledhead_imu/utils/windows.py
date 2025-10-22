"""Window-based processing utilities."""

import pandas as pd
from typing import List


def create_sliding_windows(
    df: pd.DataFrame, window_size: int, step_size: int
) -> List[pd.DataFrame]:
    """Create sliding windows from DataFrame.

    Args:
        df: Input DataFrame
        window_size: Size of each window
        step_size: Step size between windows

    Returns:
        List of window DataFrames
    """
    windows = []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i : i + window_size].copy()
        window["window_id"] = i // step_size
        windows.append(window)

    return windows


def extract_window_features(window: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    """Extract features from a window.

    Args:
        window: Window DataFrame
        feature_cols: Columns to extract features from

    Returns:
        Series of window features
    """
    features = {}
    for col in feature_cols:
        if col in window.columns:
            features[f"{col}_mean"] = window[col].mean()
            features[f"{col}_std"] = window[col].std()
            features[f"{col}_max"] = window[col].max()
            features[f"{col}_min"] = window[col].min()

    return pd.Series(features)
