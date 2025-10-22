"""Savitzky-Golay filtering for IMU data."""

import pandas as pd
from scipy import signal
from typing import Optional


def savgol_filter(
    df: pd.DataFrame,
    window_length: int = 51,
    polyorder: int = 3,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """Apply Savitzky-Golay filter to IMU data.
    
    Optimized for head-inclusive motion at 2000 Hz:
    - Window: 51 samples (~25 ms at 2000 Hz)
    - Polynomial order: 3
    - Preserves signal shape while smoothing

    Args:
        df: IMU data
        window_length: Length of filter window (default: 51)
        polyorder: Order of polynomial (default: 3)
        columns: Columns to filter (default: auto-detect acceleration columns)

    Returns:
        Filtered DataFrame
    """
    if columns is None:
        # Auto-detect acceleration columns
        columns = [col for col in df.columns if col in ['x', 'y', 'z', 'g_x', 'g_y', 'g_z', 'x_mean', 'y_mean', 'z_mean']]

    df_filtered = df.copy()

    for col in columns:
        if col in df.columns:
            # Check if we have enough data points for Savitzky-Golay filter
            if len(df[col]) >= window_length:
                df_filtered[f"{col}_savgol"] = signal.savgol_filter(df[col], window_length, polyorder)
            else:
                # For short sequences, use a smaller window or just copy
                if len(df[col]) >= 5:
                    # Use a smaller window that fits the data
                    small_window = min(window_length, len(df[col]) if len(df[col]) % 2 == 1 else len(df[col]) - 1)
                    df_filtered[f"{col}_savgol"] = signal.savgol_filter(df[col], small_window, polyorder)
                else:
                    print(f"Warning: Column {col} has only {len(df[col])} samples, skipping Savitzky-Golay filter")
                    df_filtered[f"{col}_savgol"] = df[col]

    return df_filtered


def rolling_average_filter(
    df: pd.DataFrame, window_size: int = 10, columns: Optional[list] = None
) -> pd.DataFrame:
    """Apply rolling average filter to IMU data.

    Args:
        df: IMU data
        window_size: Size of rolling window
        columns: Columns to filter (default: g_x, g_y, g_z)

    Returns:
        Filtered DataFrame
    """
    if columns is None:
        columns = ["g_x", "g_y", "g_z"]

    df_filtered = df.copy()

    for col in columns:
        if col in df.columns:
            df_filtered[f"{col}_rolling"] = df[col].rolling(window=window_size, center=True).mean()

    return df_filtered
