"""Median filtering for IMU data."""

import pandas as pd
from scipy import signal
from typing import Optional


def median_filter(
    df: pd.DataFrame,
    kernel_size: int = 11,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """Apply median filter to IMU data.
    
    Optimized for head-inclusive motion at 2000 Hz:
    - Kernel: 11 samples (~5.5 ms at 2000 Hz)
    - Removes impulse noise while preserving edges

    Args:
        df: IMU data
        kernel_size: Size of median kernel (default: 11)
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
            # Check if we have enough data points for median filter
            if len(df[col]) >= kernel_size:
                df_filtered[f"{col}_median"] = signal.medfilt(df[col], kernel_size=kernel_size)
            else:
                # For short sequences, use a smaller kernel or just copy
                if len(df[col]) >= 3:
                    small_kernel = min(kernel_size, len(df[col]) if len(df[col]) % 2 == 1 else len(df[col]) - 1)
                    df_filtered[f"{col}_median"] = signal.medfilt(df[col], kernel_size=small_kernel)
                else:
                    print(f"Warning: Column {col} has only {len(df[col])} samples, skipping median filter")
                    df_filtered[f"{col}_median"] = df[col]

    return df_filtered
