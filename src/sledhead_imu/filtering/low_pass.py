"""Low-pass filtering for IMU data."""

import pandas as pd
from scipy import signal
from typing import Optional


def low_pass_filter(
    df: pd.DataFrame,
    cutoff_freq: float = 30.0,
    sampling_rate: float = 2000.0,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """Apply low-pass Butterworth filter to IMU data.
    
    Optimized for head-inclusive motion at 2000 Hz:
    - Cutoff: 30 Hz
    - Order: 4
    - Zero-phase filtering with filtfilt

    Args:
        df: IMU data
        cutoff_freq: Cutoff frequency in Hz (default: 30.0)
        sampling_rate: Sampling rate in Hz (default: 2000.0)
        columns: Columns to filter (default: auto-detect acceleration columns)

    Returns:
        Filtered DataFrame
    """
    if columns is None:
        # Auto-detect acceleration columns
        columns = [col for col in df.columns if col in ['x', 'y', 'z', 'g_x', 'g_y', 'g_z', 'x_mean', 'y_mean', 'z_mean']]

    df_filtered = df.copy()

    # Design low-pass Butterworth filter (order = 4)
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype="low")

    for col in columns:
        if col in df.columns:
            # Check if we have enough data points for filtering
            if len(df[col]) > 9:  # Minimum required for filtfilt
                df_filtered[f"{col}_lp"] = signal.filtfilt(b, a, df[col])
            else:
                # For short sequences, just copy the original data
                print(f"Warning: Column {col} has only {len(df[col])} samples, skipping low-pass filter")
                df_filtered[f"{col}_lp"] = df[col]

    return df_filtered
