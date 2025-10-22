"""Comprehensive filtering pipeline for IMU data."""

import pandas as pd
from typing import Optional
from .high_pass import high_pass_filter
from .low_pass import low_pass_filter
from .median import median_filter
from .rolling_avg import savgol_filter


def apply_filtering_pipeline(
    df: pd.DataFrame,
    sampling_rate: float = 2000.0,
    columns: Optional[list] = None,
    hp_cutoff: float = 0.3,
    lp_cutoff: float = 30.0,
    median_kernel: int = 11,
    savgol_window: int = 51,
    savgol_polyorder: int = 3,
) -> pd.DataFrame:
    """Apply complete filtering pipeline in the correct order.
    
    Optimized for head-inclusive motion at 2000 Hz:
    1. High-pass Butterworth: 0.3 Hz, order = 2
    2. Low-pass Butterworth: 30 Hz, order = 4  
    3. Median filter: kernel = 11 samples (~5.5 ms)
    4. Savitzky-Golay: window = 51 samples (~25 ms), polyorder = 3
    
    All filters use filtfilt for zero-phase filtering.

    Args:
        df: IMU data
        sampling_rate: Sampling rate in Hz (default: 2000.0)
        columns: Columns to filter (default: auto-detect acceleration columns)
        hp_cutoff: High-pass cutoff frequency in Hz (default: 0.3)
        lp_cutoff: Low-pass cutoff frequency in Hz (default: 30.0)
        median_kernel: Median filter kernel size (default: 11)
        savgol_window: Savitzky-Golay window length (default: 51)
        savgol_polyorder: Savitzky-Golay polynomial order (default: 3)

    Returns:
        Filtered DataFrame with all processing steps applied
    """
    if columns is None:
        # Auto-detect acceleration columns
        columns = [col for col in df.columns if col in ['x', 'y', 'z', 'g_x', 'g_y', 'g_z', 'x_mean', 'y_mean', 'z_mean']]

    # Step 1: High-pass filter (0.3 Hz, order = 2)
    print("Step 1: Applying high-pass filter (0.3 Hz, order=2)...")
    df_hp = high_pass_filter(df, cutoff_freq=hp_cutoff, sampling_rate=sampling_rate, columns=columns)
    
    # Step 2: Low-pass filter (30 Hz, order = 4)
    print("Step 2: Applying low-pass filter (30 Hz, order=4)...")
    df_lp = low_pass_filter(df_hp, cutoff_freq=lp_cutoff, sampling_rate=sampling_rate, columns=columns)
    
    # Step 3: Median filter (kernel = 11)
    print("Step 3: Applying median filter (kernel=11)...")
    df_median = median_filter(df_lp, kernel_size=median_kernel, columns=columns)
    
    # Step 4: Savitzky-Golay filter (window=51, polyorder=3)
    print("Step 4: Applying Savitzky-Golay filter (window=51, polyorder=3)...")
    df_final = savgol_filter(df_median, window_length=savgol_window, polyorder=savgol_polyorder, columns=columns)
    
    print("Filtering pipeline complete!")
    return df_final
