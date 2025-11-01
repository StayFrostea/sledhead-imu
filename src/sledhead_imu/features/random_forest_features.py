"""Extract features for Random Forest classifier from IMU data."""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict, Optional


def extract_rf_features(
    df_run: pd.DataFrame,
    fs: float = 2000.0,
    accel_cols: Optional[Dict[str, str]] = None,
    gyro_cols: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Extract Random Forest features from a single run.
    
    Args:
        df_run: DataFrame with IMU data for one run (must have athlete_id, run_id)
        fs: Sampling rate in Hz
        accel_cols: Dict mapping {'x','y','z'} to column names. 
                    If None, tries g_x, g_y, g_z or x, y, z
        gyro_cols: Dict mapping {'x','y','z'} to column names.
                   If None, assumes gyro not available and skips gyro features
                   
    Returns:
        Dictionary of feature values
    """
    if df_run.empty:
        return {}
    
    # Default column mappings
    if accel_cols is None:
        if 'g_x' in df_run.columns:
            accel_cols = {'x': 'g_x', 'y': 'g_y', 'z': 'g_z'}
        elif 'x' in df_run.columns:
            accel_cols = {'x': 'x', 'y': 'y', 'z': 'z'}
        else:
            raise ValueError("No accelerometer columns found")
    
    if gyro_cols is None:
        if all(col in df_run.columns for col in ['gx', 'gy', 'gz']):
            gyro_cols = {'x': 'gx', 'y': 'gy', 'z': 'gz'}
        elif all(col in df_run.columns for col in ['gyrox', 'gyroy', 'gyroz']):
            # Raw gyro data - would need conversion, but skipping for now
            gyro_cols = None
        else:
            gyro_cols = None
    
    features = {}
    
    # Extract metadata
    if 'athlete_id' in df_run.columns:
        features['athlete_id'] = df_run['athlete_id'].iloc[0]
    if 'run_id' in df_run.columns:
        features['run_id'] = df_run['run_id'].iloc[0]
    if 'num_symptoms' in df_run.columns:
        # Get first non-null symptom value if present
        symptoms = df_run['num_symptoms'].dropna()
        if len(symptoms) > 0:
            features['num_symptoms'] = float(symptoms.iloc[0])
    
    # Get accelerometer data
    accel_x = df_run[accel_cols['x']].values
    accel_y = df_run[accel_cols['y']].values
    accel_z = df_run[accel_cols['z']].values
    
    # Compute resultant acceleration
    accel_res = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    
    # Basic stats for acceleration
    features['accel_mean'] = float(np.mean(accel_res))
    features['accel_std'] = float(np.std(accel_res))
    features['accel_min'] = float(np.min(accel_res))
    features['accel_max'] = float(np.max(accel_res))
    features['accel_range'] = features['accel_max'] - features['accel_min']
    
    # Gyroscope features (if available)
    if gyro_cols is not None:
        gyro_x = df_run[gyro_cols['x']].values
        gyro_y = df_run[gyro_cols['y']].values
        gyro_z = df_run[gyro_cols['z']].values
        gyro_res = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        
        features['gyro_mean'] = float(np.mean(gyro_res))
        features['gyro_std'] = float(np.std(gyro_res))
        features['gyro_max'] = float(np.max(gyro_res))
    else:
        features['gyro_mean'] = np.nan
        features['gyro_std'] = np.nan
        features['gyro_max'] = np.nan
    
    # Time step
    dt = 1.0 / fs
    
    # G-seconds style load metrics
    thresholds = [2.0, 3.0]
    for thresh in thresholds:
        time_above = np.sum(accel_res > thresh) * dt
        features[f'time_above_{thresh}g'] = float(time_above)
        
        # Cumulative load
        load_samples = np.maximum(accel_res - thresh, 0)
        g_seconds = np.sum(load_samples * dt)
        features[f'g_seconds_{thresh}g'] = float(g_seconds)
    
    # Peak/impact features
    features['num_peaks_over_3g'] = int(np.sum(accel_res > 3.0))
    features['num_peaks_over_4g'] = int(np.sum(accel_res > 4.0))
    
    # Find actual peaks above 2g using peak detection
    try:
        peaks, properties = signal.find_peaks(accel_res, height=2.0, distance=int(fs * 0.01))
        if len(peaks) > 0:
            features['highest_peak_g'] = float(np.max(accel_res[peaks]))
        else:
            features['highest_peak_g'] = 0.0
    except Exception:
        features['highest_peak_g'] = float(np.max(accel_res))
    
    # Longest continuous high-G duration
    above_2g = accel_res > 2.0
    if np.any(above_2g):
        # Find consecutive segments
        run_lengths = []
        current_run = 0
        for val in above_2g:
            if val:
                current_run += 1
            else:
                if current_run > 0:
                    run_lengths.append(current_run)
                    current_run = 0
        if current_run > 0:
            run_lengths.append(current_run)
        
        longest_run_samples = max(run_lengths) if run_lengths else 0
        features['longest_2g_duration'] = float(longest_run_samples * dt)
    else:
        features['longest_2g_duration'] = 0.0
    
    # Jerk (rate of change of acceleration)
    jerk = np.diff(accel_res) / dt
    features['jerk_mean'] = float(np.mean(np.abs(jerk)))
    features['jerk_max'] = float(np.max(np.abs(jerk)))
    
    # Frequency features
    try:
        # Use FFT to find dominant frequency
        fft_vals = np.fft.rfft(accel_res)
        fft_mag = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(len(accel_res), dt)
        
        # Find peak frequency (ignore DC component at index 0)
        if len(fft_mag) > 1:
            peak_idx = np.argmax(fft_mag[1:]) + 1
            features['dominant_freq'] = float(freqs[peak_idx])
        else:
            features['dominant_freq'] = np.nan
    except Exception:
        features['dominant_freq'] = np.nan
    
    # Run duration
    if 'timestamp' in df_run.columns:
        timestamps = pd.to_datetime(df_run['timestamp'])
        duration = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
        features['run_duration'] = float(duration)
    else:
        features['run_duration'] = float(len(accel_res) * dt)
    
    return features


def extract_all_runs(
    df: pd.DataFrame,
    group_cols: list = None,
    fs: float = 2000.0,
    **kwargs
) -> pd.DataFrame:
    """Extract features for all runs in a DataFrame.
    
    Args:
        df: DataFrame with IMU data
        group_cols: Columns to group by (default: ['athlete_id', 'run_id'])
        fs: Sampling rate in Hz
        **kwargs: Additional arguments passed to extract_rf_features
        
    Returns:
        DataFrame with one row per run and extracted features
    """
    if group_cols is None:
        if all(col in df.columns for col in ['athlete_id', 'run_id']):
            group_cols = ['athlete_id', 'run_id']
        elif 'run_id' in df.columns:
            group_cols = ['run_id']
        else:
            raise ValueError("Could not find suitable grouping columns")
    
    # Sort by timestamp if available
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Group and extract features
    all_features = []
    for group_key, group_df in df.groupby(group_cols):
        try:
            features = extract_rf_features(group_df, fs=fs, **kwargs)
            all_features.append(features)
        except Exception as e:
            print(f"Error processing group {group_key}: {e}")
            continue
    
    return pd.DataFrame(all_features)


def aggregate_rf_features_daily(
    df_rf_features: pd.DataFrame,
    date_col: str = 'date'
) -> pd.DataFrame:
    """Aggregate run-level RF features to daily per-athlete features.
    
    Args:
        df_rf_features: DataFrame with run-level RF features
        date_col: Column name for date (if not present, extracts from run_id)
        
    Returns:
        DataFrame with one row per athlete per day
        
    Aggregation strategy:
    - Sum: time_above_*g, g_seconds_*g, num_peaks_*g, longest_2g_duration, run_duration
    - Mean: accel_mean, accel_std, gyro_mean, gyro_std, jerk_mean, dominant_freq
    - Max: accel_max, accel_range, gyro_max, jerk_max, highest_peak_g
    - Min: accel_min
    """
    df = df_rf_features.copy()
    
    # Add date column if not present
    if date_col not in df.columns:
        # Try to extract date from run_id or add generic date
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df[date_col] = df['timestamp'].dt.date
        else:
            # Use a default date for grouping
            df[date_col] = '2025-01-01'
    
    # Define aggregation strategy
    sum_cols = [
        'time_above_2.0g', 'time_above_3.0g',
        'g_seconds_2.0g', 'g_seconds_3.0g',
        'num_peaks_over_3g', 'num_peaks_over_4g',
        'longest_2g_duration', 'run_duration'
    ]
    
    mean_cols = [
        'accel_mean', 'accel_std',
        'gyro_mean', 'gyro_std',
        'jerk_mean', 'dominant_freq'
    ]
    
    max_cols = [
        'accel_max', 'accel_range',
        'gyro_max', 'jerk_max',
        'highest_peak_g', 'num_symptoms'
    ]
    
    min_cols = ['accel_min']
    
    # Build aggregation dict
    agg_dict = {}
    
    # Sum columns
    for col in sum_cols:
        if col in df.columns:
            agg_dict[col] = 'sum'
    
    # Mean columns
    for col in mean_cols:
        if col in df.columns:
            agg_dict[col] = 'mean'
    
    # Max columns
    for col in max_cols:
        if col in df.columns:
            agg_dict[col] = 'max'
    
    # Min columns
    for col in min_cols:
        if col in df.columns:
            agg_dict[col] = 'min'
    
    # Group by athlete_id and date
    grouped = df.groupby(['athlete_id', date_col]).agg(agg_dict).reset_index()
    
    # Rename date column if it was added
    if 'date' != date_col:
        grouped = grouped.rename(columns={date_col: 'date'})
    
    return grouped

