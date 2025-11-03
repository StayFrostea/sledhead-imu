"""Normalize and clean IMU data."""

import pandas as pd
from typing import Dict, Any


def normalize_imu_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Normalize IMU data to unified schema.
    
    Args:
        df: Raw IMU data
        config: Normalization configuration
        
    Returns:
        Normalized DataFrame
    """
    df = df.copy()
    
    # Handle acceleration data - prioritize x/y/z (already in g) over accx/accy/accz (m/s²)
    # First, normalize column names (handle 'x acc', 'y acc', etc.)
    if 'x acc' in df.columns and 'y acc' in df.columns and 'z acc' in df.columns:
        df = df.rename(columns={'x acc': 'x', 'y acc': 'y', 'z acc': 'z'})
    
    if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
        # Already in g units
        pass
    elif 'accx' in df.columns or 'acc_x' in df.columns:
        # Check if we have both or just one
        if 'accx' in df.columns:
            df = df.rename(columns={'accx': 'acc_x', 'accy': 'acc_y', 'accz': 'acc_z'})
        
        if 'acc_x' in df.columns and 'x' not in df.columns:
            # Only have m/s² data, convert to g
            # Standard gravity: 9.80665 m/s²
            df['x'] = df['acc_x'] / 9.80665
            df['y'] = df['acc_y'] / 9.80665  
            df['z'] = df['acc_z'] / 9.80665
    else:
        raise ValueError("No acceleration data found. Expected 'accx/accy/accz' or 'x/y/z'")
    
    # Handle timestamp
    if 'timestamp' not in df.columns:
        # Create timestamp from 't' column if available
        if 't' in df.columns:
            # Start from an arbitrary date, use t as seconds offset
            base_date = pd.Timestamp('2025-01-01 00:00:00')
            df['timestamp'] = base_date + pd.to_timedelta(df['t'], unit='s')
        else:
            raise ValueError("No timestamp or 't' column found")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Rename columns to standard format
    df = df.rename(columns={'x': 'g_x', 'y': 'g_y', 'z': 'g_z'})
    
    # Calculate magnitude if not already present
    if 'g_mag' not in df.columns:
        df['g_mag'] = (df['g_x']**2 + df['g_y']**2 + df['g_z']**2)**0.5
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df
