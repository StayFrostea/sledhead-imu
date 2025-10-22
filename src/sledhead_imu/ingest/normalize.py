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
    
    # Ensure required columns exist
    required_cols = ['timestamp', 'x', 'y', 'z']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Rename columns to standard format
    df = df.rename(columns={'x': 'g_x', 'y': 'g_y', 'z': 'g_z'})
    
    # Calculate magnitude
    df['g_mag'] = (df['g_x']**2 + df['g_y']**2 + df['g_z']**2)**0.5
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df
