"""Attach symptom labels to IMU data."""

import pandas as pd


def attach_labels(imu_df: pd.DataFrame, symptoms_df: pd.DataFrame) -> pd.DataFrame:
    """Attach symptom labels to IMU data.

    Args:
        imu_df: IMU data with athlete_id and timestamp
        symptoms_df: Symptom data with athlete_id, timestamp, and symptom flags

    Returns:
        Merged DataFrame with labels
    """
    # Check if timestamp columns exist
    if "timestamp" not in imu_df.columns or "timestamp" not in symptoms_df.columns:
        print("Warning: No timestamp columns found. Merging on athlete_id only.")
        # Simple merge on athlete_id
        merged = pd.merge(imu_df, symptoms_df, on="athlete_id", how="left", suffixes=("", "_symptom"))
        return merged
    
    # Ensure timestamp columns are datetime
    imu_df = imu_df.copy()
    symptoms_df = symptoms_df.copy()
    imu_df["timestamp"] = pd.to_datetime(imu_df["timestamp"])
    symptoms_df["timestamp"] = pd.to_datetime(symptoms_df["timestamp"])
    
    # Merge on athlete_id and closest timestamp
    merged = pd.merge_asof(
        imu_df.sort_values("timestamp"),
        symptoms_df.sort_values("timestamp"),
        on="timestamp",
        by="athlete_id",
        direction="nearest",
        tolerance=pd.Timedelta("1D"),  # Within 1 day
    )

    return merged
