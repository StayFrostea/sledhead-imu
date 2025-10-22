"""Aggregate data per day."""

import pandas as pd


def aggregate_daily_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate IMU data by athlete and day.

    Args:
        df: IMU data with athlete_id and timestamp columns

    Returns:
        Daily aggregated data
    """
    df = df.copy()
    df["date"] = df["timestamp"].dt.date

    # Determine which columns to use based on what's available
    mag_col = 'g_mag' if 'g_mag' in df.columns else 'r_gs'
    x_col = 'g_x' if 'g_x' in df.columns else 'x'
    y_col = 'g_y' if 'g_y' in df.columns else 'y'
    z_col = 'g_z' if 'g_z' in df.columns else 'z'
    
    # Group by athlete and date
    daily_agg = (
        df.groupby(["athlete_id", "date"])
        .agg(
            {mag_col: ["mean", "max", "std", "count"], 
             x_col: "mean", 
             y_col: "mean", 
             z_col: "mean"}
        )
        .reset_index()
    )

    # Flatten column names
    daily_agg.columns = [
        "athlete_id",
        "date",
        f"{mag_col}_mean",
        f"{mag_col}_max",
        f"{mag_col}_std",
        "sample_count",
        f"{x_col}_mean",
        f"{y_col}_mean",
        f"{z_col}_mean",
    ]

    return daily_agg
