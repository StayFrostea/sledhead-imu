"""Data schema definitions."""

from typing import Dict
import pandas as pd


# IMU data schema
IMU_SCHEMA = {
    "timestamp": "datetime64[ns]",
    "athlete_id": "object",
    "run_id": "object",
    "g_x": "float64",
    "g_y": "float64",
    "g_z": "float64",
    "g_mag": "float64",
}

# Symptom data schema
SYMPTOM_SCHEMA = {
    "timestamp": "datetime64[ns]",
    "athlete_id": "object",
    "symptom_type": "object",
    "severity": "int64",
    "duration_minutes": "float64",
}

# Model-ready data schema
MODEL_READY_SCHEMA = {
    "athlete_id": "object",
    "date": "datetime64[ns]",
    "exposure_s": "float64",
    "duration_s": "float64",
    "g_mag_mean": "float64",
    "g_mag_max": "float64",
    "g_mag_std": "float64",
    "sample_count": "int64",
}


def validate_schema(df: pd.DataFrame, schema: Dict[str, str]) -> bool:
    """Validate DataFrame against schema.

    Args:
        df: DataFrame to validate
        schema: Expected schema

    Returns:
        True if schema matches
    """
    for col, expected_dtype in schema.items():
        if col not in df.columns:
            return False
        if str(df[col].dtype) != expected_dtype:
            return False
    return True


def enforce_schema(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    """Enforce schema on DataFrame.

    Args:
        df: DataFrame to enforce schema on
        schema: Target schema

    Returns:
        DataFrame with enforced schema
    """
    df_schema = df.copy()

    for col, dtype in schema.items():
        if col in df_schema.columns:
            df_schema[col] = df_schema[col].astype(dtype)

    return df_schema
