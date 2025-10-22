import pandas as pd
from pathlib import Path

def compute_exposure(df: pd.DataFrame, g_col: str, threshold: float = 2.0) -> pd.DataFrame:
    """
    Compute ≥2 g exposure 'dose' per run/day.
    Expects columns: ['athlete_id','run_id','timestamp', g_col].
    """
    df = df.copy()
    df["above"] = df[g_col] >= threshold
    df["dt"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
    df["dose"] = df["dt"].where(df["above"], 0.0) * df[g_col].where(df["above"], 0.0)
    # Calculate exposure and duration for each group
    exposure_data = []
    for (athlete_id, run_id), group in df.groupby(["athlete_id", "run_id"]):
        exposure_s = group["dose"].sum()
        duration_s = group[group["above"]]["dt"].sum()
        exposure_data.append({
            "athlete_id": athlete_id,
            "run_id": run_id, 
            "exposure_s": exposure_s,
            "duration_s": duration_s
        })
    
    agg = pd.DataFrame(exposure_data)
    return agg
