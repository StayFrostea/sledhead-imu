"""Load symptom data from various formats."""

import pandas as pd
from pathlib import Path
from typing import Union


def load_symptom_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load symptom data from file.

    Args:
        file_path: Path to symptom data file

    Returns:
        DataFrame with symptom data
    """
    file_path = Path(file_path)

    if file_path.suffix == ".parquet":
        return pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
