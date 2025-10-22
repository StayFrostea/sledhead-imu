"""Save artifacts and results."""

import pandas as pd
import pickle
from pathlib import Path
from typing import Union, Any


def save_dataframe(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """Save DataFrame to file.

    Args:
        df: DataFrame to save
        file_path: Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.suffix == ".parquet":
        df.to_parquet(file_path, index=False)
    elif file_path.suffix == ".csv":
        df.to_csv(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_model(model: Any, file_path: Union[str, Path]) -> None:
    """Save trained model.

    Args:
        model: Trained model object
        file_path: Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(model, f)
