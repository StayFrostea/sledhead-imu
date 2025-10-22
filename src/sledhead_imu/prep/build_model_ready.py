"""Build model-ready dataset."""

import pandas as pd
from typing import Tuple, List


def build_model_ready_data(
    df: pd.DataFrame, feature_cols: List[str], label_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build model-ready features and labels.

    Args:
        df: Processed IMU data with features and labels
        feature_cols: List of feature column names
        label_col: Label column name

    Returns:
        Tuple of (features_df, labels_series)
    """
    # Select features
    X = df[feature_cols].copy()

    # Select labels
    y = df[label_col].copy()

    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(0)  # Assuming 0 for missing labels

    return X, y
