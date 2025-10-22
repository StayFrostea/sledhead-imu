"""Test data preparation and splitting functions."""

import pandas as pd
import numpy as np
from sledhead_imu.prep.build_model_ready import build_model_ready_data
from sledhead_imu.prep.split_train_test import split_train_test


def test_build_model_ready_data(sample_imu_data):
    """Test model-ready data building."""
    # Add some feature columns
    sample_imu_data["feature1"] = np.random.randn(len(sample_imu_data))
    sample_imu_data["feature2"] = np.random.randn(len(sample_imu_data))
    sample_imu_data["label"] = np.random.randint(0, 2, len(sample_imu_data))

    X, y = build_model_ready_data(sample_imu_data, ["feature1", "feature2"], "label")

    assert len(X) == len(y)
    assert X.shape[1] == 2
    assert y.name == "label"


def test_split_train_test():
    """Test train/test splitting."""
    X = pd.DataFrame({"feature1": np.random.randn(100), "feature2": np.random.randn(100)})
    y = pd.Series(np.random.randint(0, 2, 100), name="label")

    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert len(X_test) == 20  # 20% of 100
