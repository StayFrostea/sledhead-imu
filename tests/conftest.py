"""Test configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_imu_data():
    """Sample IMU data for testing."""
    dates = pd.date_range("2025-01-01", periods=100, freq="1s")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "athlete_id": ["A001"] * 100,
            "run_id": ["R001"] * 100,
            "g_x": np.random.normal(0, 1, 100),
            "g_y": np.random.normal(0, 1, 100),
            "g_z": np.random.normal(9.8, 1, 100),
            "g_mag": np.random.normal(10, 1, 100),
        }
    )


@pytest.fixture
def sample_symptom_data():
    """Sample symptom data for testing."""
    dates = pd.date_range("2025-01-01", periods=10, freq="1H")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "athlete_id": ["A001"] * 10,
            "symptom_type": ["headache"] * 10,
            "severity": np.random.randint(1, 5, 10),
            "duration_minutes": np.random.uniform(10, 60, 10),
        }
    )
