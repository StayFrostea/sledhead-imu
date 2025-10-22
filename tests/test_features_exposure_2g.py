import pandas as pd
from sledhead_imu.features.exposure_2g import compute_exposure


def test_exposure_runs():
    df = pd.DataFrame(
        {
            "athlete_id": [1, 1, 1],
            "run_id": [10, 10, 10],
            "timestamp": pd.to_datetime(
                ["2025-01-01T00:00:00", "2025-01-01T00:00:01", "2025-01-01T00:00:02"]
            ),
            "g": [1.5, 2.2, 2.5],
        }
    )
    res = compute_exposure(df, "g", 2.0)
    assert not res.empty
