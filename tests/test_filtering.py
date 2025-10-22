"""Test filtering functions."""

from sledhead_imu.filtering.high_pass import high_pass_filter
from sledhead_imu.filtering.low_pass import low_pass_filter
from sledhead_imu.filtering.rolling_avg import rolling_average_filter


def test_high_pass_filter(sample_imu_data):
    """Test high-pass filtering."""
    filtered = high_pass_filter(sample_imu_data, cutoff_freq=0.1)
    assert "g_x_hp" in filtered.columns
    assert "g_y_hp" in filtered.columns
    assert "g_z_hp" in filtered.columns


def test_low_pass_filter(sample_imu_data):
    """Test low-pass filtering."""
    filtered = low_pass_filter(sample_imu_data, cutoff_freq=10.0)
    assert "g_x_lp" in filtered.columns
    assert "g_y_lp" in filtered.columns
    assert "g_z_lp" in filtered.columns


def test_rolling_average_filter(sample_imu_data):
    """Test rolling average filtering."""
    filtered = rolling_average_filter(sample_imu_data, window_size=5)
    assert "g_x_rolling" in filtered.columns
    assert "g_y_rolling" in filtered.columns
    assert "g_z_rolling" in filtered.columns
