"""Test segmentation functions."""

from sledhead_imu.segmentation.detect_trim_runs import detect_run_segments, trim_run_segment
from sledhead_imu.segmentation.segment_by_athlete import segment_by_athlete
from sledhead_imu.segmentation.aggregate_per_day import aggregate_daily_data


def test_detect_run_segments(sample_imu_data):
    """Test run segment detection."""
    segments = detect_run_segments(sample_imu_data, threshold=1.5)
    assert isinstance(segments, list)
    assert all(isinstance(seg, tuple) and len(seg) == 2 for seg in segments)


def test_trim_run_segment(sample_imu_data):
    """Test run segment trimming."""
    segment = trim_run_segment(sample_imu_data, 0, 10)
    assert len(segment) == 10
    assert segment.index[0] == 0


def test_segment_by_athlete(sample_imu_data):
    """Test athlete segmentation."""
    segments = segment_by_athlete(sample_imu_data)
    assert "A001" in segments
    assert len(segments["A001"]) == len(sample_imu_data)


def test_aggregate_daily_data(sample_imu_data):
    """Test daily aggregation."""
    daily = aggregate_daily_data(sample_imu_data)
    assert "athlete_id" in daily.columns
    assert "date" in daily.columns
    assert "g_mag_mean" in daily.columns
