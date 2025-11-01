"""Tests for Random Forest feature extraction."""

import numpy as np
import pandas as pd
import pytest

from sledhead_imu.features.random_forest_features import (
    extract_rf_features,
    extract_all_runs
)


@pytest.fixture
def sample_imu_run():
    """Create sample IMU data for one run."""
    n_samples = 1000
    
    # Generate synthetic IMU data
    np.random.seed(42)
    accel_x = np.random.normal(0, 0.1, n_samples)
    accel_y = np.random.normal(0, 0.1, n_samples)
    accel_z = np.random.normal(-1.0, 0.1, n_samples)  # Mostly in -z direction
    
    # Add some high-g spikes
    spike_indices = [100, 200, 500, 700]
    for idx in spike_indices:
        accel_x[idx] += 2.0
        accel_y[idx] += 3.0
        accel_z[idx] += 4.0
    
    gyro_x = np.random.normal(0, 0.5, n_samples)
    gyro_y = np.random.normal(0, 0.5, n_samples)
    gyro_z = np.random.normal(0, 0.5, n_samples)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='500us'),
        'athlete_id': 'A001',
        'run_id': 'R001',
        'g_x': accel_x,
        'g_y': accel_y,
        'g_z': accel_z,
        'gx': gyro_x,
        'gy': gyro_y,
        'gz': gyro_z,
    })
    
    return df


def test_extract_rf_features_basic(sample_imu_run):
    """Test basic feature extraction."""
    features = extract_rf_features(sample_imu_run, fs=2000.0)
    
    assert 'athlete_id' in features
    assert 'run_id' in features
    assert 'accel_mean' in features
    assert 'accel_std' in features
    assert 'accel_max' in features
    assert 'accel_range' in features
    assert 'gyro_mean' in features
    assert 'gyro_max' in features
    assert 'time_above_2.0g' in features
    assert 'time_above_3.0g' in features
    assert 'g_seconds_2.0g' in features
    assert 'g_seconds_3.0g' in features
    assert 'num_peaks_over_3g' in features
    assert 'num_peaks_over_4g' in features
    assert 'highest_peak_g' in features
    assert 'longest_2g_duration' in features
    assert 'jerk_mean' in features
    assert 'jerk_max' in features
    assert 'dominant_freq' in features
    assert 'run_duration' in features
    
    # Check reasonable values
    assert features['accel_max'] > 0
    assert features['highest_peak_g'] > 0
    assert features['run_duration'] > 0
    assert features['accel_mean'] >= 0
    assert features['accel_std'] >= 0


def test_extract_rf_features_empty():
    """Test feature extraction with empty DataFrame."""
    df_empty = pd.DataFrame(columns=['athlete_id', 'run_id', 'g_x', 'g_y', 'g_z'])
    features = extract_rf_features(df_empty, fs=2000.0)
    
    assert features == {}


def test_extract_all_runs(sample_imu_run):
    """Test extracting features for multiple runs."""
    # Create data for multiple runs
    df2 = sample_imu_run.copy()
    df2['run_id'] = 'R002'
    df2['athlete_id'] = 'A002'
    
    df_all = pd.concat([sample_imu_run, df2], ignore_index=True)
    
    result = extract_all_runs(df_all, fs=2000.0)
    
    assert len(result) == 2
    assert set(result['run_id']) == {'R001', 'R002'}
    assert set(result['athlete_id']) == {'A001', 'A002'}
    
    # Check all features are present
    expected_features = [
        'athlete_id', 'run_id', 'accel_mean', 'accel_std', 'accel_min',
        'accel_max', 'accel_range', 'gyro_mean', 'gyro_std', 'gyro_max',
        'time_above_2.0g', 'time_above_3.0g', 'g_seconds_2.0g', 'g_seconds_3.0g',
        'num_peaks_over_3g', 'num_peaks_over_4g', 'highest_peak_g',
        'longest_2g_duration', 'jerk_mean', 'jerk_max', 'dominant_freq',
        'run_duration'
    ]
    assert all(col in result.columns for col in expected_features)


def test_extract_rf_features_custom_columns():
    """Test feature extraction with custom column names."""
    n_samples = 500
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='500us'),
        'athlete_id': 'A003',
        'run_id': 'R003',
        'x': np.random.normal(0, 0.5, n_samples),
        'y': np.random.normal(0, 0.5, n_samples),
        'z': np.random.normal(-1.0, 0.5, n_samples),
    })
    
    features = extract_rf_features(df, fs=2000.0)
    
    assert 'athlete_id' in features
    assert 'run_id' in features
    assert 'accel_mean' in features


def test_features_numeric():
    """Test that all numeric features are valid."""
    n_samples = 1000
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='500us'),
        'athlete_id': 'A004',
        'run_id': 'R004',
        'g_x': np.random.normal(0, 0.5, n_samples),
        'g_y': np.random.normal(0, 0.5, n_samples),
        'g_z': np.random.normal(-1.0, 0.5, n_samples),
        'gx': np.random.normal(0, 0.5, n_samples),
        'gy': np.random.normal(0, 0.5, n_samples),
        'gz': np.random.normal(0, 0.5, n_samples),
    })
    
    features = extract_rf_features(df, fs=2000.0)
    
    numeric_features = [k for k, v in features.items() 
                       if k not in ['athlete_id', 'run_id']]
    
    for feat_name in numeric_features:
        val = features[feat_name]
        assert isinstance(val, (int, float, np.number))
        assert not np.isinf(val)
        assert not np.isnan(val) or feat_name == 'gyro_mean'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

