"""Test data preparation and splitting functions."""

import pandas as pd
import numpy as np
from pathlib import Path
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


def test_train_val_test_split_roundtrip(tmp_path):
    """Test 60/20/20 train/val/test split with save/load roundtrip."""
    # Create minimal dataset with 10 samples
    np.random.seed(42)
    n_samples = 10
    
    # Create feature columns matching RF features
    X = pd.DataFrame({
        'time_above_2.0g': np.random.exponential(10, n_samples),
        'time_above_3.0g': np.random.exponential(3, n_samples),
        'g_seconds_2.0g': np.random.exponential(15, n_samples),
        'g_seconds_3.0g': np.random.exponential(5, n_samples),
        'num_peaks_over_3g': np.random.poisson(5, n_samples),
        'num_peaks_over_4g': np.random.poisson(2, n_samples),
        'longest_2g_duration': np.random.exponential(0.1, n_samples),
        'run_duration': np.random.uniform(60, 180, n_samples),
        'accel_mean': np.random.uniform(1, 3, n_samples),
        'accel_std': np.random.uniform(0.5, 2, n_samples),
        'gyro_mean': np.random.uniform(0, 50, n_samples),
        'gyro_std': np.random.uniform(0, 20, n_samples),
        'jerk_mean': np.random.uniform(10, 100, n_samples),
        'dominant_freq': np.random.uniform(0.01, 0.5, n_samples),
        'accel_max': np.random.uniform(2, 10, n_samples),
        'accel_range': np.random.uniform(1, 8, n_samples),
        'gyro_max': np.random.uniform(20, 200, n_samples),
        'jerk_max': np.random.uniform(100, 2000, n_samples),
        'highest_peak_g': np.random.uniform(2, 9, n_samples),
        'num_symptoms': np.random.randint(0, 5, n_samples),
        'accel_min': np.random.uniform(0, 1, n_samples),
    })
    
    # Create labels with severity 0-5
    y = pd.DataFrame({
        'severity': np.random.choice([0, 1, 2, 3, 4, 5], n_samples)
    })
    
    # First split: train+val vs test (80/20)
    X_train_val, X_test, y_train_val, y_test = split_train_test(
        X, y, test_size=0.2, random_state=42
    )
    
    # Second split: train vs val (75/25 of remaining 80%)
    X_train, X_val, y_train, y_val = split_train_test(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )
    
    # Verify split sizes (60/20/20 with 10 samples)
    assert len(X_train) + len(X_val) + len(X_test) == n_samples
    assert len(X_train) == 6  # 60% of 10
    assert len(X_val) == 2    # 20% of 10
    assert len(X_test) == 2   # 20% of 10
    
    # Verify feature count
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1] == 21
    assert all(X_train.columns == X_val.columns)
    assert all(X_train.columns == X_test.columns)
    
    # Verify labels
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert len(y_test) == len(X_test)
    
    # Save splits
    splits_dir = tmp_path / 'splits'
    train_dir = splits_dir / 'train'
    val_dir = splits_dir / 'val'
    test_dir = splits_dir / 'test'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(train_dir / 'X_train.csv', index=False)
    y_train.to_csv(train_dir / 'y_train.csv', index=False)
    X_val.to_csv(val_dir / 'X_val.csv', index=False)
    y_val.to_csv(val_dir / 'y_val.csv', index=False)
    X_test.to_csv(test_dir / 'X_test.csv', index=False)
    y_test.to_csv(test_dir / 'y_test.csv', index=False)
    
    # Verify files exist
    assert (train_dir / 'X_train.csv').exists()
    assert (train_dir / 'y_train.csv').exists()
    assert (val_dir / 'X_val.csv').exists()
    assert (val_dir / 'y_val.csv').exists()
    assert (test_dir / 'X_test.csv').exists()
    assert (test_dir / 'y_test.csv').exists()
    
    # Load and verify roundtrip
    X_train_loaded = pd.read_csv(train_dir / 'X_train.csv')
    y_train_loaded = pd.read_csv(train_dir / 'y_train.csv')
    X_val_loaded = pd.read_csv(val_dir / 'X_val.csv')
    y_val_loaded = pd.read_csv(val_dir / 'y_val.csv')
    X_test_loaded = pd.read_csv(test_dir / 'X_test.csv')
    y_test_loaded = pd.read_csv(test_dir / 'y_test.csv')
    
    # Verify data integrity (reset index for comparison since CSV loads with RangeIndex)
    pd.testing.assert_frame_equal(X_train.reset_index(drop=True), X_train_loaded.reset_index(drop=True))
    pd.testing.assert_frame_equal(y_train.reset_index(drop=True), y_train_loaded.reset_index(drop=True))
    pd.testing.assert_frame_equal(X_val.reset_index(drop=True), X_val_loaded.reset_index(drop=True))
    pd.testing.assert_frame_equal(y_val.reset_index(drop=True), y_val_loaded.reset_index(drop=True))
    pd.testing.assert_frame_equal(X_test.reset_index(drop=True), X_test_loaded.reset_index(drop=True))
    pd.testing.assert_frame_equal(y_test.reset_index(drop=True), y_test_loaded.reset_index(drop=True))
    
    # Verify label distribution
    train_severities = y_train_loaded['severity'].values
    val_severities = y_val_loaded['severity'].values
    test_severities = y_test_loaded['severity'].values
    
    # All severities from original should be preserved across splits
    original_severities = y['severity'].values
    all_split_severities = np.concatenate([train_severities, val_severities, test_severities])
    assert set(original_severities) == set(all_split_severities)
