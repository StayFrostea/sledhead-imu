#!/usr/bin/env python
"""
Random Forest Feature Extraction Demo

This script demonstrates how to extract Random Forest features from IMU data
and prepare them for machine learning classification.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sledhead_imu.features import extract_all_runs
from sledhead_imu.io.load_imu import load_imu_data


def main():
    """Demonstrate Random Forest feature extraction."""
    print("=" * 80)
    print("Random Forest Feature Extraction Demo")
    print("=" * 80)
    
    # Load sample data
    data_dir = Path(__file__).parent.parent / "data"
    collect_dir = data_dir / "00_collect" / "imu"
    
    imu_files = list(collect_dir.glob("sample_imu_A*.csv"))
    
    # For demo purposes, always use synthetic data to show variety
    use_synthetic = True
    
    if not imu_files or use_synthetic:
        print("Using synthetic data for demonstration...")
        
        # Generate synthetic data for two runs
        df_list = []
        
        # Run 1: Moderate-g baseline with some impacts
        np.random.seed(42)
        n_samples = 2000
        timestamps = pd.date_range("2025-01-01", periods=n_samples, freq="500us")
        
        accel_base = np.array([0.0, 0.0, -1.0])
        accel_data = np.tile(accel_base, (n_samples, 1)) + np.random.normal(0, 0.2, (n_samples, 3))
        
        # Add impacts ranging from low to moderate
        spike_configs1 = [
            (300, [2.0, 0.5, 0.0]),    # ~3.0g total
            (800, [2.5, 2.0, 0.0]),    # ~4.5g total  
            (1200, [0.0, 0.0, 5.0]),   # ~6.0g total
            (1700, [3.5, 1.5, 1.0]),   # ~6.0g total
        ]
        for idx, spike in spike_configs1:
            accel_data[idx] += np.array(spike)
        
        df1 = pd.DataFrame({
            "timestamp": timestamps,
            "athlete_id": "A001",
            "run_id": "R001",
            "g_x": accel_data[:, 0],
            "g_y": accel_data[:, 1],
            "g_z": accel_data[:, 2],
        })
        df_list.append(df1)
        
        # Run 2: Higher-g impacts in 2-6g range
        np.random.seed(123)
        timestamps2 = pd.date_range("2025-01-01", periods=n_samples, freq="500us")
        accel_data2 = np.tile(accel_base, (n_samples, 1)) + np.random.normal(0, 0.3, (n_samples, 3))
        
        # Add various high-g impacts from 0.5g to 6g
        spike_configs = [
            (200, [2.0, 0.0, 0.0]),     # ~2.8g impact
            (600, [5.0, 3.5, 0.0]),     # ~8.5g impact
            (1000, [0.0, 0.0, 7.0]),    # ~6.0g impact
            (1400, [5.0, 4.5, 2.5]),    # ~8.9g impact
            (1800, [6.0, 2.5, 1.0]),    # ~8.0g impact
        ]
        for idx, spike in spike_configs:
            accel_data2[idx] += np.array(spike)
        
        df2 = pd.DataFrame({
            "timestamp": timestamps2,
            "athlete_id": "A002",
            "run_id": "R002",
            "g_x": accel_data2[:, 0],
            "g_y": accel_data2[:, 1],
            "g_z": accel_data2[:, 2],
        })
        df_list.append(df2)
        
        df_raw = pd.concat(df_list, ignore_index=True)
        print(f"Generated synthetic data with {len(df_raw)} samples across {len(df_list)} runs")
    else:
        print(f"Found {len(imu_files)} IMU files")
        
        # Load first file
        df_raw = load_imu_data(imu_files[0])
        print(f"Loaded data with {len(df_raw)} samples")
    
    print("\n" + "-" * 80)
    print("1. Extracting features for all runs")
    print("-" * 80)
    
    # Extract features
    features_df = extract_all_runs(df_raw, fs=2000.0)
    
    print(f"\nExtracted {len(features_df)} runs with {len(features_df.columns)} features")
    print("\nFeature columns:")
    for col in features_df.columns:
        print(f"  - {col}")
    
    print("\n" + "-" * 80)
    print("2. Feature summary statistics")
    print("-" * 80)
    
    # Display summary
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    print(features_df[numeric_cols].describe())
    
    print("\n" + "-" * 80)
    print("3. Feature values by run")
    print("-" * 80)
    
    # Show comparison for key features
    key_features = ['accel_mean', 'accel_max', 'highest_peak_g', 'time_above_2.0g', 
                    'g_seconds_2.0g', 'num_peaks_over_3g', 'jerk_max']
    
    for idx, row in features_df.iterrows():
        print(f"\nRun: {row['athlete_id']} - {row['run_id']}")
        for feat in key_features:
            if feat in features_df.columns:
                print(f"  {feat}: {row[feat]:.4f}")
    
    print("\n" + "-" * 80)
    print("4. Ready for machine learning")
    print("-" * 80)
    
    print("\nTo use with sklearn RandomForestClassifier:")
    print("""
    from sklearn.ensemble import RandomForestClassifier
    
    # Prepare data
    X = features_df.drop(['athlete_id', 'run_id'], axis=1)
    y = your_symptom_labels  # Binary labels
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)
    
    # Evaluate
    print('Accuracy: {:.3f}'.format(rf.score(X, y)))
    """)
    
    # Check for high-g events
    print("\n" + "-" * 80)
    print("5. High-g event summary")
    print("-" * 80)
    
    if "highest_peak_g" in features_df.columns:
        max_peak = features_df["highest_peak_g"].max()
        print(f"  Highest peak across all runs: {max_peak:.2f} g")
    
    if "time_above_2.0g" in features_df.columns:
        total_time_2g = features_df["time_above_2.0g"].sum()
        print(f"  Total time above 2.0g: {total_time_2g:.3f} seconds")
    
    if "num_peaks_over_4g" in features_df.columns:
        total_peaks_4g = features_df["num_peaks_over_4g"].sum()
        print(f"  Total peaks over 4.0g: {total_peaks_4g}")
    
    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

