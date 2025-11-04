#!/usr/bin/env python3
"""Full pipeline script to process real data and train model."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sledhead_imu.io.load_imu import load_imu_data
from sledhead_imu.ingest.normalize import normalize_imu_data
from sledhead_imu.features.random_forest_features import extract_all_runs
from sledhead_imu.models.random_forest import train_random_forest, evaluate_random_forest
from sklearn.model_selection import train_test_split

print("="*80)
print("PROCESSING REAL DATA & TRAINING MODEL")
print("="*80)

# Step 1: Load all files
data_dir = Path('data/00_collect/imu_raw')
imu_files = list(data_dir.glob('*.csv'))
print(f"\nFound {len(imu_files)} files")

all_runs = []
for imu_file in imu_files:
    print(f"\nProcessing: {imu_file.name}")
    
    df_raw = load_imu_data(imu_file)
    
    # Extract athlete and run from filename
    match = re.search(r'New-(\d+)-(\d+)', imu_file.name)
    if match:
        athlete_id = f"A{match.group(1)}"
        run_id = f"R{match.group(2)}"
        df_raw['athlete_id'] = athlete_id
        df_raw['run_id'] = run_id
        print(f"  → athlete={athlete_id}, run={run_id}")
    
    # Normalize
    df_norm = normalize_imu_data(df_raw, {})
    all_runs.append(df_norm)
    print(f"  → {len(df_norm):,} samples")

# Combine all runs
df_all = pd.concat(all_runs, ignore_index=True)
print(f"\n✓ Total: {len(df_all):,} samples from {df_all['run_id'].nunique()} runs")

# Step 2: Extract features
print("\n" + "="*80)
print("EXTRACTING FEATURES")
print("="*80)

features_df = extract_all_runs(df_all, fs=2000.0)
print(f"✓ Extracted {len(features_df)} run-level features")

# Show feature stats
print(f"\nFeature columns: {list(features_df.columns)[:10]}...")

# Step 3: Create labels
print("\n" + "="*80)
print("CREATING LABELS")
print("="*80)

# Check if we have actual symptom data
if 'num_symptoms' in features_df.columns:
    num_symptoms = features_df['num_symptoms']
    print(f"\nFound num_symptoms column")
    print(f"Num symptoms range: {num_symptoms.min():.0f} to {num_symptoms.max():.0f}")
    
    # Binary classification: 0-1 symptoms = 0, 2+ symptoms = 1
    y = (num_symptoms >= 2).astype(int)
    
    # Check if we have both classes
    unique_labels = y.unique()
    if len(unique_labels) < 2:
        # All runs have same label, need fallback
        print(f"⚠️  All runs have {(y == 0).sum()} symptom(s). Using synthetic labels for binary classification.")
        highest_peak = features_df['highest_peak_g']
        median_peak = highest_peak.median()
        y = (highest_peak > median_peak).astype(int)
        print(f"\nBinary label distribution (synthetic based on peak G):")
        print(y.value_counts().sort_index())
        print(f"\n  Label 0 (≤{median_peak:.2f}g): {(y == 0).sum()} runs")
        print(f"  Label 1 (>{median_peak:.2f}g): {(y == 1).sum()} runs")
    else:
        print(f"\nBinary label distribution:")
        print(y.value_counts().sort_index())
        print(f"\n  Label 0 (0-1 symptoms): {(y == 0).sum()} runs")
        print(f"  Label 1 (2+ symptoms): {(y == 1).sum()} runs")
else:
    # Fallback: Use highest_peak_g to create synthetic binary labels
    highest_peak = features_df['highest_peak_g']
    print(f"⚠️  No num_symptoms found. Using synthetic labels from peak G.")
    print(f"Peak G range: {highest_peak.min():.2f}g to {highest_peak.max():.2f}g")
    
    # Binary: 0 = lower G forces, 1 = higher G forces
    median_peak = highest_peak.median()
    y = (highest_peak > median_peak).astype(int)
    
    print(f"\nBinary label distribution (synthetic):")
    print(y.value_counts().sort_index())
    print(f"\n  Label 0 (≤{median_peak:.2f}g): {(y == 0).sum()} runs")
    print(f"  Label 1 (>{median_peak:.2f}g): {(y == 1).sum()} runs")

# Step 4: Prepare data
X = features_df.drop(columns=['athlete_id', 'run_id', 'num_symptoms'], errors='ignore')

print(f"\n✓ X: {X.shape}, y: {y.shape}")

# Step 5: Train/val/test split (60/20/20)
if len(X) > 3:
    try:
        # First split: train+val vs test (80/20)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # Second split: train vs val (75/25 of remaining 80% = 60/20 overall)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
        )
        print(f"\n✓ Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    except ValueError:
        # Not enough samples per class for stratification
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42
        )
        print(f"\n✓ Split (no stratification): Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Save splits to 09_splits/ for evaluation scripts
    splits_dir = Path('data/09_splits')
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, X_split, y_split in [
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test', X_test, y_test)
    ]:
        split_dir = splits_dir / split_name
        split_dir.mkdir(exist_ok=True)
        X_split.to_csv(split_dir / f'X_{split_name}.csv', index=False)
        # Convert y to DataFrame with column name
        if isinstance(y_split, pd.Series):
            y_df = pd.DataFrame({y_split.name if y_split.name else 'label': y_split})
        else:
            y_df = y_split
        y_df.to_csv(split_dir / f'y_{split_name}.csv', index=False)
    print(f"✓ Saved splits to data/09_splits/")
else:
    print("⚠️  Not enough samples for train/test split. Using all for training.")
    X_train, X_val, X_test = X, X, X
    y_train, y_val, y_test = y, y, y

# Step 6: Train
print("\n" + "="*80)
print("TRAINING MODEL")
print("="*80)

config = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 2,
    'class_weight': 'balanced',
    'random_state': 42
}

model = train_random_forest(X_train, y_train, X_val, y_val, config)
print("✓ Model trained")

# Step 7: Evaluate
train_results = evaluate_random_forest(model, X_train, y_train)
val_results = evaluate_random_forest(model, X_val, y_val)
test_results = evaluate_random_forest(model, X_test, y_test)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Train Accuracy: {train_results['accuracy']:.3f}")
print(f"Val Accuracy: {val_results['accuracy']:.3f}")
print(f"Test Accuracy: {test_results['accuracy']:.3f}")

print(f"\nTop Features:")
print(test_results['feature_importance'].head(10).to_string(index=False))

# Save model
Path('data/10_models/rf').mkdir(parents=True, exist_ok=True)
import pickle
with open('data/10_models/rf/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\n✓ Model saved to data/10_models/rf/model.pkl")

print("\n✅ Pipeline complete!")
print("="*80)

