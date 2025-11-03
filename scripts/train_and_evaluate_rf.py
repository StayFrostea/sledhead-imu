#!/usr/bin/env python3
"""Train Random Forest model and show comprehensive evaluation metrics with visualizations."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import pickle
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sledhead_imu.models.random_forest import (
    train_random_forest,
    evaluate_random_forest
)
from sledhead_imu.validate.validate_cutoffs import (
    validate_model_performance,
    validate_per_class_performance,
    get_confusion_matrix_report
)
from sledhead_imu.features.random_forest_features import extract_all_runs, aggregate_rf_features_daily

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("="*80)
print("RANDOM FOREST TRAINING & EVALUATION WITH VISUALIZATIONS")
print("="*80)

# Load data from splits
data_dir = Path('data')
X_train = pd.read_csv(data_dir / '09_splits' / 'train' / 'X_train.csv')
y_train = pd.read_csv(data_dir / '09_splits' / 'train' / 'y_train.csv')
X_val = pd.read_csv(data_dir / '09_splits' / 'val' / 'X_val.csv')
y_val = pd.read_csv(data_dir / '09_splits' / 'val' / 'y_val.csv')
X_test = pd.read_csv(data_dir / '09_splits' / 'test' / 'X_test.csv')
y_test = pd.read_csv(data_dir / '09_splits' / 'test' / 'y_test.csv')

if isinstance(y_train, pd.DataFrame):
    y_train = y_train.iloc[:, 0]
if isinstance(y_val, pd.DataFrame):
    y_val = y_val.iloc[:, 0]
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.iloc[:, 0]

print(f"\nLoaded splits:")
print(f"  Train: {X_train.shape}")
print(f"  Val: {X_val.shape}")
print(f"  Test: {X_test.shape}")

# Train model
print("\n" + "="*80)
print("TRAINING MODEL")
print("="*80)

config = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

model = train_random_forest(X_train, y_train, X_val, y_val, config)
print(f"\n✓ Model trained with {config['n_estimators']} trees")

# Evaluate on all splits
train_results = evaluate_random_forest(model, X_train, y_train)
val_results = evaluate_random_forest(model, X_val, y_val)
test_results = evaluate_random_forest(model, X_test, y_test)

# Print metrics
print("\n" + "="*80)
print("MODEL PERFORMANCE METRICS")
print("="*80)

print("\n📊 OVERALL ACCURACY:")
print(f"  Train: {train_results['accuracy']:.3f}")
print(f"  Val:   {val_results['accuracy']:.3f}")
print(f"  Test:  {test_results['accuracy']:.3f}")

# Per-split metrics
print("\n📈 DETAILED METRICS:")
for name, results in [('Train', train_results), ('Val', val_results), ('Test', test_results)]:
    metrics = validate_model_performance(y_test if name == 'Test' else (y_train if name == 'Train' else y_val), 
                                         results['predictions'])
    print(f"\n{name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")

# Per-class performance
print("\n" + "="*80)
print("PER-CLASS PERFORMANCE (Test Set)")
print("="*80)

per_class = validate_per_class_performance(y_test, test_results['predictions'])
print(per_class.to_string(index=False))

# Feature importance
print("\n" + "="*80)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*80)
print(test_results['feature_importance'].head(10).to_string(index=False))

# Confusion matrix
cm, report_df = get_confusion_matrix_report(y_test, test_results['predictions'])
print("\n" + "="*80)
print("CONFUSION MATRIX (Test Set)")
print("="*80)
print(cm)

# Save visualizations
output_dir = Path('data/10_models/rf/evaluations')
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Feature Importance Plot
plt.figure(figsize=(12, 8))
top_features = test_results['feature_importance'].head(15)
plt.barh(range(len(top_features)), top_features['importance'].values)
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: feature_importance.png")
plt.close()

# 2. Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('Predicted Severity')
plt.ylabel('Actual Severity')
plt.title('Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: confusion_matrix.png")
plt.close()

# 3. Per-Class Metrics Bar Chart
plt.figure(figsize=(12, 6))
x_pos = np.arange(len(per_class))
width = 0.25

plt.bar(x_pos - width, per_class['precision'], width, label='Precision', alpha=0.8)
plt.bar(x_pos, per_class['recall'], width, label='Recall', alpha=0.8)
plt.bar(x_pos + width, per_class['f1'], width, label='F1 Score', alpha=0.8)

plt.xlabel('Severity Class')
plt.ylabel('Score')
plt.title('Per-Class Performance Metrics (Test Set)')
plt.xticks(x_pos, [f"Severity {c}" for c in per_class['class']])
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.ylim([0, 1.1])
plt.tight_layout()
plt.savefig(output_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: per_class_metrics.png")
plt.close()

# 4. Class Distribution
plt.figure(figsize=(10, 6))
severity_counts = pd.Series(test_results['predictions']).value_counts().sort_index()
plt.bar(severity_counts.index, severity_counts.values, alpha=0.7)
plt.xlabel('Severity')
plt.ylabel('Count')
plt.title('Predicted Severity Distribution (Test Set)')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'severity_distribution.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: severity_distribution.png")
plt.close()

# 5. Metrics Comparison Across Splits
plt.figure(figsize=(12, 6))
splits = ['Train', 'Val', 'Test']
accuracies = [train_results['accuracy'], val_results['accuracy'], test_results['accuracy']]
colors = ['#2ecc71', '#3498db', '#e74c3c']

plt.bar(splits, accuracies, color=colors, alpha=0.8)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Across Splits')
plt.ylim([0, 1.1])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: accuracy_comparison.png")
plt.close()

# Save model
models_dir = Path('data/10_models/rf')
models_dir.mkdir(parents=True, exist_ok=True)

with open(models_dir / 'model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n" + "="*80)
print("✅ TRAINING & EVALUATION COMPLETE")
print("="*80)
print(f"\n📁 Outputs saved to: {output_dir}")
print(f"📁 Model saved to: {models_dir / 'model.pkl'}")
print("\n📊 All visualizations saved!")
print("\nKey Insights:")
print(f"  • Test Accuracy: {test_results['accuracy']:.1%}")
print(f"  • Top Feature: {test_results['feature_importance'].iloc[0]['feature']}")
print(f"  • Test Samples: {len(X_test)}")
print("="*80)

