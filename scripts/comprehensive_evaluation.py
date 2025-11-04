#!/usr/bin/env python3
"""Comprehensive evaluation with extended visualizations."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sledhead_imu.models.random_forest import train_random_forest, evaluate_random_forest
from sledhead_imu.validate.validate_cutoffs import validate_model_performance, validate_per_class_performance

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("="*80)
print("COMPREHENSIVE MODEL EVALUATION WITH EXTENDED VISUALIZATIONS")
print("="*80)

# Load data
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

# Train model
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

# Evaluate
train_results = evaluate_random_forest(model, X_train, y_train)
val_results = evaluate_random_forest(model, X_val, y_val)
test_results = evaluate_random_forest(model, X_test, y_test)

# Check if binary classification
unique_labels = sorted(pd.concat([y_train, y_val, y_test]).unique())
is_binary = len(unique_labels) == 2 and set(unique_labels) == {0, 1}

# Label mapping
if is_binary:
    label_map = {0: 'No Symptoms', 1: 'Symptoms'}
    label_name = 'Symptom Status'
else:
    label_map = {i: f'Severity {i}' for i in range(6)}
    label_name = 'Severity'

# Output directory
output_dir = Path('data/10_models/rf/evaluations')
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Class Distribution Across Splits
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (split_name, y_data) in enumerate([('Train', y_train), ('Val', y_val), ('Test', y_test)]):
    counts = pd.Series(y_data).value_counts().sort_index()
    if is_binary:
        labels = [label_map[i] for i in counts.index]
        axes[idx].bar(range(len(counts)), counts.values, alpha=0.7, color=['#2ecc71', '#e74c3c'])
        axes[idx].set_xticks(range(len(counts)))
        axes[idx].set_xticklabels(labels, rotation=45, ha='right')
    else:
        axes[idx].bar(counts.index, counts.values, alpha=0.7)
        axes[idx].set_xlabel(label_name)
    axes[idx].set_ylabel('Count')
    axes[idx].set_title(f'{split_name} Set')
    axes[idx].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: class_distribution.png")
plt.close()

# 2. Feature Correlation Heatmap
corr_matrix = X_train.corr()
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=False, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(output_dir / 'feature_correlation.png', dpi=150, bbox_inches='tight')
print("✓ Saved: feature_correlation.png")
plt.close()

# 3. Feature Importance with Error Bars (if we had multiple runs)
fi_df = test_results['feature_importance'].head(15)
fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(range(len(fi_df)), fi_df['importance'].values)
ax.set_yticks(range(len(fi_df)))
ax.set_yticklabels(fi_df['feature'].values)
ax.set_xlabel('Importance')
ax.set_title('Top 15 Feature Importances (with percentages)')
ax.invert_yaxis()

# Add percentage labels
total_importance = fi_df['importance'].sum()
for i, imp in enumerate(fi_df['importance'].values):
    pct = (imp / total_importance) * 100
    ax.text(imp, i, f' {pct:.1f}%', va='center')

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance_detailed.png', dpi=150, bbox_inches='tight')
print("✓ Saved: feature_importance_detailed.png")
plt.close()

# 4. Predictions vs Actual Scatter
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (split_name, y_actual, results) in enumerate([
    ('Train', y_train, train_results),
    ('Val', y_val, val_results),
    ('Test', y_test, test_results)
]):
    y_pred = results['predictions']
    axes[idx].scatter(y_actual, y_pred, alpha=0.6)
    axes[idx].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
    axes[idx].set_xlabel(f'Actual {label_name}')
    axes[idx].set_ylabel(f'Predicted {label_name}')
    axes[idx].set_title(f'{split_name}')
    if is_binary:
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels([label_map[0], label_map[1]])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_yticklabels([label_map[0], label_map[1]])
    axes[idx].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'predictions_vs_actual.png', dpi=150, bbox_inches='tight')
print("✓ Saved: predictions_vs_actual.png")
plt.close()

# 5. Metric Comparison Across Splits
metrics_dict = {}
for split_name, y_true, results in [('Train', y_train, train_results), ('Val', y_val, val_results), ('Test', y_test, test_results)]:
    metrics = validate_model_performance(y_true, results['predictions'])
    metrics_dict[split_name] = metrics

metrics_df = pd.DataFrame(metrics_dict).T
fig, ax = plt.subplots(figsize=(10, 6))
metrics_df[['accuracy', 'precision', 'recall', 'f1']].plot(kind='bar', ax=ax, alpha=0.8)
ax.set_ylabel('Score')
ax.set_xlabel('Split')
ax.set_title('Performance Metrics Across Splits')
ax.set_ylim([0, 1.1])
ax.legend(title='Metric')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: metrics_comparison.png")
plt.close()

# 6. Confusion Matrix Heatmap (enhanced)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, test_results['predictions'])

fig, ax = plt.subplots(figsize=(10, 8) if not is_binary else (8, 6))
if is_binary:
    labels = [label_map[0], label_map[1]]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                xticklabels=labels, yticklabels=labels)
else:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                xticklabels=[f'Severity {i}' for i in range(6)],
                yticklabels=[f'Severity {i}' for i in range(6)])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - Test Set')
plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix_enhanced.png', dpi=150, bbox_inches='tight')
print("✓ Saved: confusion_matrix_enhanced.png")
plt.close()

# 7. Per-Class Precision, Recall, F1
per_class = validate_per_class_performance(y_test, test_results['predictions'])

fig, ax = plt.subplots(figsize=(12, 6) if not is_binary else (8, 6))
x_pos = np.arange(len(per_class))
width = 0.25

ax.bar(x_pos - width, per_class['precision'], width, label='Precision', alpha=0.8, color='#3498db')
ax.bar(x_pos, per_class['recall'], width, label='Recall', alpha=0.8, color='#2ecc71')
ax.bar(x_pos + width, per_class['f1'], width, label='F1 Score', alpha=0.8, color='#e74c3c')

ax.set_xlabel(label_name)
ax.set_ylabel('Score')
ax.set_title('Per-Class Performance Metrics (Test Set)')
ax.set_xticks(x_pos)
if is_binary:
    ax.set_xticklabels([label_map[int(c)] for c in per_class['class']])
else:
    ax.set_xticklabels([f"Severity {c}" for c in per_class['class']])
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1.1])

# Add value labels
for i, (prec, rec, f1) in enumerate(zip(per_class['precision'], per_class['recall'], per_class['f1'])):
    if prec > 0:
        ax.text(i - width, prec, f'{prec:.2f}', ha='center', va='bottom', fontsize=8)
    if rec > 0:
        ax.text(i, rec, f'{rec:.2f}', ha='center', va='bottom', fontsize=8)
    if f1 > 0:
        ax.text(i + width, f1, f'{f1:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'per_class_metrics_detailed.png', dpi=150, bbox_inches='tight')
print("✓ Saved: per_class_metrics_detailed.png")
plt.close()

# 8. Feature Distribution Boxplot (top 5 features)
top_features = test_results['feature_importance'].head(5)['feature'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feat in enumerate(top_features):
    if idx < len(axes):
        df_list = []
        labels_list = []
        for split_name, X_split in [('Train', X_train), ('Test', X_test)]:
            if feat in X_split.columns:
                df_list.append(X_split[feat].values)
                labels_list.append([split_name] * len(X_split))
        
        axes[idx].boxplot(df_list, tick_labels=[l[0] for l in labels_list if l])
        axes[idx].set_title(f'{feat}')
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=45)

# Hide extra subplots
for idx in range(len(top_features), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(output_dir / 'feature_distributions.png', dpi=150, bbox_inches='tight')
print("✓ Saved: feature_distributions.png")
plt.close()

# 9. Model Accuracy Over Time (simulated with feature counts)
fig, ax = plt.subplots(figsize=(12, 6))
feature_counts = range(5, min(21, len(X_train.columns) + 1))
accuracies = []

for n_features in feature_counts:
    top_feat = test_results['feature_importance'].head(n_features)['feature'].tolist()
    X_train_subset = X_train[top_feat]
    X_test_subset = X_test[top_feat]
    
    mini_model = train_random_forest(X_train_subset, y_train, X_val[top_feat], y_val, config)
    mini_results = evaluate_random_forest(mini_model, X_test_subset, y_test)
    accuracies.append(mini_results['accuracy'])

ax.plot(feature_counts, accuracies, marker='o', linewidth=2, markersize=8)
ax.set_xlabel('Number of Top Features')
ax.set_ylabel('Test Accuracy')
ax.set_title('Model Performance vs Number of Features')
ax.grid(alpha=0.3)
ax.axhline(y=test_results['accuracy'], color='r', linestyle='--', label=f'Full Model ({test_results["accuracy"]:.3f})')
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / 'feature_count_vs_accuracy.png', dpi=150, bbox_inches='tight')
print("✓ Saved: feature_count_vs_accuracy.png")
plt.close()

print("\n" + "="*80)
print("✅ EXTENDED EVALUATION COMPLETE")
print("="*80)
print(f"\n📊 Generated 9 additional visualizations in {output_dir}")
print("\nVisualizations:")
print("  1. Class distribution across splits")
print("  2. Feature correlation heatmap")
print("  3. Detailed feature importance with percentages")
print("  4. Predictions vs actual scatter plots")
print("  5. Metrics comparison across splits")
print("  6. Enhanced confusion matrix")
print("  7. Detailed per-class metrics with values")
print("  8. Feature distributions (boxplots)")
print("  9. Feature count vs accuracy curve")
print("="*80)

