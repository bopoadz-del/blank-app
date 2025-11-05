## Model Evaluation Framework

A comprehensive Python framework for evaluating machine learning models with production-ready metrics and visualizations for both classification and regression tasks.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Metrics](#metrics)
- [Visualizations](#visualizations)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)

## Overview

This framework provides a complete suite of evaluation tools for machine learning models:

### Classification Metrics
- **Confusion Matrix**: With normalization options
- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1**: With micro/macro/weighted averaging
- **ROC Curve & AUC**: Receiver Operating Characteristic analysis
- **Precision-Recall Curve & AP**: Precision-Recall analysis
- **Classification Report**: Comprehensive per-class metrics

### Regression Metrics
- **MSE, RMSE, MAE**: Standard error metrics
- **R² Score**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error
- **Regression Report**: Comprehensive metrics summary

### Visualizations
- **Confusion Matrix Heatmaps**: Color-coded confusion visualization
- **ROC Curves**: With AUC scores
- **Precision-Recall Curves**: With Average Precision
- **Prediction Plots**: Actual vs Predicted
- **Residual Analysis**: Residual plots and distributions
- **Learning Curves**: Training progress visualization

### Custom Metrics
- **CustomMetric Class**: Base class for custom metrics
- **make_scorer**: Convert any function to a scorer

## Installation

### Core Dependencies
```bash
pip install numpy matplotlib
```

### Full Installation
```bash
# For complete functionality
pip install numpy matplotlib scipy
```

## Quick Start

### 1. Binary Classification

```python
from model_evaluation import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    plot_classification_report
)
import numpy as np

# Your predictions
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
y_score = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.3, 0.7, 0.85])

# Calculate metrics
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision: {precision_score(y_true, y_pred):.3f}")
print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.3f}")
print(f"ROC AUC:   {roc_auc_score(y_true, y_score):.3f}")

# Visualize
fig = plot_classification_report(y_true, y_pred, y_score)
plt.show()
```

### 2. Multi-class Classification

```python
from model_evaluation import (
    confusion_matrix,
    classification_report,
    plot_confusion_matrix
)

# Multi-class predictions
y_true = np.array([0, 1, 2, 0, 1, 2, 1, 2, 0])
y_pred = np.array([0, 2, 2, 0, 1, 1, 1, 2, 0])

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Classification report
report = classification_report(
    y_true, y_pred,
    target_names=['Class A', 'Class B', 'Class C']
)
print(report)

# Visualize
fig = plot_confusion_matrix(
    y_true, y_pred,
    target_names=['Class A', 'Class B', 'Class C'],
    normalize='true'
)
plt.show()
```

### 3. Regression

```python
from model_evaluation import (
    mean_squared_error,
    r2_score,
    regression_report,
    plot_regression_report
)

# Your predictions
y_true = np.array([3.0, -0.5, 2.0, 7.0, 4.2])
y_pred = np.array([2.5, 0.0, 2.1, 7.8, 4.0])

# Calculate metrics
print(f"MSE:  {mean_squared_error(y_true, y_pred):.3f}")
print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.3f}")
print(f"MAE:  {mean_absolute_error(y_true, y_pred):.3f}")
print(f"R²:   {r2_score(y_true, y_pred):.3f}")

# Full report
report = regression_report(y_true, y_pred)
print(report)

# Visualize
fig = plot_regression_report(y_true, y_pred)
plt.show()
```

## Metrics

### Classification Metrics

#### Confusion Matrix

```python
from model_evaluation import confusion_matrix

# Basic confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Normalized by true labels
cm_norm = confusion_matrix(y_true, y_pred, normalize='true')

# Normalized by predictions
cm_norm = confusion_matrix(y_true, y_pred, normalize='pred')
```

#### Accuracy, Precision, Recall, F1

```python
from model_evaluation import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Binary classification
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Multi-class with averaging
precision_macro = precision_score(y_true, y_pred, average='macro')
precision_weighted = precision_score(y_true, y_pred, average='weighted')
precision_per_class = precision_score(y_true, y_pred, average=None)
```

#### ROC Analysis

```python
from model_evaluation import roc_curve, roc_auc_score

# ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_score)

# AUC
auc = roc_auc_score(y_true, y_score)

# Find optimal threshold (Youden's index)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]
```

#### Precision-Recall Analysis

```python
from model_evaluation import precision_recall_curve, average_precision_score

# PR curve
precision, recall, thresholds = precision_recall_curve(y_true, y_score)

# Average Precision
ap = average_precision_score(y_true, y_score)
```

#### Classification Report

```python
from model_evaluation import classification_report

report = classification_report(
    y_true, y_pred,
    target_names=['Negative', 'Positive']
)
print(report)

# Output:
#               precision    recall  f1-score   support
#
#     Negative     0.8333    0.8333    0.8333         6
#     Positive     0.7500    0.7500    0.7500         4
#
#     accuracy                         0.8000        10
#    macro avg     0.7917    0.7917    0.7917        10
# weighted avg     0.8000    0.8000    0.8000        10
```

### Regression Metrics

#### Basic Metrics

```python
from model_evaluation import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

mse = mean_squared_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
```

#### Regression Report

```python
from model_evaluation import regression_report

report = regression_report(y_true, y_pred)
print(report)

# Output:
# {
#     'mse': 0.1234,
#     'rmse': 0.3513,
#     'mae': 0.2800,
#     'r2': 0.9456,
#     'mape': 5.23
# }
```

### Custom Metrics

#### Using CustomMetric Class

```python
from model_evaluation import CustomMetric, confusion_matrix

class BusinessCostMetric(CustomMetric):
    """Calculate business cost based on FP and FN."""

    def __call__(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        fp_cost = cm[0, 1] * 10   # False Positive cost
        fn_cost = cm[1, 0] * 100  # False Negative cost
        return fp_cost + fn_cost

metric = BusinessCostMetric()
cost = metric(y_true, y_pred)
```

#### Using make_scorer

```python
from model_evaluation import make_scorer, f1_score

def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

scorer = make_scorer(weighted_f1, greater_is_better=True)
score = scorer(y_true, y_pred)
```

## Visualizations

### Confusion Matrix

```python
from model_evaluation import plot_confusion_matrix

fig = plot_confusion_matrix(
    y_true, y_pred,
    target_names=['Cat', 'Dog', 'Bird'],
    normalize='true',
    cmap='Blues',
    title='Species Classification'
)
plt.savefig('confusion_matrix.png')
```

### ROC Curve

```python
from model_evaluation import plot_roc_curve

# Single model
fig = plot_roc_curve(y_true, y_score)

# Multiple models
models = {
    'Model A': y_score_a,
    'Model B': y_score_b,
    'Model C': y_score_c
}
fig = plot_roc_curve(y_true, models)
plt.savefig('roc_curves.png')
```

### Precision-Recall Curve

```python
from model_evaluation import plot_precision_recall_curve

# Single model
fig = plot_precision_recall_curve(y_true, y_score)

# Multiple models
fig = plot_precision_recall_curve(y_true, models)
plt.savefig('pr_curves.png')
```

### Complete Classification Report

```python
from model_evaluation import plot_classification_report

# Combines confusion matrix, ROC, and PR curves
fig = plot_classification_report(
    y_true, y_pred, y_score,
    target_names=['Negative', 'Positive']
)
plt.savefig('classification_report.png')
```

### Regression Plots

```python
from model_evaluation import (
    plot_predictions,
    plot_residuals,
    plot_regression_report
)

# Actual vs Predicted
fig = plot_predictions(y_true, y_pred)
plt.savefig('predictions.png')

# Residual analysis
fig = plot_residuals(y_true, y_pred)
plt.savefig('residuals.png')

# Complete regression report
fig = plot_regression_report(y_true, y_pred)
plt.savefig('regression_report.png')
```

### Learning Curves

```python
from model_evaluation import plot_learning_curve, plot_multiple_learning_curves

# Single model
train_scores = [0.6, 0.7, 0.75, 0.78, 0.80, 0.82, 0.83]
val_scores = [0.58, 0.68, 0.72, 0.74, 0.75, 0.755, 0.76]

fig = plot_learning_curve(
    train_scores, val_scores,
    metric_name='Accuracy',
    title='Training Progress'
)
plt.savefig('learning_curve.png')

# Multiple models
curves = {
    'Model A': (train_a, val_a),
    'Model B': (train_b, val_b),
    'Model C': (train_c, val_c)
}
fig = plot_multiple_learning_curves(curves, metric_name='F1 Score')
plt.savefig('model_comparison.png')
```

## Examples

### Example 1: Complete Binary Classification Workflow

```python
import numpy as np
from model_evaluation import *

# Load data
y_true = np.array([...])  # True labels
y_pred = np.array([...])  # Predicted labels
y_score = np.array([...]) # Prediction probabilities

# 1. Calculate all metrics
print("=== Classification Metrics ===")
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_true, y_score):.4f}")

# 2. Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# 3. Full classification report
report = classification_report(y_true, y_pred)
print("\nClassification Report:")
print(report)

# 4. Find optimal threshold
fpr, tpr, thresholds = roc_curve(y_true, y_score)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"\nOptimal Threshold: {optimal_threshold:.4f}")

# 5. Visualize everything
fig = plot_classification_report(y_true, y_pred, y_score)
plt.savefig('full_evaluation.png', dpi=150, bbox_inches='tight')
```

### Example 2: Model Comparison

```python
# Compare 3 models
models = {
    'Logistic Regression': lr_scores,
    'Random Forest': rf_scores,
    'Neural Network': nn_scores
}

# Compare ROC AUC
print("=== Model Comparison (ROC AUC) ===")
for name, scores in models.items():
    auc = roc_auc_score(y_true, scores)
    print(f"{name:20s}: {auc:.4f}")

# Compare Average Precision
print("\n=== Model Comparison (AP) ===")
for name, scores in models.items():
    ap = average_precision_score(y_true, scores)
    print(f"{name:20s}: {ap:.4f}")

# Visualize ROC curves
fig = plot_roc_curve(y_true, models, title='Model Comparison')
plt.savefig('model_comparison_roc.png')

# Visualize PR curves
fig = plot_precision_recall_curve(y_true, models, title='Model Comparison')
plt.savefig('model_comparison_pr.png')
```

### Example 3: Regression Evaluation

```python
# Calculate all regression metrics
print("=== Regression Metrics ===")
print(f"MSE:  {mean_squared_error(y_true, y_pred):.4f}")
print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")
print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
print(f"R²:   {r2_score(y_true, y_pred):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred):.2f}%")

# Residual analysis
residuals = y_true - y_pred
print(f"\nMean Residual: {np.mean(residuals):.4f}")
print(f"Std Residual:  {np.std(residuals):.4f}")

# Visualize
fig = plot_regression_report(y_true, y_pred)
plt.savefig('regression_evaluation.png', dpi=150, bbox_inches='tight')
```

### Example 4: Custom Business Metric

```python
class FraudDetectionMetric(CustomMetric):
    """
    Custom metric for fraud detection.

    Priorities:
    - Minimize false negatives (missed fraud)
    - Acceptable false positive rate
    """

    def __call__(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)

        # Calculate metrics
        fn = cm[1, 0]  # Missed fraud (critical)
        fp = cm[0, 1]  # False alarms (acceptable)
        tp = cm[1, 1]  # Caught fraud

        # Custom score: heavily penalize missed fraud
        penalty = fn * 100 + fp * 1
        reward = tp * 50

        return reward - penalty

metric = FraudDetectionMetric()
score = metric(y_true, y_pred)
print(f"Fraud Detection Score: {score:.2f}")
```

## API Reference

### Classification Metrics

#### confusion_matrix
```python
confusion_matrix(y_true, y_pred, labels=None, normalize=None)
```
Compute confusion matrix.

**Parameters:**
- `y_true` (array): True labels
- `y_pred` (array): Predicted labels
- `labels` (list): Label ordering
- `normalize` (str): 'true', 'pred', 'all', or None

**Returns:** Confusion matrix (NxN array)

#### accuracy_score
```python
accuracy_score(y_true, y_pred)
```
Calculate accuracy: (TP + TN) / Total

#### precision_score
```python
precision_score(y_true, y_pred, average='binary', zero_division=0.0)
```
Calculate precision: TP / (TP + FP)

**Parameters:**
- `average`: 'binary', 'micro', 'macro', 'weighted', or None

#### recall_score
```python
recall_score(y_true, y_pred, average='binary', zero_division=0.0)
```
Calculate recall: TP / (TP + FN)

#### f1_score
```python
f1_score(y_true, y_pred, average='binary', zero_division=0.0)
```
Calculate F1: 2 * (precision * recall) / (precision + recall)

#### roc_curve
```python
roc_curve(y_true, y_score, pos_label=1)
```
Compute ROC curve.

**Returns:** (fpr, tpr, thresholds)

#### roc_auc_score
```python
roc_auc_score(y_true, y_score)
```
Compute Area Under ROC Curve.

#### precision_recall_curve
```python
precision_recall_curve(y_true, y_score, pos_label=1)
```
Compute Precision-Recall curve.

**Returns:** (precision, recall, thresholds)

#### average_precision_score
```python
average_precision_score(y_true, y_score)
```
Compute Average Precision.

### Regression Metrics

#### mean_squared_error
```python
mean_squared_error(y_true, y_pred)
```
MSE = mean((y_true - y_pred)²)

#### root_mean_squared_error
```python
root_mean_squared_error(y_true, y_pred)
```
RMSE = √MSE

#### mean_absolute_error
```python
mean_absolute_error(y_true, y_pred)
```
MAE = mean(|y_true - y_pred|)

#### r2_score
```python
r2_score(y_true, y_pred)
```
R² = 1 - (SS_res / SS_tot)

#### mean_absolute_percentage_error
```python
mean_absolute_percentage_error(y_true, y_pred)
```
MAPE = mean(|y_true - y_pred| / |y_true|) × 100

### Custom Metrics

#### CustomMetric
```python
class CustomMetric:
    def __call__(self, y_true, y_pred):
        # Implement your metric
        return score
```

#### make_scorer
```python
make_scorer(score_func, greater_is_better=True, **kwargs)
```
Create scorer from function.

## Best Practices

### 1. Choose Appropriate Metrics

**Imbalanced Classification:**
```python
# Don't rely solely on accuracy
# Use precision, recall, F1, and ROC AUC

print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
print(f"F1:        {f1_score(y_true, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_true, y_score):.4f}")
```

**Multi-class:**
```python
# Use macro/weighted averaging
precision_macro = precision_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
```

### 2. Set Operating Points Based on Business Requirements

```python
# Find threshold for specific precision/recall
precision, recall, thresholds = precision_recall_curve(y_true, y_score)

# 90% precision threshold
high_precision_idx = np.where(precision >= 0.9)[0][0]
threshold_90p = thresholds[high_precision_idx]
```

### 3. Always Visualize

```python
# Don't just look at numbers - visualize!
fig = plot_classification_report(y_true, y_pred, y_score)
plt.show()
```

### 4. Check for Overfitting

```python
# Use learning curves
fig = plot_learning_curve(train_scores, val_scores)

# Large gap indicates overfitting
train_val_gap = train_scores[-1] - val_scores[-1]
if train_val_gap > 0.1:
    print("⚠ Warning: Possible overfitting detected!")
```

### 5. Compare Multiple Models

```python
# Never evaluate just one model
models = {'Model A': scores_a, 'Model B': scores_b, 'Model C': scores_c}

for name, scores in models.items():
    auc = roc_auc_score(y_true, scores)
    print(f"{name}: {auc:.4f}")

# Visualize comparison
plot_roc_curve(y_true, models)
```

### 6. Use Cross-Validation

```python
# Don't evaluate on a single train-test split
# Use k-fold cross-validation for robust estimates

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"F1: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

## File Structure

```
model-evaluation/
├── __init__.py              # Package initialization
├── metrics.py               # All evaluation metrics
├── visualization.py         # Plotting utilities
├── examples.py              # Comprehensive examples
└── README.md                # This file
```

## Requirements

- Python >= 3.7
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0

Optional:
- SciPy >= 1.6.0 (for advanced statistics)

## Performance Notes

All metrics are implemented in pure NumPy for maximum performance:

| Metric | Time Complexity | Space Complexity |
|--------|----------------|------------------|
| Confusion Matrix | O(n) | O(k²) where k = # classes |
| Accuracy | O(n) | O(1) |
| Precision/Recall | O(n) | O(k) |
| ROC Curve | O(n log n) | O(n) |
| PR Curve | O(n log n) | O(n) |

## License

This framework is provided as-is for educational and commercial use.

## Citation

```bibtex
@software{model_evaluation_framework,
  title = {Model Evaluation Framework},
  author = {ML Framework Team},
  year = {2024},
  description = {Comprehensive Python framework for ML model evaluation}
}
```

---

**Happy Evaluating! 📊**
