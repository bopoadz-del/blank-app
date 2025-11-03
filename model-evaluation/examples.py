"""
Model Evaluation Framework Examples

Comprehensive examples demonstrating metrics and visualization utilities
for classification and regression tasks.

Author: ML Framework Team
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# Import metrics
from metrics import (
    # Classification
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    # Regression
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    regression_report,
    # Custom
    CustomMetric,
    make_scorer
)

# Import visualization
from visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_classification_report,
    plot_predictions,
    plot_residuals,
    plot_regression_report,
    plot_learning_curve,
    plot_multiple_learning_curves
)


# ============================================================================
# EXAMPLE 1: BINARY CLASSIFICATION EVALUATION
# ============================================================================

def example_binary_classification():
    """
    Complete evaluation workflow for binary classification.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: BINARY CLASSIFICATION EVALUATION")
    print("=" * 70)

    # Generate synthetic data (e.g., fraud detection)
    np.random.seed(42)
    n_samples = 1000

    # Simulate imbalanced dataset (10% fraud)
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

    # Simulate model predictions (better than random)
    y_score = np.random.rand(n_samples)
    y_score[y_true == 1] += 0.3  # Boost fraud scores
    y_score = np.clip(y_score, 0, 1)

    y_pred = (y_score > 0.5).astype(int)

    # 1. Basic Metrics
    print("\n1. Basic Classification Metrics")
    print("-" * 70)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")

    # 2. Confusion Matrix
    print("\n2. Confusion Matrix")
    print("-" * 70)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("\nInterpretation:")
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")

    # 3. ROC Analysis
    print("\n3. ROC Analysis")
    print("-" * 70)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    print(f"ROC AUC: {auc:.4f}")

    # Find optimal threshold (Youden's index)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  TPR at optimal: {tpr[optimal_idx]:.4f}")
    print(f"  FPR at optimal: {fpr[optimal_idx]:.4f}")

    # 4. Precision-Recall Analysis
    print("\n4. Precision-Recall Analysis")
    print("-" * 70)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    print(f"Average Precision: {ap:.4f}")

    # Find threshold for 90% precision
    high_precision_idx = np.where(precision >= 0.9)[0]
    if len(high_precision_idx) > 0:
        idx = high_precision_idx[0]
        print(f"Threshold for 90% precision: {pr_thresholds[idx]:.4f}")
        print(f"  Recall at 90% precision: {recall[idx]:.4f}")

    # 5. Full Classification Report
    print("\n5. Full Classification Report")
    print("-" * 70)
    report = classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Fraud']
    )
    print(report)

    # 6. Visualizations
    print("\n6. Creating Visualizations...")
    print("-" * 70)

    # Complete classification report plot
    fig = plot_classification_report(
        y_true, y_pred, y_score,
        target_names=['Normal', 'Fraud']
    )
    plt.savefig('/tmp/binary_classification_full_report.png', dpi=150, bbox_inches='tight')
    print("Saved full report to /tmp/binary_classification_full_report.png")
    plt.close()

    print("\n✓ Binary classification evaluation complete!")


# ============================================================================
# EXAMPLE 2: MULTI-CLASS CLASSIFICATION EVALUATION
# ============================================================================

def example_multiclass_classification():
    """
    Complete evaluation workflow for multi-class classification.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: MULTI-CLASS CLASSIFICATION EVALUATION")
    print("=" * 70)

    # Generate synthetic data (e.g., iris classification)
    np.random.seed(42)
    n_samples = 500
    n_classes = 3

    y_true = np.random.randint(0, n_classes, n_samples)

    # Simulate predictions with some confusion between classes
    y_pred = y_true.copy()
    confusion_indices = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    y_pred[confusion_indices] = np.random.randint(0, n_classes, len(confusion_indices))

    class_names = ['Setosa', 'Versicolor', 'Virginica']

    # 1. Overall Accuracy
    print("\n1. Overall Metrics")
    print("-" * 70)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    # 2. Per-Class Metrics
    print("\n2. Per-Class Metrics")
    print("-" * 70)

    # Precision for each class
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    for i, class_name in enumerate(class_names):
        print(f"\n{class_name}:")
        print(f"  Precision: {precision_per_class[i]:.4f}")
        print(f"  Recall:    {recall_per_class[i]:.4f}")
        print(f"  F1 Score:  {f1_per_class[i]:.4f}")

    # 3. Averaged Metrics
    print("\n3. Averaged Metrics")
    print("-" * 70)
    print(f"Macro Precision:    {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Macro F1:           {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1:        {f1_score(y_true, y_pred, average='weighted'):.4f}")

    # 4. Confusion Matrix Analysis
    print("\n4. Confusion Matrix")
    print("-" * 70)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Analyze most confused pairs
    print("\nMost confused class pairs:")
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                print(f"  {class_names[i]} → {class_names[j]}: {cm[i, j]} samples")

    # 5. Full Classification Report
    print("\n5. Full Classification Report")
    print("-" * 70)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # 6. Visualizations
    print("\n6. Creating Visualizations...")
    print("-" * 70)

    # Normalized confusion matrix
    fig = plot_confusion_matrix(
        y_true, y_pred,
        target_names=class_names,
        normalize='true',
        title='Multi-class Confusion Matrix (Normalized)'
    )
    plt.savefig('/tmp/multiclass_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Saved confusion matrix to /tmp/multiclass_confusion_matrix.png")
    plt.close()

    print("\n✓ Multi-class classification evaluation complete!")


# ============================================================================
# EXAMPLE 3: REGRESSION EVALUATION
# ============================================================================

def example_regression():
    """
    Complete evaluation workflow for regression.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: REGRESSION EVALUATION")
    print("=" * 70)

    # Generate synthetic data (e.g., house price prediction)
    np.random.seed(42)
    n_samples = 300

    # True values (house prices in thousands)
    y_true = np.random.randn(n_samples) * 50 + 300

    # Predictions with some error
    noise = np.random.randn(n_samples) * 20
    y_pred = y_true + noise

    # Add some outliers
    outlier_indices = np.random.choice(n_samples, size=10, replace=False)
    y_pred[outlier_indices] += np.random.randn(10) * 100

    # 1. Basic Regression Metrics
    print("\n1. Basic Regression Metrics")
    print("-" * 70)
    print(f"MSE:  {mean_squared_error(y_true, y_pred):.2f}")
    print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.2f}")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"R²:   {r2_score(y_true, y_pred):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred):.2f}%")

    # 2. Residual Analysis
    print("\n2. Residual Analysis")
    print("-" * 70)
    residuals = y_true - y_pred
    print(f"Mean Residual:   {np.mean(residuals):.4f}")
    print(f"Std Residual:    {np.std(residuals):.4f}")
    print(f"Min Residual:    {np.min(residuals):.2f}")
    print(f"Max Residual:    {np.max(residuals):.2f}")
    print(f"Median Residual: {np.median(residuals):.2f}")

    # 3. Error Distribution
    print("\n3. Error Distribution")
    print("-" * 70)
    abs_errors = np.abs(residuals)
    print(f"Mean Absolute Error:   {np.mean(abs_errors):.2f}")
    print(f"Median Absolute Error: {np.median(abs_errors):.2f}")
    print(f"90th Percentile Error: {np.percentile(abs_errors, 90):.2f}")
    print(f"95th Percentile Error: {np.percentile(abs_errors, 95):.2f}")

    # 4. Prediction Accuracy by Range
    print("\n4. Accuracy by Price Range")
    print("-" * 70)

    ranges = [(0, 250), (250, 300), (300, 350), (350, 400), (400, 1000)]
    for low, high in ranges:
        mask = (y_true >= low) & (y_true < high)
        if np.sum(mask) > 0:
            range_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            range_r2 = r2_score(y_true[mask], y_pred[mask])
            print(f"  ${low}k-${high}k: MAE={range_mae:.2f}, R²={range_r2:.4f}, n={np.sum(mask)}")

    # 5. Full Regression Report
    print("\n5. Full Regression Report")
    print("-" * 70)
    report = regression_report(y_true, y_pred)
    for metric, value in report.items():
        print(f"{metric.upper():8s}: {value:.4f}")

    # 6. Visualizations
    print("\n6. Creating Visualizations...")
    print("-" * 70)

    # Complete regression report
    fig = plot_regression_report(y_true, y_pred)
    plt.savefig('/tmp/regression_full_report.png', dpi=150, bbox_inches='tight')
    print("Saved full report to /tmp/regression_full_report.png")
    plt.close()

    print("\n✓ Regression evaluation complete!")


# ============================================================================
# EXAMPLE 4: MODEL COMPARISON
# ============================================================================

def example_model_comparison():
    """
    Compare multiple models using various metrics.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: MODEL COMPARISON")
    print("=" * 70)

    # Generate data
    np.random.seed(42)
    n_samples = 500
    y_true = np.random.randint(0, 2, n_samples)

    # Simulate 3 different models
    models = {
        'Logistic Regression': np.random.rand(n_samples),
        'Random Forest': np.random.rand(n_samples),
        'Neural Network': np.random.rand(n_samples)
    }

    # Make each model slightly better
    for i, (name, scores) in enumerate(models.items()):
        scores[y_true == 1] += 0.2 + i * 0.1
        models[name] = np.clip(scores, 0, 1)

    # 1. Compare ROC AUC
    print("\n1. ROC AUC Comparison")
    print("-" * 70)
    auc_scores = {}
    for name, scores in models.items():
        auc = roc_auc_score(y_true, scores)
        auc_scores[name] = auc
        print(f"{name:20s}: {auc:.4f}")

    # Best model by AUC
    best_model = max(auc_scores.items(), key=lambda x: x[1])
    print(f"\nBest model: {best_model[0]} (AUC = {best_model[1]:.4f})")

    # 2. Compare Average Precision
    print("\n2. Average Precision Comparison")
    print("-" * 70)
    ap_scores = {}
    for name, scores in models.items():
        ap = average_precision_score(y_true, scores)
        ap_scores[name] = ap
        print(f"{name:20s}: {ap:.4f}")

    # 3. Compare at Fixed Operating Point
    print("\n3. Performance at Threshold = 0.5")
    print("-" * 70)
    for name, scores in models.items():
        y_pred = (scores > 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

    # 4. Visualizations
    print("\n4. Creating Visualizations...")
    print("-" * 70)

    # Compare ROC curves
    fig = plot_roc_curve(y_true, models, title='Model Comparison: ROC Curves')
    plt.savefig('/tmp/model_comparison_roc.png', dpi=150, bbox_inches='tight')
    print("Saved ROC comparison to /tmp/model_comparison_roc.png")
    plt.close()

    # Compare PR curves
    fig = plot_precision_recall_curve(y_true, models, title='Model Comparison: PR Curves')
    plt.savefig('/tmp/model_comparison_pr.png', dpi=150, bbox_inches='tight')
    print("Saved PR comparison to /tmp/model_comparison_pr.png")
    plt.close()

    print("\n✓ Model comparison complete!")


# ============================================================================
# EXAMPLE 5: CUSTOM METRICS
# ============================================================================

def example_custom_metrics():
    """
    Demonstrate custom metric creation.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: CUSTOM METRICS")
    print("=" * 70)

    # Generate data
    np.random.seed(42)
    n_samples = 200
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    y_score = np.random.rand(n_samples)

    # 1. Custom Metric: Business Cost Function
    print("\n1. Custom Business Cost Metric")
    print("-" * 70)

    class BusinessCostMetric(CustomMetric):
        """
        Custom metric: Calculate business cost.
        False Positive cost: $10
        False Negative cost: $100
        """
        def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
            cm = confusion_matrix(y_true, y_pred)
            fp_cost = cm[0, 1] * 10  # False positives
            fn_cost = cm[1, 0] * 100  # False negatives
            total_cost = fp_cost + fn_cost
            return total_cost

    cost_metric = BusinessCostMetric()
    cost = cost_metric(y_true, y_pred)
    print(f"Total Business Cost: ${cost:.2f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"  False Positives: {cm[0, 1]} × $10 = ${cm[0, 1] * 10:.2f}")
    print(f"  False Negatives: {cm[1, 0]} × $100 = ${cm[1, 0] * 100:.2f}")

    # 2. Custom Metric: Top-K Accuracy
    print("\n2. Custom Top-K Accuracy")
    print("-" * 70)

    def top_k_accuracy(y_true: np.ndarray, y_scores: np.ndarray, k: int = 2) -> float:
        """
        Calculate top-k accuracy for multi-class classification.
        """
        n_classes = len(np.unique(y_true))

        # Generate fake probability scores for demo
        y_proba = np.random.rand(len(y_true), n_classes)

        # Get top-k predictions
        top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]

        # Check if true label is in top-k
        correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])

        return np.mean(correct)

    top1_acc = accuracy_score(y_true, y_pred)
    top2_acc = top_k_accuracy(y_true, y_score, k=2)

    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-2 Accuracy: {top2_acc:.4f}")

    # 3. Custom Metric with make_scorer
    print("\n3. Using make_scorer")
    print("-" * 70)

    def weighted_f1(y_true, y_pred):
        """F1 score with custom weighting."""
        return f1_score(y_true, y_pred, average='weighted')

    scorer = make_scorer(weighted_f1, greater_is_better=True)
    score = scorer(y_true, y_pred)
    print(f"Weighted F1 Score: {score:.4f}")

    print("\n✓ Custom metrics examples complete!")


# ============================================================================
# EXAMPLE 6: LEARNING CURVES AND TRAINING MONITORING
# ============================================================================

def example_learning_curves():
    """
    Demonstrate learning curve visualization.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: LEARNING CURVES")
    print("=" * 70)

    # Simulate training history for 3 models
    np.random.seed(42)
    epochs = 50

    # Model 1: Good convergence
    train1 = 0.5 + 0.4 * (1 - np.exp(-np.arange(epochs) / 10)) + np.random.randn(epochs) * 0.02
    val1 = 0.5 + 0.35 * (1 - np.exp(-np.arange(epochs) / 10)) + np.random.randn(epochs) * 0.03

    # Model 2: Overfitting
    train2 = 0.5 + 0.45 * (1 - np.exp(-np.arange(epochs) / 8)) + np.random.randn(epochs) * 0.02
    val2 = 0.5 + 0.3 * (1 - np.exp(-np.arange(epochs) / 8))
    val2[25:] -= (np.arange(25) * 0.004)  # Start overfitting

    # Model 3: Underfitting
    train3 = 0.5 + 0.25 * (1 - np.exp(-np.arange(epochs) / 15)) + np.random.randn(epochs) * 0.02
    val3 = 0.5 + 0.23 * (1 - np.exp(-np.arange(epochs) / 15)) + np.random.randn(epochs) * 0.03

    # Clip values
    train1, val1 = np.clip(train1, 0, 1), np.clip(val1, 0, 1)
    train2, val2 = np.clip(train2, 0, 1), np.clip(val2, 0, 1)
    train3, val3 = np.clip(train3, 0, 1), np.clip(val3, 0, 1)

    print("\n1. Model Diagnostics")
    print("-" * 70)

    # Analyze each model
    models_analysis = {
        'Model 1 (Well-fitted)': (train1, val1),
        'Model 2 (Overfitting)': (train2, val2),
        'Model 3 (Underfitting)': (train3, val3)
    }

    for name, (train, val) in models_analysis.items():
        final_train = train[-1]
        final_val = val[-1]
        gap = final_train - final_val

        print(f"\n{name}:")
        print(f"  Final Train Accuracy: {final_train:.4f}")
        print(f"  Final Val Accuracy:   {final_val:.4f}")
        print(f"  Train-Val Gap:        {gap:.4f}")

        if gap > 0.1:
            print(f"  → Likely overfitting")
        elif final_val < 0.7:
            print(f"  → Likely underfitting")
        else:
            print(f"  → Well-fitted")

    # 2. Visualizations
    print("\n2. Creating Visualizations...")
    print("-" * 70)

    # Individual learning curve
    fig = plot_learning_curve(
        train1, val1,
        metric_name='Accuracy',
        title='Learning Curve: Well-Fitted Model'
    )
    plt.savefig('/tmp/learning_curve_single.png', dpi=150, bbox_inches='tight')
    print("Saved single learning curve to /tmp/learning_curve_single.png")
    plt.close()

    # Compare all models
    curves = {
        'Well-fitted': (train1, val1),
        'Overfitting': (train2, val2),
        'Underfitting': (train3, val3)
    }

    fig = plot_multiple_learning_curves(
        curves,
        metric_name='Accuracy',
        title='Model Comparison: Training Dynamics'
    )
    plt.savefig('/tmp/learning_curves_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved learning curves comparison to /tmp/learning_curves_comparison.png")
    plt.close()

    print("\n✓ Learning curves examples complete!")


# ============================================================================
# MAIN: RUN ALL EXAMPLES
# ============================================================================

def main():
    """
    Run all examples.
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION FRAMEWORK - COMPREHENSIVE EXAMPLES")
    print("=" * 70)

    # Run all examples
    example_binary_classification()
    example_multiclass_classification()
    example_regression()
    example_model_comparison()
    example_custom_metrics()
    example_learning_curves()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated files in /tmp/:")
    print("  - binary_classification_full_report.png")
    print("  - multiclass_confusion_matrix.png")
    print("  - regression_full_report.png")
    print("  - model_comparison_roc.png")
    print("  - model_comparison_pr.png")
    print("  - learning_curve_single.png")
    print("  - learning_curves_comparison.png")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
