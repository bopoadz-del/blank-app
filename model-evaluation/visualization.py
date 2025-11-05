"""
Model Evaluation Visualization Utilities

Plotting functions for confusion matrices, ROC curves, PR curves, and regression diagnostics.

Author: ML Framework Team
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Union
import warnings

try:
    from .metrics import (
        confusion_matrix,
        roc_curve,
        roc_auc_score,
        precision_recall_curve,
        average_precision_score
    )
except ImportError:
    from metrics import (
        confusion_matrix,
        roc_curve,
        roc_auc_score,
        precision_recall_curve,
        average_precision_score
    )


# ============================================================================
# CLASSIFICATION VISUALIZATIONS
# ============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
    target_names: Optional[List[str]] = None,
    normalize: Optional[str] = None,
    cmap: str = 'Blues',
    figsize: Tuple[int, int] = (8, 6),
    title: str = 'Confusion Matrix',
    colorbar: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    labels : list, optional
        List of label indices to include.
    target_names : list of str, optional
        Display names for labels.
    normalize : str, optional
        Normalization mode: 'true', 'pred', 'all', or None.
    cmap : str, default='Blues'
        Colormap for the heatmap.
    figsize : tuple, default=(8, 6)
        Figure size.
    title : str, default='Confusion Matrix'
        Plot title.
    colorbar : bool, default=True
        Whether to show colorbar.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns:
    --------
    fig : plt.Figure
        Figure object.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    if colorbar:
        fig.colorbar(im, ax=ax)

    # Set labels
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    if target_names is None:
        target_names = [str(label) for label in labels]

    # Set ticks
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(target_names, rotation=45, ha='right')
    ax.set_yticklabels(target_names)

    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=10)

    ax.set_ylabel('True label', fontsize=11)
    ax.set_xlabel('Predicted label', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: Union[np.ndarray, dict],
    pos_label: int = 1,
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = 'ROC Curve',
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot ROC curve(s).

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray or dict
        Predicted probabilities or dict of {name: scores} for multiple curves.
    pos_label : int, default=1
        Label of positive class.
    labels : list of str, optional
        Labels for multiple curves (used if y_score is dict).
    figsize : tuple, default=(8, 6)
        Figure size.
    title : str, default='ROC Curve'
        Plot title.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns:
    --------
    fig : plt.Figure
        Figure object.
    """
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Handle single curve
    if isinstance(y_score, np.ndarray):
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.3f})')

    # Handle multiple curves
    elif isinstance(y_score, dict):
        for name, scores in y_score.items():
            fpr, tpr, _ = roc_curve(y_true, scores, pos_label=pos_label)
            auc = roc_auc_score(y_true, scores)
            ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')

    else:
        raise ValueError("y_score must be np.ndarray or dict")

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: Union[np.ndarray, dict],
    pos_label: int = 1,
    figsize: Tuple[int, int] = (8, 6),
    title: str = 'Precision-Recall Curve',
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot Precision-Recall curve(s).

    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray or dict
        Predicted probabilities or dict of {name: scores} for multiple curves.
    pos_label : int, default=1
        Label of positive class.
    figsize : tuple, default=(8, 6)
        Figure size.
    title : str, default='Precision-Recall Curve'
        Plot title.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns:
    --------
    fig : plt.Figure
        Figure object.
    """
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Handle single curve
    if isinstance(y_score, np.ndarray):
        precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, lw=2, label=f'PR curve (AP = {ap:.3f})')

    # Handle multiple curves
    elif isinstance(y_score, dict):
        for name, scores in y_score.items():
            precision, recall, _ = precision_recall_curve(y_true, scores, pos_label=pos_label)
            ap = average_precision_score(y_true, scores)
            ax.plot(recall, precision, lw=2, label=f'{name} (AP = {ap:.3f})')

    else:
        raise ValueError("y_score must be np.ndarray or dict")

    # Plot baseline
    baseline = np.sum(y_true == pos_label) / len(y_true)
    ax.plot([0, 1], [baseline, baseline], 'k--', lw=1,
            label=f'Random (AP = {baseline:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Comprehensive classification visualization with confusion matrix, ROC, and PR curves.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_score : np.ndarray, optional
        Predicted probabilities (for ROC and PR curves).
    target_names : list of str, optional
        Display names for classes.
    figsize : tuple, default=(14, 10)
        Figure size.

    Returns:
    --------
    fig : plt.Figure
        Figure object.
    """
    # Determine if binary classification
    unique_labels = np.unique(y_true)
    is_binary = len(unique_labels) == 2

    # Create subplots
    if is_binary and y_score is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]

    # Plot confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        target_names=target_names,
        normalize='true',
        ax=axes[0],
        title='Normalized Confusion Matrix'
    )

    # Plot ROC and PR curves for binary classification
    if is_binary and y_score is not None:
        pos_label = unique_labels[1] if len(unique_labels) == 2 else 1

        plot_roc_curve(
            y_true, y_score,
            pos_label=pos_label,
            ax=axes[1],
            title='ROC Curve'
        )

        plot_precision_recall_curve(
            y_true, y_score,
            pos_label=pos_label,
            ax=axes[2],
            title='Precision-Recall Curve'
        )

    plt.tight_layout()
    return fig


# ============================================================================
# REGRESSION VISUALIZATIONS
# ============================================================================

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
    title: str = 'Actual vs Predicted',
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot actual vs predicted values.

    Parameters:
    -----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    figsize : tuple, default=(8, 6)
        Figure size.
    title : str, default='Actual vs Predicted'
        Plot title.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns:
    --------
    fig : plt.Figure
        Figure object.
    """
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

    # Calculate R²
    from metrics import r2_score
    r2 = r2_score(y_true, y_pred)

    ax.set_xlabel('Actual Values', fontsize=11)
    ax.set_ylabel('Predicted Values', fontsize=11)
    ax.set_title(f'{title}\n$R^2$ = {r2:.4f}', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (12, 5),
    title: str = 'Residual Analysis'
) -> plt.Figure:
    """
    Plot residual analysis (residuals vs predicted and residual distribution).

    Parameters:
    -----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    figsize : tuple, default=(12, 5)
        Figure size.
    title : str, default='Residual Analysis'
        Plot title.

    Returns:
    --------
    fig : plt.Figure
        Figure object.
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values', fontsize=11)
    axes[0].set_ylabel('Residuals', fontsize=11)
    axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Residual distribution
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    axes[1].text(0.05, 0.95,
                f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}',
                transform=axes[1].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Comprehensive regression visualization with predictions and residuals.

    Parameters:
    -----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    figsize : tuple, default=(14, 5)
        Figure size.

    Returns:
    --------
    fig : plt.Figure
        Figure object.
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Actual vs Predicted
    plot_predictions(y_true, y_pred, ax=axes[0], title='Actual vs Predicted')

    # Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values', fontsize=11)
    axes[1].set_ylabel('Residuals', fontsize=11)
    axes[1].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Residual distribution
    axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[2].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[2].set_xlabel('Residuals', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    # Calculate and display metrics
    from metrics import regression_report
    metrics = regression_report(y_true, y_pred)

    metrics_text = '\n'.join([f'{k.upper()}: {v:.4f}' for k, v in metrics.items()])
    axes[2].text(0.05, 0.95,
                metrics_text,
                transform=axes[2].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8)

    plt.tight_layout()
    return fig


# ============================================================================
# LEARNING CURVES
# ============================================================================

def plot_learning_curve(
    train_scores: Union[np.ndarray, list],
    val_scores: Optional[Union[np.ndarray, list]] = None,
    train_label: str = 'Train',
    val_label: str = 'Validation',
    metric_name: str = 'Score',
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Learning Curve',
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot learning curve(s).

    Parameters:
    -----------
    train_scores : np.ndarray or list
        Training scores over epochs/iterations.
    val_scores : np.ndarray or list, optional
        Validation scores over epochs/iterations.
    train_label : str, default='Train'
        Label for training curve.
    val_label : str, default='Validation'
        Label for validation curve.
    metric_name : str, default='Score'
        Name of the metric being plotted.
    figsize : tuple, default=(10, 6)
        Figure size.
    title : str, default='Learning Curve'
        Plot title.
    ax : plt.Axes, optional
        Axes to plot on.

    Returns:
    --------
    fig : plt.Figure
        Figure object.
    """
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    epochs = np.arange(1, len(train_scores) + 1)

    # Plot training scores
    ax.plot(epochs, train_scores, 'o-', lw=2, label=train_label, markersize=4)

    # Plot validation scores if provided
    if val_scores is not None:
        ax.plot(epochs, val_scores, 's-', lw=2, label=val_label, markersize=4)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_multiple_learning_curves(
    curves: dict,
    metric_name: str = 'Score',
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Model Comparison'
) -> plt.Figure:
    """
    Plot multiple learning curves for model comparison.

    Parameters:
    -----------
    curves : dict
        Dictionary of {model_name: (train_scores, val_scores)}.
    metric_name : str, default='Score'
        Name of the metric being plotted.
    figsize : tuple, default=(10, 6)
        Figure size.
    title : str, default='Model Comparison'
        Plot title.

    Returns:
    --------
    fig : plt.Figure
        Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for model_name, (train_scores, val_scores) in curves.items():
        epochs = np.arange(1, len(train_scores) + 1)

        # Plot training curves
        axes[0].plot(epochs, train_scores, 'o-', lw=2, label=model_name, markersize=3)

        # Plot validation curves
        if val_scores is not None:
            axes[1].plot(epochs, val_scores, 's-', lw=2, label=model_name, markersize=3)

    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel(metric_name, fontsize=11)
    axes[0].set_title('Training', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel(metric_name, fontsize=11)
    axes[1].set_title('Validation', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ============================================================================
# EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MODEL EVALUATION VISUALIZATION EXAMPLES")
    print("=" * 70)

    # Generate sample classification data
    np.random.seed(42)
    n_samples = 1000

    # Binary classification
    y_true_binary = np.random.randint(0, 2, n_samples)
    y_score_binary = np.random.rand(n_samples)
    y_pred_binary = (y_score_binary > 0.5).astype(int)

    # Multi-class classification
    y_true_multi = np.random.randint(0, 3, n_samples)
    y_pred_multi = np.random.randint(0, 3, n_samples)

    # Regression
    y_true_reg = np.random.randn(n_samples) * 10 + 50
    y_pred_reg = y_true_reg + np.random.randn(n_samples) * 3

    # Example 1: Confusion Matrix
    print("\n1. Confusion Matrix")
    print("-" * 70)
    fig = plot_confusion_matrix(
        y_true_multi, y_pred_multi,
        target_names=['Class A', 'Class B', 'Class C'],
        normalize='true',
        title='Multi-class Confusion Matrix'
    )
    plt.savefig('/tmp/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/confusion_matrix.png")
    plt.close()

    # Example 2: ROC Curve
    print("\n2. ROC Curve")
    print("-" * 70)
    fig = plot_roc_curve(
        y_true_binary, y_score_binary,
        title='Binary Classification ROC'
    )
    plt.savefig('/tmp/roc_curve.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/roc_curve.png")
    plt.close()

    # Example 3: Precision-Recall Curve
    print("\n3. Precision-Recall Curve")
    print("-" * 70)
    fig = plot_precision_recall_curve(
        y_true_binary, y_score_binary,
        title='Binary Classification PR Curve'
    )
    plt.savefig('/tmp/pr_curve.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/pr_curve.png")
    plt.close()

    # Example 4: Classification Report
    print("\n4. Full Classification Report")
    print("-" * 70)
    fig = plot_classification_report(
        y_true_binary, y_pred_binary, y_score_binary,
        target_names=['Negative', 'Positive']
    )
    plt.savefig('/tmp/classification_report.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/classification_report.png")
    plt.close()

    # Example 5: Predictions Plot
    print("\n5. Actual vs Predicted")
    print("-" * 70)
    fig = plot_predictions(y_true_reg, y_pred_reg)
    plt.savefig('/tmp/predictions.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/predictions.png")
    plt.close()

    # Example 6: Residuals
    print("\n6. Residual Analysis")
    print("-" * 70)
    fig = plot_residuals(y_true_reg, y_pred_reg)
    plt.savefig('/tmp/residuals.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/residuals.png")
    plt.close()

    # Example 7: Regression Report
    print("\n7. Full Regression Report")
    print("-" * 70)
    fig = plot_regression_report(y_true_reg, y_pred_reg)
    plt.savefig('/tmp/regression_report.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/regression_report.png")
    plt.close()

    # Example 8: Learning Curve
    print("\n8. Learning Curve")
    print("-" * 70)
    train_scores = [0.6, 0.7, 0.75, 0.78, 0.8, 0.82, 0.83, 0.84, 0.845, 0.85]
    val_scores = [0.58, 0.68, 0.72, 0.74, 0.75, 0.755, 0.76, 0.76, 0.755, 0.75]

    fig = plot_learning_curve(
        train_scores, val_scores,
        metric_name='Accuracy',
        title='Model Training Progress'
    )
    plt.savefig('/tmp/learning_curve.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/learning_curve.png")
    plt.close()

    # Example 9: Multiple Model Comparison
    print("\n9. Multiple Model Comparison")
    print("-" * 70)
    curves = {
        'Model A': (
            [0.6, 0.7, 0.75, 0.78, 0.8, 0.82, 0.83, 0.84, 0.845, 0.85],
            [0.58, 0.68, 0.72, 0.74, 0.75, 0.755, 0.76, 0.76, 0.755, 0.75]
        ),
        'Model B': (
            [0.55, 0.65, 0.72, 0.76, 0.79, 0.81, 0.825, 0.835, 0.84, 0.845],
            [0.54, 0.63, 0.70, 0.73, 0.755, 0.765, 0.77, 0.77, 0.765, 0.76]
        ),
        'Model C': (
            [0.5, 0.6, 0.68, 0.73, 0.77, 0.8, 0.82, 0.83, 0.835, 0.84],
            [0.49, 0.58, 0.66, 0.71, 0.74, 0.76, 0.77, 0.775, 0.775, 0.77]
        )
    }

    fig = plot_multiple_learning_curves(
        curves,
        metric_name='Accuracy',
        title='Model Comparison: Training Progress'
    )
    plt.savefig('/tmp/model_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/model_comparison.png")
    plt.close()

    print("\n" + "=" * 70)
    print("All visualization examples completed!")
    print("Check /tmp/ for generated plots.")
    print("=" * 70)
