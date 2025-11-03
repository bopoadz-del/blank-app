"""
Model Evaluation Metrics

Comprehensive metrics for classification and regression tasks including:
- Classification: Confusion matrix, Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Regression: R², MSE, MAE, RMSE, MAPE
- Custom metrics framework

Author: ML Framework Team
"""

import numpy as np
from typing import Union, Optional, Callable, Dict, List, Tuple
import warnings


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix.

    Parameters:
    -----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        List of label values. If None, uses sorted unique values.
    normalize : str, optional
        'true', 'pred', 'all' for normalization. None for counts.

    Returns:
    --------
    cm : np.ndarray
        Confusion matrix of shape (n_classes, n_classes).

    Example:
    --------
    >>> y_true = [0, 1, 0, 1, 0, 1]
    >>> y_pred = [0, 1, 1, 1, 0, 0]
    >>> cm = confusion_matrix(y_true, y_pred)
    >>> print(cm)
    [[2 1]
     [1 2]]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)

    n_labels = len(labels)
    label_to_ind = {label: i for i, label in enumerate(labels)}

    # Build confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_ind and pred_label in label_to_ind:
            cm[label_to_ind[true_label], label_to_ind[pred_label]] += 1

    # Normalize if requested
    if normalize == 'true':
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm.astype(float) / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm.astype(float) / cm.sum()

    return cm


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy classification score.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'binary',
    zero_division: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Compute precision score.

    Precision = TP / (TP + FP)

    Parameters:
    -----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    average : str, default='binary'
        'binary', 'micro', 'macro', 'weighted', or None.
    zero_division : float, default=0.0
        Value to return when there is a zero division.

    Returns:
    --------
    precision : float or array
        Precision score(s).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else zero_division

    elif average == 'micro':
        cm = confusion_matrix(y_true, y_pred)
        tp = np.diag(cm).sum()
        fp = cm.sum(axis=0) - np.diag(cm)
        return tp / (tp + fp.sum()) if (tp + fp.sum()) > 0 else zero_division

    elif average in ['macro', 'weighted']:
        labels = np.unique(y_true)
        precisions = []
        weights = []

        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fp = np.sum((y_true != label) & (y_pred == label))

            if (tp + fp) > 0:
                precisions.append(tp / (tp + fp))
            else:
                precisions.append(zero_division)

            if average == 'weighted':
                weights.append(np.sum(y_true == label))

        if average == 'macro':
            return np.mean(precisions)
        else:
            return np.average(precisions, weights=weights)

    elif average is None:
        labels = np.unique(y_true)
        precisions = []

        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fp = np.sum((y_true != label) & (y_pred == label))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else zero_division)

        return np.array(precisions)

    else:
        raise ValueError(f"Unknown average type: {average}")


def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'binary',
    zero_division: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Compute recall score.

    Recall = TP / (TP + FN)

    Parameters same as precision_score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if average == 'binary':
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else zero_division

    elif average == 'micro':
        cm = confusion_matrix(y_true, y_pred)
        tp = np.diag(cm).sum()
        fn = cm.sum(axis=1) - np.diag(cm)
        return tp / (tp + fn.sum()) if (tp + fn.sum()) > 0 else zero_division

    elif average in ['macro', 'weighted']:
        labels = np.unique(y_true)
        recalls = []
        weights = []

        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fn = np.sum((y_true == label) & (y_pred != label))

            if (tp + fn) > 0:
                recalls.append(tp / (tp + fn))
            else:
                recalls.append(zero_division)

            if average == 'weighted':
                weights.append(np.sum(y_true == label))

        if average == 'macro':
            return np.mean(recalls)
        else:
            return np.average(recalls, weights=weights)

    elif average is None:
        labels = np.unique(y_true)
        recalls = []

        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fn = np.sum((y_true == label) & (y_pred != label))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else zero_division)

        return np.array(recalls)

    else:
        raise ValueError(f"Unknown average type: {average}")


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'binary',
    zero_division: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Compute F1 score.

    F1 = 2 * (precision * recall) / (precision + recall)

    Parameters same as precision_score.

    Example:
    --------
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> f1 = f1_score(y_true, y_pred)
    >>> print(f"F1 Score: {f1:.4f}")
    """
    precision = precision_score(y_true, y_pred, average=average, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, average=average, zero_division=zero_division)

    if isinstance(precision, np.ndarray):
        f1 = np.zeros_like(precision)
        mask = (precision + recall) > 0
        f1[mask] = 2 * (precision[mask] * recall[mask]) / (precision[mask] + recall[mask])
        f1[~mask] = zero_division
        return f1
    else:
        if (precision + recall) > 0:
            return 2 * (precision * recall) / (precision + recall)
        return zero_division


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
    target_names: Optional[List[str]] = None,
    digits: int = 4
) -> str:
    """
    Generate text report showing classification metrics.

    Parameters:
    -----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        Labels to include in report.
    target_names : list of str, optional
        Display names for labels.
    digits : int, default=4
        Number of decimal places.

    Returns:
    --------
    report : str
        Text summary of classification metrics.

    Example:
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> model = RandomForestClassifier()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>>
    >>> print(classification_report(y_test, y_pred))
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(y_true)

    if target_names is None:
        target_names = [str(label) for label in labels]

    # Compute metrics for each class
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Support (number of samples per class)
    supports = [np.sum(y_true == label) for label in labels]

    # Build report
    width = max(len(name) for name in target_names)
    width = max(width, len('weighted avg'))

    headers = ['precision', 'recall', 'f1-score', 'support']
    head_fmt = '{:>{width}s} ' + ' {:>9s}' * len(headers)
    report = head_fmt.format('', *headers, width=width)
    report += '\n\n'

    row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'

    # Per-class metrics
    for i, label in enumerate(labels):
        report += row_fmt.format(
            target_names[i],
            precisions[i],
            recalls[i],
            f1s[i],
            supports[i],
            width=width,
            digits=digits
        )

    report += '\n'

    # Average metrics
    report += row_fmt.format(
        'accuracy',
        '',
        '',
        accuracy_score(y_true, y_pred),
        np.sum(supports),
        width=width,
        digits=digits
    )

    # Macro average
    report += row_fmt.format(
        'macro avg',
        np.mean(precisions),
        np.mean(recalls),
        np.mean(f1s),
        np.sum(supports),
        width=width,
        digits=digits
    )

    # Weighted average
    report += row_fmt.format(
        'weighted avg',
        np.average(precisions, weights=supports),
        np.average(recalls, weights=supports),
        np.average(f1s, weights=supports),
        np.sum(supports),
        width=width,
        digits=digits
    )

    return report


def roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pos_label: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Receiver Operating Characteristic (ROC) curve.

    Parameters:
    -----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Target scores (probability estimates).
    pos_label : int, default=1
        Label of positive class.

    Returns:
    --------
    fpr : np.ndarray
        False Positive Rate.
    tpr : np.ndarray
        True Positive Rate.
    thresholds : np.ndarray
        Thresholds used.

    Example:
    --------
    >>> y_true = [0, 0, 1, 1]
    >>> y_scores = [0.1, 0.4, 0.35, 0.8]
    >>> fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Convert to binary
    y_true = (y_true == pos_label).astype(int)

    # Sort by score
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true = y_true[desc_score_indices]
    y_score = y_score[desc_score_indices]

    # Get thresholds
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    thresholds = y_score[threshold_idxs]

    # Compute TPR and FPR
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    # Add (0, 0) point
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    # Compute rates
    if fps[-1] == 0:
        fpr = np.zeros_like(fps, dtype=float)
    else:
        fpr = fps / fps[-1]

    if tps[-1] == 0:
        tpr = np.zeros_like(tps, dtype=float)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve (AUC).

    Parameters:
    -----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Target scores.

    Returns:
    --------
    auc : float
        Area under ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)

    return auc


def precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pos_label: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall curve.

    Parameters:
    -----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Target scores.
    pos_label : int, default=1
        Label of positive class.

    Returns:
    --------
    precision : np.ndarray
        Precision values.
    recall : np.ndarray
        Recall values.
    thresholds : np.ndarray
        Thresholds.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Convert to binary
    y_true = (y_true == pos_label).astype(int)

    # Sort by score
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true = y_true[desc_score_indices]
    y_score = y_score[desc_score_indices]

    # Get thresholds
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    thresholds = y_score[threshold_idxs]

    # Compute precision and recall
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    precision = tps / (tps + fps)
    recall = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)

    # Add endpoint
    precision = np.r_[1, precision]
    recall = np.r_[0, recall]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    return precision, recall, thresholds


def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute average precision (AP) from prediction scores.

    AP = Σ(R_n - R_{n-1}) * P_n

    Parameters:
    -----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Target scores.

    Returns:
    --------
    ap : float
        Average precision score.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    # Compute AP using step function
    ap = -np.sum(np.diff(recall) * np.array(precision)[:-1])

    return ap


# ============================================================================
# REGRESSION METRICS
# ============================================================================

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error (MSE).

    MSE = (1/n) * Σ(y_true - y_pred)²
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error (RMSE).

    RMSE = √MSE
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE).

    MAE = (1/n) * Σ|y_true - y_pred|
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² (coefficient of determination) regression score.

    R² = 1 - (SS_res / SS_tot)

    where:
    - SS_res = Σ(y_true - y_pred)²
    - SS_tot = Σ(y_true - mean(y_true))²

    Best possible score is 1.0. Can be negative.

    Example:
    --------
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2 = r2_score(y_true, y_pred)
    >>> print(f"R² Score: {r2:.4f}")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / (ss_tot + 1e-10))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    MAPE = (100/n) * Σ|(y_true - y_pred) / y_true|
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Generate comprehensive regression metrics report.

    Parameters:
    -----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
    metrics : list of str, optional
        Metrics to include. If None, includes all.

    Returns:
    --------
    report : dict
        Dictionary of metric names and values.

    Example:
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LinearRegression
    >>>
    >>> X, y = make_regression(n_samples=1000, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> model = LinearRegression()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>>
    >>> report = regression_report(y_test, y_pred)
    >>> for metric, value in report.items():
    ...     print(f"{metric}: {value:.4f}")
    """
    if metrics is None:
        metrics = ['mse', 'rmse', 'mae', 'r2', 'mape']

    report = {}

    if 'mse' in metrics:
        report['MSE'] = mean_squared_error(y_true, y_pred)

    if 'rmse' in metrics:
        report['RMSE'] = root_mean_squared_error(y_true, y_pred)

    if 'mae' in metrics:
        report['MAE'] = mean_absolute_error(y_true, y_pred)

    if 'r2' in metrics:
        report['R²'] = r2_score(y_true, y_pred)

    if 'mape' in metrics:
        try:
            report['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
        except:
            report['MAPE'] = np.nan

    return report


# ============================================================================
# CUSTOM METRICS
# ============================================================================

class CustomMetric:
    """
    Base class for custom metrics.

    Example:
    --------
    >>> class MyMetric(CustomMetric):
    ...     def __call__(self, y_true, y_pred):
    ...         return np.mean(np.abs(y_true - y_pred) ** 3)
    >>>
    >>> metric = MyMetric(name='cube_error')
    >>> score = metric(y_true, y_pred)
    """

    def __init__(self, name: str = 'custom_metric'):
        self.name = name

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute metric. Override this method."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


def make_scorer(
    score_func: Callable,
    greater_is_better: bool = True,
    **kwargs
) -> Callable:
    """
    Make a scorer from a performance metric or loss function.

    Parameters:
    -----------
    score_func : callable
        Score function with signature score_func(y_true, y_pred, **kwargs).
    greater_is_better : bool, default=True
        Whether higher score is better.
    **kwargs : dict
        Additional arguments to pass to score_func.

    Returns:
    --------
    scorer : callable
        Scorer function.

    Example:
    --------
    >>> def my_metric(y_true, y_pred):
    ...     return np.mean((y_true - y_pred) ** 2)
    >>>
    >>> scorer = make_scorer(my_metric, greater_is_better=False)
    >>> score = scorer(y_true, y_pred)
    """
    def scorer(y_true, y_pred):
        score = score_func(y_true, y_pred, **kwargs)
        return score if greater_is_better else -score

    return scorer


if __name__ == "__main__":
    print("=" * 70)
    print("MODEL EVALUATION METRICS EXAMPLES")
    print("=" * 70)

    # Example 1: Classification Metrics
    print("\n1. Classification Metrics")
    print("-" * 70)

    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Example 2: Classification Report
    print("\n2. Classification Report")
    print("-" * 70)
    report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'])
    print(report)

    # Example 3: ROC Curve
    print("\n3. ROC Curve and AUC")
    print("-" * 70)

    y_scores = np.array([0.1, 0.9, 0.6, 0.2, 0.8, 0.95, 0.1, 0.4, 0.85, 0.15])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    print(f"ROC AUC: {auc:.4f}")

    # Example 4: Regression Metrics
    print("\n4. Regression Metrics")
    print("-" * 70)

    y_true_reg = np.array([3.0, -0.5, 2.0, 7.0, 4.2])
    y_pred_reg = np.array([2.5, 0.0, 2.1, 7.8, 4.0])

    mse = mean_squared_error(y_true_reg, y_pred_reg)
    rmse = root_mean_squared_error(y_true_reg, y_pred_reg)
    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    r2 = r2_score(y_true_reg, y_pred_reg)

    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    # Example 5: Regression Report
    print("\n5. Regression Report")
    print("-" * 70)
    report = regression_report(y_true_reg, y_pred_reg)
    for metric, value in report.items():
        print(f"{metric}: {value:.4f}")

    # Example 6: Custom Metric
    print("\n6. Custom Metric")
    print("-" * 70)

    def weighted_accuracy(y_true, y_pred, weights):
        return np.sum(weights * (y_true == y_pred)) / np.sum(weights)

    weights = np.array([1, 2, 1, 1, 2, 2, 1, 1, 2, 1])
    scorer = make_scorer(lambda y_t, y_p: weighted_accuracy(y_t, y_p, weights))
    score = scorer(y_true, y_pred)
    print(f"Weighted Accuracy: {score:.4f}")

    print("\n" + "=" * 70)
    print("All metric examples completed!")
    print("=" * 70)
