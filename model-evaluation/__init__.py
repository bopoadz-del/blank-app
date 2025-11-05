"""
Model Evaluation Framework

A comprehensive framework for evaluating machine learning models with:
- Classification metrics (accuracy, precision, recall, F1, ROC, PR)
- Regression metrics (MSE, RMSE, MAE, R², MAPE)
- Confusion matrix with normalization
- ROC and Precision-Recall curves
- Comprehensive visualizations
- Custom metrics support

Author: ML Framework Team
Version: 1.0.0
"""

# Classification metrics
from .metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# Regression metrics
from .metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    regression_report
)

# Custom metrics
from .metrics import (
    CustomMetric,
    make_scorer
)

# Visualizations
from .visualization import (
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

__all__ = [
    # Classification metrics
    'confusion_matrix',
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'classification_report',
    'roc_curve',
    'roc_auc_score',
    'precision_recall_curve',
    'average_precision_score',

    # Regression metrics
    'mean_squared_error',
    'root_mean_squared_error',
    'mean_absolute_error',
    'r2_score',
    'mean_absolute_percentage_error',
    'regression_report',

    # Custom metrics
    'CustomMetric',
    'make_scorer',

    # Visualizations
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_classification_report',
    'plot_predictions',
    'plot_residuals',
    'plot_regression_report',
    'plot_learning_curve',
    'plot_multiple_learning_curves',
]

__version__ = '1.0.0'
__author__ = 'ML Framework Team'
