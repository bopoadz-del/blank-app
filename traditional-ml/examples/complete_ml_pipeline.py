"""
Complete Machine Learning Pipeline Example
Demonstrates end-to-end ML workflow with traditional algorithms
"""

import sys
sys.path.append('..')

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from classifiers.ensemble import RandomForestClassifierWrapper, XGBoostClassifierWrapper
from classifiers.linear import LogisticRegressionWrapper
from classifiers.svm import SVMClassifier
from classifiers.neighbors import KNNClassifier
from feature_selection.statistical import UnivariateSelector
from dimensionality_reduction.linear import PCAReducer

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Complete ML pipeline demonstration"""

    logger.info("=" * 70)
    logger.info("Complete Machine Learning Pipeline")
    logger.info("=" * 70)

    # 1. Generate Dataset
    logger.info("\n1. Generating Dataset")
    logger.info("-" * 70)

    X, y = make_classification(
        n_samples=2000,
        n_features=50,
        n_informative=20,
        n_redundant=10,
        n_repeated=0,
        n_classes=2,
        class_sep=0.8,
        random_state=42
    )

    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Classes: {np.unique(y)}")
    logger.info(f"Class distribution: {np.bincount(y)}")

    # 2. Train-Test Split
    logger.info("\n2. Train-Test Split")
    logger.info("-" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    # 3. Feature Selection
    logger.info("\n3. Feature Selection")
    logger.info("-" * 70)

    selector = UnivariateSelector(score_func='f_classif', mode='k_best', k=20)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    logger.info(f"Original features: {X_train.shape[1]}")
    logger.info(f"Selected features: {X_train_selected.shape[1]}")

    # 4. Dimensionality Reduction (optional)
    logger.info("\n4. Dimensionality Reduction")
    logger.info("-" * 70)

    pca = PCAReducer(n_components=0.95)  # Keep 95% of variance
    X_train_pca = pca.fit_transform(X_train_selected)
    X_test_pca = pca.transform(X_test_selected)

    logger.info(f"Components after PCA: {X_train_pca.shape[1]}")
    logger.info(f"Variance explained: {np.sum(pca.get_explained_variance_ratio()):.4f}")

    # 5. Feature Scaling
    logger.info("\n5. Feature Scaling")
    logger.info("-" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    logger.info("Features standardized (zero mean, unit variance)")

    # 6. Model Training and Evaluation
    logger.info("\n6. Model Training and Evaluation")
    logger.info("=" * 70)

    models = {
        'Random Forest': RandomForestClassifierWrapper(n_estimators=100),
        'Logistic Regression': LogisticRegressionWrapper(),
        'SVM (RBF)': SVMClassifier(kernel='rbf', C=1.0),
        'KNN': KNNClassifier(n_neighbors=5)
    }

    # Try XGBoost if available
    try:
        models['XGBoost'] = XGBoostClassifierWrapper(n_estimators=100)
    except ImportError:
        logger.warning("XGBoost not available")

    results = {}

    for name, model in models.items():
        logger.info(f"\n{name}")
        logger.info("-" * 70)

        # Train
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
        logger.info(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")

    # 7. Model Comparison
    logger.info("\n7. Model Comparison")
    logger.info("=" * 70)

    logger.info(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    logger.info("-" * 70)

    for name, metrics in results.items():
        logger.info(
            f"{name:<25} "
            f"{metrics['accuracy']:<10.4f} "
            f"{metrics['precision']:<10.4f} "
            f"{metrics['recall']:<10.4f} "
            f"{metrics['f1']:<10.4f}"
        )

    # 8. Best Model Selection
    logger.info("\n8. Best Model Selection")
    logger.info("-" * 70)

    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    logger.info(f"Best model: {best_model[0]}")
    logger.info(f"Best accuracy: {best_model[1]['accuracy']:.4f}")

    # 9. Cross-Validation
    logger.info("\n9. Cross-Validation (Best Model)")
    logger.info("-" * 70)

    best_model_instance = models[best_model[0]]

    # Recreate model to ensure it's unfitted
    if best_model[0] == 'Random Forest':
        cv_model = RandomForestClassifierWrapper(n_estimators=100).model
    elif best_model[0] == 'XGBoost':
        cv_model = XGBoostClassifierWrapper(n_estimators=100).model
    elif best_model[0] == 'Logistic Regression':
        cv_model = LogisticRegressionWrapper().model
    elif best_model[0] == 'SVM (RBF)':
        cv_model = SVMClassifier(kernel='rbf', C=1.0).model
    else:
        cv_model = KNNClassifier(n_neighbors=5).model

    cv_scores = cross_val_score(cv_model, X_train_scaled, y_train, cv=5, scoring='accuracy')

    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    logger.info("\n" + "=" * 70)
    logger.info("Pipeline Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
