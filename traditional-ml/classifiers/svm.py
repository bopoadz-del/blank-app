"""
Support Vector Machine Classifiers
SVM with various kernels for classification
"""

import numpy as np
from sklearn.svm import SVC
from typing import Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SVMClassifier:
    """
    Support Vector Machine Classifier

    Finds optimal hyperplane to separate classes.
    Effective in high-dimensional spaces.

    Best for:
    - Binary classification
    - High-dimensional data
    - Clear margin of separation
    - Non-linear decision boundaries (with kernel trick)

    Important: Always scale features before using SVM!
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: Union[str, float] = 'scale',
        degree: int = 3,
        class_weight: Optional[Union[str, dict]] = None,
        probability: bool = True,
        random_state: int = 42,
        max_iter: int = -1,
        verbose: bool = False
    ):
        """
        Initialize SVM Classifier

        Args:
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient ('scale', 'auto', or float)
            degree: Degree for polynomial kernel
            class_weight: Weights for classes ('balanced' or dict)
            probability: Enable probability estimates
            random_state: Random seed
            max_iter: Maximum iterations (-1 = no limit)
            verbose: Verbose output
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.class_weight = class_weight
        self.probability = probability
        self.random_state = random_state
        self.max_iter = max_iter
        self.verbose = verbose

        # Create model
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            class_weight=class_weight,
            probability=probability,
            random_state=random_state,
            max_iter=max_iter,
            verbose=verbose
        )

        self.support_vectors_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        logger.info(f"Training SVM with {self.kernel} kernel...")
        self.model.fit(X, y)
        self.support_vectors_ = self.model.support_vectors_
        self.classes_ = self.model.classes_
        logger.info(f"Training complete. Support vectors: {len(self.support_vectors_)}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.probability:
            raise ValueError("Probability estimates not enabled. Set probability=True")
        return self.model.predict_proba(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get decision function values"""
        return self.model.decision_function(X)

    def get_support_vectors(self) -> np.ndarray:
        """Get support vectors"""
        if self.support_vectors_ is None:
            raise ValueError("Model not fitted yet")
        return self.support_vectors_

    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters"""
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """Set model parameters"""
        self.model.set_params(**params)
        return self


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    # Create dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (VERY IMPORTANT for SVM!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("=" * 70)
    print("SVM Classifiers Test")
    print("=" * 70)

    # Test Linear kernel
    print("\n1. Linear Kernel")
    print("-" * 70)
    svm_linear = SVMClassifier(kernel='linear', C=1.0)
    svm_linear.fit(X_train_scaled, y_train)
    y_pred = svm_linear.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Support vectors: {len(svm_linear.get_support_vectors())}")

    # Test RBF kernel
    print("\n2. RBF Kernel")
    print("-" * 70)
    svm_rbf = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
    svm_rbf.fit(X_train_scaled, y_train)
    y_pred = svm_rbf.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Support vectors: {len(svm_rbf.get_support_vectors())}")

    # Test Polynomial kernel
    print("\n3. Polynomial Kernel (degree=3)")
    print("-" * 70)
    svm_poly = SVMClassifier(kernel='poly', degree=3, C=1.0)
    svm_poly.fit(X_train_scaled, y_train)
    y_pred = svm_poly.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Support vectors: {len(svm_poly.get_support_vectors())}")

    print("\n" + "=" * 70)
    print("SVM tested successfully!")
