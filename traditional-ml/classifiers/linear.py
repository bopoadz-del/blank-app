"""
Linear Classification Models
Logistic Regression with various regularization options
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogisticRegressionWrapper:
    """
    Logistic Regression Classifier

    Linear model for binary and multiclass classification.
    Uses logistic function to model probability of classes.

    Best for:
    - Linear decision boundaries
    - Probabilistic predictions
    - Baseline models
    - When interpretability is crucial
    - High-dimensional sparse data
    """

    def __init__(
        self,
        penalty: str = 'l2',
        C: float = 1.0,
        solver: str = 'lbfgs',
        max_iter: int = 100,
        multi_class: str = 'auto',
        class_weight: Optional[Union[str, dict]] = None,
        random_state: int = 42,
        n_jobs: Optional[int] = None,
        verbose: int = 0
    ):
        """
        Initialize Logistic Regression

        Args:
            penalty: Regularization norm ('l1', 'l2', 'elasticnet', 'none')
            C: Inverse of regularization strength (smaller = stronger)
            solver: Optimization algorithm ('lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga')
            max_iter: Maximum number of iterations
            multi_class: Multiclass strategy ('ovr', 'multinomial', 'auto')
            class_weight: Weights for classes ('balanced' or dict)
            random_state: Random seed
            n_jobs: Number of CPU cores to use
            verbose: Verbosity level
        """
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Create model
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose
        )

        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        logger.info("Training Logistic Regression...")
        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.classes_ = self.model.classes_
        logger.info("Training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        return self.model.predict_proba(X)

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict log probabilities"""
        return self.model.predict_log_proba(X)

    def get_coefficients(self) -> np.ndarray:
        """Get model coefficients"""
        if self.coef_ is None:
            raise ValueError("Model not fitted yet")
        return self.coef_

    def get_intercept(self) -> np.ndarray:
        """Get model intercept"""
        if self.intercept_ is None:
            raise ValueError("Model not fitted yet")
        return self.intercept_

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
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler

    # Create dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("=" * 70)
    print("Logistic Regression Test")
    print("=" * 70)

    # Test L2 regularization
    print("\n1. L2 Regularization (Ridge)")
    print("-" * 70)
    lr_l2 = LogisticRegressionWrapper(penalty='l2', C=1.0)
    lr_l2.fit(X_train_scaled, y_train)
    y_pred = lr_l2.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Coefficients shape: {lr_l2.get_coefficients().shape}")

    # Test L1 regularization
    print("\n2. L1 Regularization (Lasso)")
    print("-" * 70)
    lr_l1 = LogisticRegressionWrapper(penalty='l1', solver='liblinear', C=1.0)
    lr_l1.fit(X_train_scaled, y_train)
    y_pred = lr_l1.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Non-zero coefficients: {np.sum(np.abs(lr_l1.get_coefficients()) > 0.01)}")

    # Test no regularization
    print("\n3. No Regularization")
    print("-" * 70)
    lr_none = LogisticRegressionWrapper(penalty='none', solver='lbfgs')
    lr_none.fit(X_train_scaled, y_train)
    y_pred = lr_none.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\n" + "=" * 70)
    print("Logistic Regression tested successfully!")
