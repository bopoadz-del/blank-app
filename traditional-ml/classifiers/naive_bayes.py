"""
Naive Bayes Classifiers
Gaussian, Multinomial, and Bernoulli variants
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GaussianNBWrapper:
    """
    Gaussian Naive Bayes

    Assumes features follow Gaussian (normal) distribution.
    Fast and works well with continuous features.

    Best for:
    - Continuous features
    - Real-valued data
    - Text classification with TF-IDF
    - Quick baselines
    """

    def __init__(self, var_smoothing: float = 1e-9):
        """
        Initialize Gaussian Naive Bayes

        Args:
            var_smoothing: Portion of largest variance added to variances
        """
        self.var_smoothing = var_smoothing

        # Create model
        self.model = GaussianNB(var_smoothing=var_smoothing)

        self.classes_ = None
        self.class_prior_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        logger.info("Training Gaussian Naive Bayes...")
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.class_prior_ = self.model.class_prior_
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

    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters"""
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """Set model parameters"""
        self.model.set_params(**params)
        return self


class MultinomialNBWrapper:
    """
    Multinomial Naive Bayes

    Assumes features are multinomially distributed (counts).
    Commonly used for text classification with word counts.

    Best for:
    - Count data (word counts, term frequencies)
    - Text classification
    - Document categorization
    - When features are discrete
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: Optional[np.ndarray] = None
    ):
        """
        Initialize Multinomial Naive Bayes

        Args:
            alpha: Additive (Laplace/Lidstone) smoothing parameter
            fit_prior: Whether to learn class prior probabilities
            class_prior: Prior probabilities of classes
        """
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

        # Create model
        self.model = MultinomialNB(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior
        )

        self.classes_ = None
        self.class_log_prior_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        logger.info("Training Multinomial Naive Bayes...")
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.class_log_prior_ = self.model.class_log_prior_
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

    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters"""
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """Set model parameters"""
        self.model.set_params(**params)
        return self


class BernoulliNBWrapper:
    """
    Bernoulli Naive Bayes

    Assumes binary features (0/1, True/False).
    Useful for binary/boolean features.

    Best for:
    - Binary features
    - Text classification with binary term occurrence
    - When presence/absence matters more than counts
    """

    def __init__(
        self,
        alpha: float = 1.0,
        binarize: Optional[float] = 0.0,
        fit_prior: bool = True,
        class_prior: Optional[np.ndarray] = None
    ):
        """
        Initialize Bernoulli Naive Bayes

        Args:
            alpha: Additive smoothing parameter
            binarize: Threshold for binarizing features (None = no binarization)
            fit_prior: Whether to learn class prior probabilities
            class_prior: Prior probabilities of classes
        """
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior

        # Create model
        self.model = BernoulliNB(
            alpha=alpha,
            binarize=binarize,
            fit_prior=fit_prior,
            class_prior=class_prior
        )

        self.classes_ = None
        self.class_log_prior_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        logger.info("Training Bernoulli Naive Bayes...")
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.class_log_prior_ = self.model.class_log_prior_
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

    print("=" * 70)
    print("Naive Bayes Classifiers Test")
    print("=" * 70)

    # Test Gaussian NB
    print("\n1. Gaussian Naive Bayes")
    print("-" * 70)
    gnb = GaussianNBWrapper()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Test Multinomial NB (need non-negative features)
    print("\n2. Multinomial Naive Bayes")
    print("-" * 70)
    # Make features non-negative
    X_train_pos = np.abs(X_train)
    X_test_pos = np.abs(X_test)

    mnb = MultinomialNBWrapper(alpha=1.0)
    mnb.fit(X_train_pos, y_train)
    y_pred = mnb.predict(X_test_pos)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Test Bernoulli NB
    print("\n3. Bernoulli Naive Bayes")
    print("-" * 70)
    # Binarize features
    bnb = BernoulliNBWrapper(alpha=1.0, binarize=0.0)
    bnb.fit(X_train, y_train)
    y_pred = bnb.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\n" + "=" * 70)
    print("Naive Bayes classifiers tested successfully!")
