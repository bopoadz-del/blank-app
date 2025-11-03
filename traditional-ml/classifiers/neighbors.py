"""
K-Nearest Neighbors Classifier
Instance-based learning algorithm
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing import Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier

    Non-parametric instance-based learning.
    Classifies based on majority vote of k nearest neighbors.

    Best for:
    - Small datasets
    - Multi-class classification
    - Non-linear decision boundaries
    - When training speed is priority

    Cons:
    - Slow prediction on large datasets
    - Sensitive to feature scaling
    - Curse of dimensionality
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        algorithm: str = 'auto',
        metric: str = 'minkowski',
        p: int = 2,
        n_jobs: Optional[int] = None
    ):
        """
        Initialize KNN Classifier

        Args:
            n_neighbors: Number of neighbors to use
            weights: Weight function ('uniform' or 'distance')
            algorithm: Algorithm to compute nearest neighbors
                      ('auto', 'ball_tree', 'kd_tree', 'brute')
            metric: Distance metric ('euclidean', 'manhattan', 'minkowski', etc.)
            p: Power parameter for Minkowski metric (p=1: Manhattan, p=2: Euclidean)
            n_jobs: Number of parallel jobs
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.n_jobs = n_jobs

        # Create model
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            metric=metric,
            p=p,
            n_jobs=n_jobs
        )

        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model (just stores the data)"""
        logger.info(f"Training KNN with k={self.n_neighbors}...")
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        logger.info("Training complete (data stored)")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        return self.model.predict_proba(X)

    def kneighbors(
        self,
        X: Optional[np.ndarray] = None,
        n_neighbors: Optional[int] = None,
        return_distance: bool = True
    ):
        """
        Find k-neighbors of a point

        Args:
            X: Query points (None = use training data)
            n_neighbors: Number of neighbors (None = use self.n_neighbors)
            return_distance: Whether to return distances

        Returns:
            Distances and indices of neighbors
        """
        return self.model.kneighbors(
            X=X,
            n_neighbors=n_neighbors,
            return_distance=return_distance
        )

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

    # Scale features (important for KNN!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("=" * 70)
    print("KNN Classifiers Test")
    print("=" * 70)

    # Test different k values
    for k in [3, 5, 7, 11]:
        print(f"\nk = {k}")
        print("-" * 70)
        knn = KNNClassifier(n_neighbors=k, weights='uniform')
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Test distance weighting
    print("\nDistance Weighting (k=5)")
    print("-" * 70)
    knn_dist = KNNClassifier(n_neighbors=5, weights='distance')
    knn_dist.fit(X_train_scaled, y_train)
    y_pred = knn_dist.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Test Manhattan distance
    print("\nManhattan Distance (p=1, k=5)")
    print("-" * 70)
    knn_manhattan = KNNClassifier(n_neighbors=5, metric='manhattan')
    knn_manhattan.fit(X_train_scaled, y_train)
    y_pred = knn_manhattan.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\n" + "=" * 70)
    print("KNN tested successfully!")
