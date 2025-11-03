"""
Centroid-based Clustering
K-Means and variants
"""

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KMeansClustering:
    """
    K-Means Clustering

    Partitions data into K clusters by minimizing within-cluster variance.
    Assigns each point to nearest centroid.

    Best for:
    - Spherical clusters
    - Similar-sized clusters
    - Large datasets (use MiniBatch variant)
    - When K is known

    Cons:
    - Sensitive to initialization
    - Assumes spherical clusters
    - Need to specify K
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = 'k-means++',
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
        n_jobs: Optional[int] = None
    ):
        """
        Initialize K-Means Clustering

        Args:
            n_clusters: Number of clusters
            init: Initialization method ('k-means++', 'random')
            n_init: Number of initializations
            max_iter: Maximum iterations
            tol: Convergence tolerance
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Create model
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_jobs=n_jobs
        )

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X: np.ndarray):
        """Fit the model"""
        logger.info(f"Fitting K-Means with {self.n_clusters} clusters...")
        self.model.fit(X)
        self.cluster_centers_ = self.model.cluster_centers_
        self.labels_ = self.model.labels_
        self.inertia_ = self.model.inertia_
        logger.info(f"Clustering complete. Inertia: {self.inertia_:.2f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels"""
        return self.model.predict(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict"""
        logger.info(f"Fitting K-Means with {self.n_clusters} clusters...")
        labels = self.model.fit_predict(X)
        self.cluster_centers_ = self.model.cluster_centers_
        self.labels_ = labels
        self.inertia_ = self.model.inertia_
        logger.info(f"Clustering complete. Inertia: {self.inertia_:.2f}")
        return labels

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to cluster-distance space"""
        return self.model.transform(X)

    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers"""
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet")
        return self.cluster_centers_

    def get_inertia(self) -> float:
        """Get sum of squared distances to closest cluster center"""
        if self.inertia_ is None:
            raise ValueError("Model not fitted yet")
        return self.inertia_


class MiniBatchKMeansClustering:
    """
    Mini-Batch K-Means

    Faster variant of K-Means using mini-batches.
    Trade-off: slightly worse quality for much faster speed.

    Best for:
    - Very large datasets
    - When speed is priority
    - Streaming data
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = 'k-means++',
        max_iter: int = 100,
        batch_size: int = 1024,
        random_state: int = 42,
        n_init: int = 3
    ):
        """Initialize Mini-Batch K-Means"""
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_init = n_init

        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            batch_size=batch_size,
            random_state=random_state,
            n_init=n_init
        )

        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray):
        """Fit the model"""
        logger.info(f"Fitting Mini-Batch K-Means with {self.n_clusters} clusters...")
        self.model.fit(X)
        self.cluster_centers_ = self.model.cluster_centers_
        self.labels_ = self.model.labels_
        logger.info("Clustering complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels"""
        return self.model.predict(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict"""
        return self.model.fit_predict(X)


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score

    # Create dataset
    X, y_true = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=42)

    print("=" * 70)
    print("K-Means Clustering Test")
    print("=" * 70)

    # Test K-Means
    print("\n1. K-Means")
    print("-" * 70)
    kmeans = KMeansClustering(n_clusters=4)
    labels = kmeans.fit_predict(X)
    print(f"Inertia: {kmeans.get_inertia():.2f}")
    print(f"Silhouette Score: {silhouette_score(X, labels):.4f}")

    # Test Mini-Batch K-Means
    print("\n2. Mini-Batch K-Means")
    print("-" * 70)
    mb_kmeans = MiniBatchKMeansClustering(n_clusters=4)
    labels = mb_kmeans.fit_predict(X)
    print(f"Silhouette Score: {silhouette_score(X, labels):.4f}")

    print("\n" + "=" * 70)
    print("K-Means tested successfully!")
