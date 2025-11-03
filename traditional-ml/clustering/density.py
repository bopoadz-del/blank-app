"""
Density-based Clustering
DBSCAN and related algorithms
"""

import numpy as np
from sklearn.cluster import DBSCAN
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DBSCANClustering:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

    Groups together points that are closely packed.
    Marks points in low-density regions as outliers.

    Best for:
    - Arbitrary-shaped clusters
    - Outlier detection
    - When number of clusters is unknown
    - Clusters of varying densities

    Cons:
    - Sensitive to eps and min_samples parameters
    - Struggles with varying density clusters
    - Not deterministic with border points
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = 'euclidean',
        algorithm: str = 'auto',
        leaf_size: int = 30,
        n_jobs: Optional[int] = None
    ):
        """
        Initialize DBSCAN

        Args:
            eps: Maximum distance between two samples for one to be in neighborhood
            min_samples: Minimum samples in neighborhood for core point
            metric: Distance metric
            algorithm: Algorithm to compute pointwise distances
            leaf_size: Leaf size for BallTree or KDTree
            n_jobs: Number of parallel jobs
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs

        # Create model
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs
        )

        self.labels_ = None
        self.core_sample_indices_ = None

    def fit(self, X: np.ndarray):
        """Fit the model"""
        logger.info(f"Fitting DBSCAN (eps={self.eps}, min_samples={self.min_samples})...")
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.core_sample_indices_ = self.model.core_sample_indices_

        # Count clusters (excluding noise points labeled as -1)
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)

        logger.info(f"Clustering complete. Clusters: {n_clusters}, Noise points: {n_noise}")
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict"""
        self.fit(X)
        return self.labels_

    def get_labels(self) -> np.ndarray:
        """Get cluster labels"""
        if self.labels_ is None:
            raise ValueError("Model not fitted yet")
        return self.labels_

    def get_core_samples(self) -> np.ndarray:
        """Get indices of core samples"""
        if self.core_sample_indices_ is None:
            raise ValueError("Model not fitted yet")
        return self.core_sample_indices_

    def get_n_clusters(self) -> int:
        """Get number of clusters (excluding noise)"""
        if self.labels_ is None:
            raise ValueError("Model not fitted yet")
        return len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)

    def get_n_noise(self) -> int:
        """Get number of noise points"""
        if self.labels_ is None:
            raise ValueError("Model not fitted yet")
        return list(self.labels_).count(-1)


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_moons
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    # Create dataset with non-spherical clusters
    X, y_true = make_moons(n_samples=1000, noise=0.05, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("=" * 70)
    print("DBSCAN Clustering Test")
    print("=" * 70)

    # Test DBSCAN with different parameters
    print("\n1. DBSCAN (eps=0.3, min_samples=5)")
    print("-" * 70)
    dbscan1 = DBSCANClustering(eps=0.3, min_samples=5)
    labels = dbscan1.fit_predict(X_scaled)
    print(f"Number of clusters: {dbscan1.get_n_clusters()}")
    print(f"Number of noise points: {dbscan1.get_n_noise()}")

    if dbscan1.get_n_clusters() > 1:
        # Only calculate silhouette if we have non-noise points
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 0:
            score = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
            print(f"Silhouette Score (excl. noise): {score:.4f}")

    print("\n2. DBSCAN (eps=0.5, min_samples=10)")
    print("-" * 70)
    dbscan2 = DBSCANClustering(eps=0.5, min_samples=10)
    labels = dbscan2.fit_predict(X_scaled)
    print(f"Number of clusters: {dbscan2.get_n_clusters()}")
    print(f"Number of noise points: {dbscan2.get_n_noise()}")

    print("\n" + "=" * 70)
    print("DBSCAN tested successfully!")
