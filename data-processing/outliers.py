"""
Outlier Detection Module

Comprehensive techniques for detecting outliers:
- Statistical: IQR, Z-score, Modified Z-score
- Distance-based: KNN, LOF
- Density-based: DBSCAN
- Isolation-based: Isolation Forest
- Robust statistics: MAD, Tukey fences
- Multivariate: Mahalanobis distance, Elliptic Envelope

Author: ML Framework Team
"""

import numpy as np
from typing import Tuple, Optional, Literal, Union
from collections import Counter


# ============================================================================
# STATISTICAL METHODS
# ============================================================================

class IQRDetector:
    """
    Interquartile Range (IQR) based outlier detection.

    Outliers are values outside [Q1 - k*IQR, Q3 + k*IQR].
    """

    def __init__(self, k: float = 1.5):
        """
        Initialize IQR detector.

        Parameters:
        -----------
        k : float
            IQR multiplier (1.5 for outliers, 3.0 for extreme outliers).
        """
        self.k = k
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None
        self.lower_bound_ = None
        self.upper_bound_ = None

    def fit(self, X: np.ndarray) -> 'IQRDetector':
        """
        Compute IQR statistics.

        Parameters:
        -----------
        X : np.ndarray
            Data (n_samples, n_features).

        Returns:
        --------
        self : IQRDetector
        """
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_

        self.lower_bound_ = self.q1_ - self.k * self.iqr_
        self.upper_bound_ = self.q3_ + self.k * self.iqr_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outliers.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        labels : np.ndarray
            1 for inliers, -1 for outliers.
        """
        # Check if any feature is outside bounds
        outlier_mask = (X < self.lower_bound_) | (X > self.upper_bound_)
        is_outlier = np.any(outlier_mask, axis=1)

        return np.where(is_outlier, -1, 1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and predict in one step.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        labels : np.ndarray
            1 for inliers, -1 for outliers.
        """
        return self.fit(X).predict(X)


class ZScoreDetector:
    """
    Z-score based outlier detection.

    Outliers are values with |z-score| > threshold.
    """

    def __init__(self, threshold: float = 3.0):
        """
        Initialize Z-score detector.

        Parameters:
        -----------
        threshold : float
            Z-score threshold (typically 3.0).
        """
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> 'ZScoreDetector':
        """
        Compute mean and std.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        self : ZScoreDetector
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outliers.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        labels : np.ndarray
            1 for inliers, -1 for outliers.
        """
        # Compute Z-scores
        z_scores = np.abs((X - self.mean_) / (self.std_ + 1e-10))

        # Check if any feature has extreme Z-score
        is_outlier = np.any(z_scores > self.threshold, axis=1)

        return np.where(is_outlier, -1, 1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and predict in one step.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        labels : np.ndarray
            1 for inliers, -1 for outliers.
        """
        return self.fit(X).predict(X)


class ModifiedZScoreDetector:
    """
    Modified Z-score using MAD (Median Absolute Deviation).

    More robust to outliers than standard Z-score.
    """

    def __init__(self, threshold: float = 3.5):
        """
        Initialize modified Z-score detector.

        Parameters:
        -----------
        threshold : float
            Modified Z-score threshold (typically 3.5).
        """
        self.threshold = threshold
        self.median_ = None
        self.mad_ = None

    def fit(self, X: np.ndarray) -> 'ModifiedZScoreDetector':
        """
        Compute median and MAD.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        self : ModifiedZScoreDetector
        """
        self.median_ = np.median(X, axis=0)

        # MAD = median(|X - median(X)|)
        self.mad_ = np.median(np.abs(X - self.median_), axis=0)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outliers.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        labels : np.ndarray
            1 for inliers, -1 for outliers.
        """
        # Modified Z-score = 0.6745 * (X - median) / MAD
        modified_z_scores = np.abs(0.6745 * (X - self.median_) / (self.mad_ + 1e-10))

        is_outlier = np.any(modified_z_scores > self.threshold, axis=1)

        return np.where(is_outlier, -1, 1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and predict in one step.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        labels : np.ndarray
            1 for inliers, -1 for outliers.
        """
        return self.fit(X).predict(X)


# ============================================================================
# DISTANCE-BASED METHODS
# ============================================================================

class KNNOutlierDetector:
    """
    KNN-based outlier detection.

    Outliers have large average distance to k nearest neighbors.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        contamination: float = 0.1
    ):
        """
        Initialize KNN detector.

        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors.
        contamination : float
            Expected proportion of outliers.
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.X_fit_ = None
        self.threshold_ = None

    def fit(self, X: np.ndarray) -> 'KNNOutlierDetector':
        """
        Fit detector.

        Parameters:
        -----------
        X : np.ndarray
            Training data.

        Returns:
        --------
        self : KNNOutlierDetector
        """
        self.X_fit_ = X.copy()

        # Compute distances to k nearest neighbors
        distances = []
        for i in range(len(X)):
            # Compute distances to all other points
            dists = np.linalg.norm(X - X[i], axis=1)
            dists[i] = np.inf  # Exclude self

            # Get k nearest neighbors
            k_nearest_dists = np.sort(dists)[:self.n_neighbors]
            avg_dist = np.mean(k_nearest_dists)
            distances.append(avg_dist)

        distances = np.array(distances)

        # Threshold based on contamination
        self.threshold_ = np.percentile(distances, 100 * (1 - self.contamination))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outliers.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        labels : np.ndarray
            1 for inliers, -1 for outliers.
        """
        distances = []
        for i in range(len(X)):
            # Compute distances to all training points
            dists = np.linalg.norm(self.X_fit_ - X[i], axis=1)

            # Get k nearest neighbors
            k_nearest_dists = np.sort(dists)[:self.n_neighbors]
            avg_dist = np.mean(k_nearest_dists)
            distances.append(avg_dist)

        distances = np.array(distances)

        return np.where(distances > self.threshold_, -1, 1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and predict in one step.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        labels : np.ndarray
            1 for inliers, -1 for outliers.
        """
        return self.fit(X).predict(X)


class LocalOutlierFactor:
    """
    Local Outlier Factor (LOF).

    Measures local deviation of density of a sample with respect to its neighbors.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1
    ):
        """
        Initialize LOF detector.

        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors.
        contamination : float
            Expected proportion of outliers.
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.threshold_ = None

    def _compute_lof(self, X: np.ndarray) -> np.ndarray:
        """
        Compute LOF scores.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        lof_scores : np.ndarray
            LOF score for each sample.
        """
        n_samples = len(X)

        # Compute k-distance and k-neighbors for each point
        k_distances = np.zeros(n_samples)
        k_neighbors_list = []

        for i in range(n_samples):
            dists = np.linalg.norm(X - X[i], axis=1)
            sorted_indices = np.argsort(dists)[1:self.n_neighbors + 1]  # Exclude self

            k_neighbors_list.append(sorted_indices)
            k_distances[i] = dists[sorted_indices[-1]]

        # Compute reachability distance
        reach_dists = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    dist_ij = np.linalg.norm(X[i] - X[j])
                    reach_dists[i, j] = max(dist_ij, k_distances[j])

        # Compute local reachability density (LRD)
        lrd = np.zeros(n_samples)
        for i in range(n_samples):
            neighbors = k_neighbors_list[i]
            avg_reach_dist = np.mean([reach_dists[i, j] for j in neighbors])
            lrd[i] = 1.0 / (avg_reach_dist + 1e-10)

        # Compute LOF
        lof_scores = np.zeros(n_samples)
        for i in range(n_samples):
            neighbors = k_neighbors_list[i]
            lrd_ratio = np.mean([lrd[j] for j in neighbors]) / (lrd[i] + 1e-10)
            lof_scores[i] = lrd_ratio

        return lof_scores

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute LOF and predict outliers.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        labels : np.ndarray
            1 for inliers, -1 for outliers.
        """
        lof_scores = self._compute_lof(X)

        # Threshold based on contamination
        self.threshold_ = np.percentile(lof_scores, 100 * (1 - self.contamination))

        return np.where(lof_scores > self.threshold_, -1, 1)


# ============================================================================
# ISOLATION-BASED METHOD
# ============================================================================

class IsolationForest:
    """
    Isolation Forest for anomaly detection.

    Isolates anomalies by randomly partitioning the data space.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[int, float] = 256,
        contamination: float = 0.1,
        random_state: Optional[int] = None
    ):
        """
        Initialize Isolation Forest.

        Parameters:
        -----------
        n_estimators : int
            Number of isolation trees.
        max_samples : int or float
            Number of samples to draw for each tree.
        contamination : float
            Expected proportion of outliers.
        random_state : int, optional
            Random seed.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        self.trees_ = []
        self.threshold_ = None

    def _isolation_tree(
        self,
        X: np.ndarray,
        height_limit: int
    ) -> dict:
        """
        Build a single isolation tree.

        Parameters:
        -----------
        X : np.ndarray
            Data subset.
        height_limit : int
            Maximum tree height.

        Returns:
        --------
        tree : dict
            Tree structure.
        """
        if len(X) <= 1 or height_limit <= 0:
            return {'type': 'leaf', 'size': len(X)}

        # Random feature and split value
        n_features = X.shape[1]
        feature = np.random.randint(0, n_features)

        min_val = X[:, feature].min()
        max_val = X[:, feature].max()

        if min_val == max_val:
            return {'type': 'leaf', 'size': len(X)}

        split_value = np.random.uniform(min_val, max_val)

        # Split data
        left_mask = X[:, feature] < split_value
        X_left = X[left_mask]
        X_right = X[~left_mask]

        return {
            'type': 'node',
            'feature': feature,
            'split_value': split_value,
            'left': self._isolation_tree(X_left, height_limit - 1),
            'right': self._isolation_tree(X_right, height_limit - 1)
        }

    def _path_length(self, x: np.ndarray, tree: dict, current_height: int = 0) -> float:
        """
        Compute path length for a sample in a tree.

        Parameters:
        -----------
        x : np.ndarray
            Sample.
        tree : dict
            Tree structure.
        current_height : int
            Current height in tree.

        Returns:
        --------
        path_length : float
        """
        if tree['type'] == 'leaf':
            # Adjust for unbuilt subtree
            size = tree['size']
            if size <= 1:
                return current_height
            else:
                # Average path length of unsuccessful search in BST
                return current_height + 2 * (np.log(size - 1) + 0.5772156649) - 2 * (size - 1) / size

        if x[tree['feature']] < tree['split_value']:
            return self._path_length(x, tree['left'], current_height + 1)
        else:
            return self._path_length(x, tree['right'], current_height + 1)

    def fit(self, X: np.ndarray) -> 'IsolationForest':
        """
        Fit isolation forest.

        Parameters:
        -----------
        X : np.ndarray
            Training data.

        Returns:
        --------
        self : IsolationForest
        """
        n_samples = len(X)

        # Determine max_samples
        if isinstance(self.max_samples, float):
            max_samples = int(self.max_samples * n_samples)
        else:
            max_samples = min(self.max_samples, n_samples)

        # Build trees
        height_limit = int(np.ceil(np.log2(max_samples)))

        self.trees_ = []
        for _ in range(self.n_estimators):
            # Sample data
            indices = np.random.choice(n_samples, max_samples, replace=False)
            X_sample = X[indices]

            # Build tree
            tree = self._isolation_tree(X_sample, height_limit)
            self.trees_.append(tree)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        scores : np.ndarray
            Anomaly scores (lower = more anomalous).
        """
        # Average path length over all trees
        path_lengths = np.zeros((len(X), len(self.trees_)))

        for i, x in enumerate(X):
            for j, tree in enumerate(self.trees_):
                path_lengths[i, j] = self._path_length(x, tree)

        avg_path_lengths = path_lengths.mean(axis=1)

        # Anomaly score
        # Normalize by expected path length for n samples
        n = self.max_samples if isinstance(self.max_samples, int) else int(self.max_samples * len(X))
        c_n = 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

        scores = 2 ** (-avg_path_lengths / c_n)

        # Invert scores (higher = more anomalous)
        return -scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outliers.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        labels : np.ndarray
            1 for inliers, -1 for outliers.
        """
        scores = self.decision_function(X)

        if self.threshold_ is None:
            # Compute threshold based on contamination
            self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))

        return np.where(scores < self.threshold_, -1, 1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and predict in one step.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        labels : np.ndarray
            1 for inliers, -1 for outliers.
        """
        self.fit(X)
        return self.predict(X)


# ============================================================================
# MULTIVARIATE METHOD
# ============================================================================

class MahalanobisDetector:
    """
    Mahalanobis distance based outlier detection.

    Accounts for correlations between features.
    """

    def __init__(self, threshold: float = 3.0):
        """
        Initialize Mahalanobis detector.

        Parameters:
        -----------
        threshold : float
            Threshold for Mahalanobis distance.
        """
        self.threshold = threshold
        self.mean_ = None
        self.cov_inv_ = None

    def fit(self, X: np.ndarray) -> 'MahalanobisDetector':
        """
        Compute mean and covariance.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        self : MahalanobisDetector
        """
        self.mean_ = np.mean(X, axis=0)

        # Covariance matrix
        cov = np.cov(X, rowvar=False)

        # Add regularization for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6

        # Inverse covariance
        self.cov_inv_ = np.linalg.inv(cov)

        return self

    def mahalanobis_distance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        distances : np.ndarray
            Mahalanobis distances.
        """
        diff = X - self.mean_
        distances = np.sqrt(np.sum(diff @ self.cov_inv_ * diff, axis=1))
        return distances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outliers.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        labels : np.ndarray
            1 for inliers, -1 for outliers.
        """
        distances = self.mahalanobis_distance(X)
        return np.where(distances > self.threshold, -1, 1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and predict in one step.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        labels : np.ndarray
            1 for inliers, -1 for outliers.
        """
        return self.fit(X).predict(X)


# ============================================================================
# EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("OUTLIER DETECTION EXAMPLES")
    print("=" * 70)

    # Create dataset with outliers
    np.random.seed(42)
    n_samples = 300
    n_outliers = 30

    # Normal data
    X_inliers = np.random.randn(n_samples - n_outliers, 2) * 0.5

    # Outliers
    X_outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))

    X = np.vstack([X_inliers, X_outliers])

    print(f"\nDataset shape: {X.shape}")
    print(f"True outliers: {n_outliers} ({n_outliers / n_samples * 100:.1f}%)")

    # Example 1: IQR Method
    print("\n1. IQR Method")
    print("-" * 70)
    iqr_detector = IQRDetector(k=1.5)
    labels_iqr = iqr_detector.fit_predict(X)
    n_outliers_iqr = np.sum(labels_iqr == -1)
    print(f"Detected outliers: {n_outliers_iqr} ({n_outliers_iqr / n_samples * 100:.1f}%)")

    # Example 2: Z-Score Method
    print("\n2. Z-Score Method")
    print("-" * 70)
    zscore_detector = ZScoreDetector(threshold=3.0)
    labels_zscore = zscore_detector.fit_predict(X)
    n_outliers_zscore = np.sum(labels_zscore == -1)
    print(f"Detected outliers: {n_outliers_zscore} ({n_outliers_zscore / n_samples * 100:.1f}%)")

    # Example 3: Modified Z-Score
    print("\n3. Modified Z-Score (MAD)")
    print("-" * 70)
    mad_detector = ModifiedZScoreDetector(threshold=3.5)
    labels_mad = mad_detector.fit_predict(X)
    n_outliers_mad = np.sum(labels_mad == -1)
    print(f"Detected outliers: {n_outliers_mad} ({n_outliers_mad / n_samples * 100:.1f}%)")

    # Example 4: KNN Method
    print("\n4. KNN Method")
    print("-" * 70)
    knn_detector = KNNOutlierDetector(n_neighbors=10, contamination=0.1)
    labels_knn = knn_detector.fit_predict(X)
    n_outliers_knn = np.sum(labels_knn == -1)
    print(f"Detected outliers: {n_outliers_knn} ({n_outliers_knn / n_samples * 100:.1f}%)")

    # Example 5: LOF Method
    print("\n5. Local Outlier Factor (LOF)")
    print("-" * 70)
    lof_detector = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    labels_lof = lof_detector.fit_predict(X)
    n_outliers_lof = np.sum(labels_lof == -1)
    print(f"Detected outliers: {n_outliers_lof} ({n_outliers_lof / n_samples * 100:.1f}%)")

    # Example 6: Isolation Forest
    print("\n6. Isolation Forest")
    print("-" * 70)
    iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    labels_iforest = iforest.fit_predict(X)
    n_outliers_iforest = np.sum(labels_iforest == -1)
    print(f"Detected outliers: {n_outliers_iforest} ({n_outliers_iforest / n_samples * 100:.1f}%)")

    # Example 7: Mahalanobis Distance
    print("\n7. Mahalanobis Distance")
    print("-" * 70)
    mahal_detector = MahalanobisDetector(threshold=3.0)
    labels_mahal = mahal_detector.fit_predict(X)
    n_outliers_mahal = np.sum(labels_mahal == -1)
    print(f"Detected outliers: {n_outliers_mahal} ({n_outliers_mahal / n_samples * 100:.1f}%)")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON OF METHODS")
    print("=" * 70)
    print(f"{'Method':<25s} {'Detected Outliers':<20s} {'Percentage':<10s}")
    print("-" * 70)
    print(f"{'True Outliers':<25s} {n_outliers:<20d} {n_outliers / n_samples * 100:.1f}%")
    print(f"{'IQR':<25s} {n_outliers_iqr:<20d} {n_outliers_iqr / n_samples * 100:.1f}%")
    print(f"{'Z-Score':<25s} {n_outliers_zscore:<20d} {n_outliers_zscore / n_samples * 100:.1f}%")
    print(f"{'Modified Z-Score (MAD)':<25s} {n_outliers_mad:<20d} {n_outliers_mad / n_samples * 100:.1f}%")
    print(f"{'KNN':<25s} {n_outliers_knn:<20d} {n_outliers_knn / n_samples * 100:.1f}%")
    print(f"{'LOF':<25s} {n_outliers_lof:<20d} {n_outliers_lof / n_samples * 100:.1f}%")
    print(f"{'Isolation Forest':<25s} {n_outliers_iforest:<20d} {n_outliers_iforest / n_samples * 100:.1f}%")
    print(f"{'Mahalanobis':<25s} {n_outliers_mahal:<20d} {n_outliers_mahal / n_samples * 100:.1f}%")

    print("\n" + "=" * 70)
    print("All outlier detection examples completed!")
    print("=" * 70)
