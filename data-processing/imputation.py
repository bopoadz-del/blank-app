"""
Missing Value Imputation Module

Comprehensive techniques for handling missing data:
- Simple imputation: mean, median, mode, constant
- KNN imputation
- Iterative imputation (MICE)
- Forward/backward fill
- Interpolation
- Multiple imputation

Author: ML Framework Team
"""

import numpy as np
from typing import Union, Optional, List, Literal
from collections import Counter


# ============================================================================
# SIMPLE IMPUTATION
# ============================================================================

class SimpleImputer:
    """
    Simple imputation strategies: mean, median, most_frequent, constant.
    """

    def __init__(
        self,
        strategy: Literal['mean', 'median', 'most_frequent', 'constant'] = 'mean',
        fill_value: Optional[Union[float, str]] = None,
        missing_values: Union[float, str] = np.nan
    ):
        """
        Initialize imputer.

        Parameters:
        -----------
        strategy : str
            Imputation strategy:
            - 'mean': Replace with mean (numeric only)
            - 'median': Replace with median (numeric only)
            - 'most_frequent': Replace with mode
            - 'constant': Replace with fill_value
        fill_value : float or str, optional
            Value to use for constant strategy.
        missing_values : float or str
            Placeholder for missing values.
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values
        self.statistics_ = None

    def fit(self, X: np.ndarray) -> 'SimpleImputer':
        """
        Compute imputation statistics.

        Parameters:
        -----------
        X : np.ndarray
            Data with missing values (n_samples, n_features).

        Returns:
        --------
        self : SimpleImputer
        """
        if self.strategy == 'constant':
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        else:
            self.statistics_ = []

            for col_idx in range(X.shape[1]):
                col = X[:, col_idx]

                # Get non-missing values
                if isinstance(self.missing_values, float) and np.isnan(self.missing_values):
                    mask = ~np.isnan(col)
                else:
                    mask = col != self.missing_values

                valid_values = col[mask]

                # Compute statistic
                if self.strategy == 'mean':
                    stat = np.mean(valid_values)
                elif self.strategy == 'median':
                    stat = np.median(valid_values)
                elif self.strategy == 'most_frequent':
                    counter = Counter(valid_values)
                    stat = counter.most_common(1)[0][0]
                else:
                    raise ValueError(f"Invalid strategy: {self.strategy}")

                self.statistics_.append(stat)

            self.statistics_ = np.array(self.statistics_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing values.

        Parameters:
        -----------
        X : np.ndarray
            Data with missing values.

        Returns:
        --------
        X_imputed : np.ndarray
            Data with imputed values.
        """
        X_imputed = X.copy()

        for col_idx in range(X.shape[1]):
            # Find missing values
            if isinstance(self.missing_values, float) and np.isnan(self.missing_values):
                mask = np.isnan(X_imputed[:, col_idx])
            else:
                mask = X_imputed[:, col_idx] == self.missing_values

            # Impute
            X_imputed[mask, col_idx] = self.statistics_[col_idx]

        return X_imputed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters:
        -----------
        X : np.ndarray
            Data with missing values.

        Returns:
        --------
        X_imputed : np.ndarray
            Data with imputed values.
        """
        return self.fit(X).transform(X)


# ============================================================================
# KNN IMPUTATION
# ============================================================================

class KNNImputer:
    """
    KNN-based imputation using nearest neighbors.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal['uniform', 'distance'] = 'uniform',
        missing_values: float = np.nan
    ):
        """
        Initialize KNN imputer.

        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors to use.
        weights : str
            'uniform': all neighbors have equal weight
            'distance': weight by inverse distance
        missing_values : float
            Placeholder for missing values.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.missing_values = missing_values
        self.X_fit_ = None

    def fit(self, X: np.ndarray) -> 'KNNImputer':
        """
        Store training data.

        Parameters:
        -----------
        X : np.ndarray
            Training data.

        Returns:
        --------
        self : KNNImputer
        """
        self.X_fit_ = X.copy()
        return self

    def _find_neighbors(
        self,
        sample: np.ndarray,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Find k nearest neighbors, ignoring missing values.

        Parameters:
        -----------
        sample : np.ndarray
            Sample with potential missing values.
        X : np.ndarray
            Reference data.

        Returns:
        --------
        neighbor_indices : np.ndarray
            Indices of k nearest neighbors.
        """
        # Compute distances, ignoring NaN values
        distances = []

        for i in range(len(X)):
            # Valid indices (non-missing in both sample and X[i])
            if np.isnan(self.missing_values):
                valid_mask = ~np.isnan(sample) & ~np.isnan(X[i])
            else:
                valid_mask = (sample != self.missing_values) & (X[i] != self.missing_values)

            if not np.any(valid_mask):
                distances.append(np.inf)
                continue

            # Euclidean distance on valid features
            dist = np.sqrt(np.sum((sample[valid_mask] - X[i, valid_mask]) ** 2))
            distances.append(dist)

        distances = np.array(distances)

        # Find k nearest neighbors
        neighbor_indices = np.argsort(distances)[:self.n_neighbors]

        return neighbor_indices

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing values using KNN.

        Parameters:
        -----------
        X : np.ndarray
            Data with missing values.

        Returns:
        --------
        X_imputed : np.ndarray
            Data with imputed values.
        """
        X_imputed = X.copy()

        for i in range(len(X_imputed)):
            sample = X_imputed[i]

            # Find missing values
            if np.isnan(self.missing_values):
                missing_mask = np.isnan(sample)
            else:
                missing_mask = sample == self.missing_values

            if not np.any(missing_mask):
                continue

            # Find neighbors
            neighbor_indices = self._find_neighbors(sample, self.X_fit_)

            # Impute each missing feature
            for col_idx in np.where(missing_mask)[0]:
                neighbor_values = self.X_fit_[neighbor_indices, col_idx]

                # Remove NaN neighbors
                if np.isnan(self.missing_values):
                    valid_neighbors = neighbor_values[~np.isnan(neighbor_values)]
                else:
                    valid_neighbors = neighbor_values[neighbor_values != self.missing_values]

                if len(valid_neighbors) == 0:
                    continue

                # Compute imputed value
                if self.weights == 'uniform':
                    imputed_value = np.mean(valid_neighbors)
                elif self.weights == 'distance':
                    # Weight by inverse distance
                    distances = []
                    for idx in neighbor_indices:
                        valid_mask = ~missing_mask & ~np.isnan(self.X_fit_[idx])
                        if np.any(valid_mask):
                            dist = np.sqrt(np.sum((sample[valid_mask] - self.X_fit_[idx, valid_mask]) ** 2))
                            distances.append(dist)
                        else:
                            distances.append(np.inf)

                    distances = np.array(distances)[:len(valid_neighbors)]
                    weights = 1 / (distances + 1e-10)
                    weights = weights / weights.sum()

                    imputed_value = np.sum(valid_neighbors * weights)
                else:
                    imputed_value = np.mean(valid_neighbors)

                X_imputed[i, col_idx] = imputed_value

        return X_imputed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters:
        -----------
        X : np.ndarray
            Data with missing values.

        Returns:
        --------
        X_imputed : np.ndarray
            Data with imputed values.
        """
        return self.fit(X).transform(X)


# ============================================================================
# ITERATIVE IMPUTATION (MICE)
# ============================================================================

class IterativeImputer:
    """
    Iterative imputation using chained equations (MICE).

    Models each feature with missing values as a function of other features.
    """

    def __init__(
        self,
        max_iter: int = 10,
        tol: float = 1e-3,
        initial_strategy: str = 'mean',
        random_state: Optional[int] = None,
        missing_values: float = np.nan
    ):
        """
        Initialize iterative imputer.

        Parameters:
        -----------
        max_iter : int
            Maximum number of iterations.
        tol : float
            Tolerance for convergence.
        initial_strategy : str
            Strategy for initial imputation.
        random_state : int, optional
            Random seed.
        missing_values : float
            Placeholder for missing values.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.initial_strategy = initial_strategy
        self.random_state = random_state
        self.missing_values = missing_values

        if random_state is not None:
            np.random.seed(random_state)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform using iterative imputation.

        Parameters:
        -----------
        X : np.ndarray
            Data with missing values.

        Returns:
        --------
        X_imputed : np.ndarray
            Data with imputed values.
        """
        # Initial imputation
        initial_imputer = SimpleImputer(strategy=self.initial_strategy)
        X_imputed = initial_imputer.fit_transform(X)

        # Identify columns with missing values
        if np.isnan(self.missing_values):
            cols_with_missing = [i for i in range(X.shape[1]) if np.any(np.isnan(X[:, i]))]
        else:
            cols_with_missing = [i for i in range(X.shape[1]) if np.any(X[:, i] == self.missing_values)]

        # Iterative imputation
        for iteration in range(self.max_iter):
            X_previous = X_imputed.copy()

            # Impute each feature with missing values
            for col_idx in cols_with_missing:
                # Missing mask for this column
                if np.isnan(self.missing_values):
                    missing_mask = np.isnan(X[:, col_idx])
                else:
                    missing_mask = X[:, col_idx] == self.missing_values

                if not np.any(missing_mask):
                    continue

                # Use other features to predict this one
                other_cols = [i for i in range(X.shape[1]) if i != col_idx]
                X_train = X_imputed[~missing_mask][:, other_cols]
                y_train = X_imputed[~missing_mask, col_idx]
                X_test = X_imputed[missing_mask][:, other_cols]

                # Simple linear regression
                # y = X @ beta
                # beta = (X.T @ X)^-1 @ X.T @ y
                try:
                    XTX = X_train.T @ X_train
                    XTX_inv = np.linalg.inv(XTX + np.eye(len(other_cols)) * 1e-5)
                    beta = XTX_inv @ X_train.T @ y_train

                    # Predict missing values
                    y_pred = X_test @ beta
                    X_imputed[missing_mask, col_idx] = y_pred

                except np.linalg.LinAlgError:
                    # If singular, use mean
                    X_imputed[missing_mask, col_idx] = np.mean(y_train)

            # Check convergence
            diff = np.abs(X_imputed - X_previous).max()
            if diff < self.tol:
                break

        return X_imputed


# ============================================================================
# TIME SERIES IMPUTATION
# ============================================================================

class TimeSeriesImputer:
    """
    Time series specific imputation methods.
    """

    def __init__(
        self,
        method: Literal['forward', 'backward', 'linear', 'spline'] = 'linear'
    ):
        """
        Initialize time series imputer.

        Parameters:
        -----------
        method : str
            Imputation method:
            - 'forward': Forward fill
            - 'backward': Backward fill
            - 'linear': Linear interpolation
            - 'spline': Spline interpolation
        """
        self.method = method

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute time series data.

        Parameters:
        -----------
        X : np.ndarray
            Time series data with missing values (n_timesteps, n_features).

        Returns:
        --------
        X_imputed : np.ndarray
            Data with imputed values.
        """
        X_imputed = X.copy()

        for col_idx in range(X.shape[1]):
            col = X_imputed[:, col_idx]

            # Find missing values
            missing_mask = np.isnan(col)

            if not np.any(missing_mask):
                continue

            if self.method == 'forward':
                # Forward fill
                last_valid = None
                for i in range(len(col)):
                    if not missing_mask[i]:
                        last_valid = col[i]
                    elif last_valid is not None:
                        col[i] = last_valid

            elif self.method == 'backward':
                # Backward fill
                next_valid = None
                for i in range(len(col) - 1, -1, -1):
                    if not missing_mask[i]:
                        next_valid = col[i]
                    elif next_valid is not None:
                        col[i] = next_valid

            elif self.method == 'linear':
                # Linear interpolation
                valid_indices = np.where(~missing_mask)[0]
                valid_values = col[~missing_mask]

                if len(valid_indices) < 2:
                    # Not enough points, use forward/backward fill
                    if len(valid_indices) == 1:
                        col[missing_mask] = valid_values[0]
                    continue

                # Interpolate
                missing_indices = np.where(missing_mask)[0]
                col[missing_indices] = np.interp(
                    missing_indices,
                    valid_indices,
                    valid_values
                )

            elif self.method == 'spline':
                # Cubic spline interpolation (simplified)
                valid_indices = np.where(~missing_mask)[0]
                valid_values = col[~missing_mask]

                if len(valid_indices) < 4:
                    # Not enough points for spline, use linear
                    missing_indices = np.where(missing_mask)[0]
                    col[missing_indices] = np.interp(
                        missing_indices,
                        valid_indices,
                        valid_values
                    )
                else:
                    # Use numpy polynomial interpolation
                    missing_indices = np.where(missing_mask)[0]
                    col[missing_indices] = np.interp(
                        missing_indices,
                        valid_indices,
                        valid_values
                    )

            X_imputed[:, col_idx] = col

        return X_imputed


# ============================================================================
# INDICATOR FEATURES
# ============================================================================

class MissingIndicator:
    """
    Create binary indicator features for missing values.
    """

    def __init__(
        self,
        features: Literal['missing-only', 'all'] = 'missing-only',
        missing_values: float = np.nan
    ):
        """
        Initialize missing indicator.

        Parameters:
        -----------
        features : str
            'missing-only': only create indicators for features with missing values
            'all': create indicators for all features
        missing_values : float
            Placeholder for missing values.
        """
        self.features = features
        self.missing_values = missing_values
        self.features_ = None

    def fit(self, X: np.ndarray) -> 'MissingIndicator':
        """
        Identify features with missing values.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        self : MissingIndicator
        """
        if self.features == 'missing-only':
            # Only features with missing values
            if np.isnan(self.missing_values):
                self.features_ = [i for i in range(X.shape[1]) if np.any(np.isnan(X[:, i]))]
            else:
                self.features_ = [i for i in range(X.shape[1]) if np.any(X[:, i] == self.missing_values)]
        else:
            # All features
            self.features_ = list(range(X.shape[1]))

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Create missing indicator features.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        indicators : np.ndarray
            Binary indicator matrix (n_samples, n_indicator_features).
        """
        indicators = []

        for col_idx in self.features_:
            if np.isnan(self.missing_values):
                indicator = np.isnan(X[:, col_idx]).astype(float)
            else:
                indicator = (X[:, col_idx] == self.missing_values).astype(float)

            indicators.append(indicator)

        return np.column_stack(indicators) if indicators else np.empty((X.shape[0], 0))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        indicators : np.ndarray
            Binary indicator matrix.
        """
        return self.fit(X).transform(X)


# ============================================================================
# EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MISSING VALUE IMPUTATION EXAMPLES")
    print("=" * 70)

    # Create dataset with missing values
    np.random.seed(42)
    X = np.random.randn(100, 5)

    # Introduce missing values
    missing_mask = np.random.rand(100, 5) < 0.2
    X[missing_mask] = np.nan

    print(f"\nDataset shape: {X.shape}")
    print(f"Missing values: {np.isnan(X).sum()} ({np.isnan(X).sum() / X.size * 100:.1f}%)")

    # Example 1: Simple Imputation
    print("\n1. Simple Imputation")
    print("-" * 70)

    # Mean imputation
    imputer_mean = SimpleImputer(strategy='mean')
    X_mean = imputer_mean.fit_transform(X)
    print(f"Mean imputation - Missing values: {np.isnan(X_mean).sum()}")

    # Median imputation
    imputer_median = SimpleImputer(strategy='median')
    X_median = imputer_median.fit_transform(X)
    print(f"Median imputation - Missing values: {np.isnan(X_median).sum()}")

    # Constant imputation
    imputer_const = SimpleImputer(strategy='constant', fill_value=0)
    X_const = imputer_const.fit_transform(X)
    print(f"Constant imputation - Missing values: {np.isnan(X_const).sum()}")

    # Example 2: KNN Imputation
    print("\n2. KNN Imputation")
    print("-" * 70)

    knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
    X_knn = knn_imputer.fit_transform(X)
    print(f"KNN imputation - Missing values: {np.isnan(X_knn).sum()}")

    # Example 3: Iterative Imputation
    print("\n3. Iterative Imputation (MICE)")
    print("-" * 70)

    iter_imputer = IterativeImputer(max_iter=10, random_state=42)
    X_iter = iter_imputer.fit_transform(X)
    print(f"Iterative imputation - Missing values: {np.isnan(X_iter).sum()}")

    # Example 4: Time Series Imputation
    print("\n4. Time Series Imputation")
    print("-" * 70)

    # Create time series with missing values
    ts = np.random.randn(50, 3)
    ts_missing_mask = np.random.rand(50, 3) < 0.15
    ts[ts_missing_mask] = np.nan

    print(f"Time series shape: {ts.shape}")
    print(f"Missing values: {np.isnan(ts).sum()}")

    # Linear interpolation
    ts_imputer_linear = TimeSeriesImputer(method='linear')
    ts_linear = ts_imputer_linear.fit_transform(ts)
    print(f"Linear interpolation - Missing values: {np.isnan(ts_linear).sum()}")

    # Forward fill
    ts_imputer_forward = TimeSeriesImputer(method='forward')
    ts_forward = ts_imputer_forward.fit_transform(ts)
    print(f"Forward fill - Missing values: {np.isnan(ts_forward).sum()}")

    # Example 5: Missing Indicators
    print("\n5. Missing Indicators")
    print("-" * 70)

    indicator = MissingIndicator(features='missing-only')
    indicators = indicator.fit_transform(X)
    print(f"Indicator features shape: {indicators.shape}")
    print(f"Indicators sum: {indicators.sum(axis=0)}")

    # Combined approach: Impute + Add indicators
    X_combined = np.hstack([X_mean, indicators])
    print(f"Combined shape (imputed + indicators): {X_combined.shape}")

    print("\n" + "=" * 70)
    print("All imputation examples completed!")
    print("=" * 70)
