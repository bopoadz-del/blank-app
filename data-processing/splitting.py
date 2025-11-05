"""
Data Splitting Module

Comprehensive data splitting utilities:
- Train/test splits
- Train/validation/test splits
- K-fold cross-validation
- Stratified sampling
- Time series splits
- Group-based splits

Author: ML Framework Team
"""

import numpy as np
from typing import Tuple, List, Optional, Union, Iterator
from collections import Counter


# ============================================================================
# BASIC SPLITS
# ============================================================================

def train_test_split(
    *arrays,
    test_size: Union[float, int] = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Split arrays into train and test subsets.

    Parameters:
    -----------
    *arrays : sequence of arrays
        Arrays to split (e.g., X, y).
    test_size : float or int
        If float, proportion of test set (0.0 to 1.0).
        If int, absolute number of test samples.
    random_state : int, optional
        Random seed for reproducibility.
    shuffle : bool
        Whether to shuffle before splitting.
    stratify : np.ndarray, optional
        If not None, data is split in a stratified fashion using this as class labels.

    Returns:
    --------
    splits : list of arrays
        List containing train-test split for each array.
        Format: [train_array1, test_array1, train_array2, test_array2, ...]
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(arrays[0])

    # Determine test size
    if isinstance(test_size, float):
        n_test = int(n_samples * test_size)
    else:
        n_test = test_size

    n_train = n_samples - n_test

    # Create indices
    indices = np.arange(n_samples)

    if stratify is not None:
        # Stratified split
        train_indices, test_indices = _stratified_split(
            indices, stratify, n_test, shuffle=shuffle
        )
    else:
        # Random split
        if shuffle:
            np.random.shuffle(indices)

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

    # Split arrays
    result = []
    for array in arrays:
        train_array = array[train_indices]
        test_array = array[test_indices]
        result.extend([train_array, test_array])

    return result


def train_val_test_split(
    *arrays,
    val_size: Union[float, int] = 0.1,
    test_size: Union[float, int] = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Split arrays into train, validation, and test subsets.

    Parameters:
    -----------
    *arrays : sequence of arrays
        Arrays to split.
    val_size : float or int
        Validation set size.
    test_size : float or int
        Test set size.
    random_state : int, optional
        Random seed.
    shuffle : bool
        Whether to shuffle.
    stratify : np.ndarray, optional
        Stratification labels.

    Returns:
    --------
    splits : list of arrays
        List containing train-val-test split for each array.
        Format: [train_array1, val_array1, test_array1, train_array2, val_array2, test_array2, ...]
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(arrays[0])

    # Determine sizes
    if isinstance(test_size, float):
        n_test = int(n_samples * test_size)
    else:
        n_test = test_size

    if isinstance(val_size, float):
        n_val = int(n_samples * val_size)
    else:
        n_val = val_size

    n_train = n_samples - n_test - n_val

    if n_train <= 0:
        raise ValueError("Not enough samples for train/val/test split")

    # Create indices
    indices = np.arange(n_samples)

    if stratify is not None:
        # Stratified split
        train_indices, temp_indices = _stratified_split(
            indices, stratify, n_val + n_test, shuffle=shuffle
        )

        # Split temp into val and test
        stratify_temp = stratify[temp_indices]
        val_indices_local, test_indices_local = _stratified_split(
            np.arange(len(temp_indices)), stratify_temp, n_test, shuffle=False
        )

        val_indices = temp_indices[val_indices_local]
        test_indices = temp_indices[test_indices_local]
    else:
        # Random split
        if shuffle:
            np.random.shuffle(indices)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

    # Split arrays
    result = []
    for array in arrays:
        train_array = array[train_indices]
        val_array = array[val_indices]
        test_array = array[test_indices]
        result.extend([train_array, val_array, test_array])

    return result


def _stratified_split(
    indices: np.ndarray,
    labels: np.ndarray,
    n_split: int,
    shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform stratified split.

    Parameters:
    -----------
    indices : np.ndarray
        Indices to split.
    labels : np.ndarray
        Labels for stratification.
    n_split : int
        Size of second split.
    shuffle : bool
        Whether to shuffle.

    Returns:
    --------
    indices1 : np.ndarray
        First split indices.
    indices2 : np.ndarray
        Second split indices.
    """
    class_counts = Counter(labels)
    classes = list(class_counts.keys())

    indices1_list = []
    indices2_list = []

    for cls in classes:
        # Get indices for this class
        class_mask = labels == cls
        class_indices = indices[class_mask]

        if shuffle:
            np.random.shuffle(class_indices)

        # Determine split size for this class
        n_class = len(class_indices)
        n_class_split = int(n_split * n_class / len(labels))

        # Split
        indices2_list.append(class_indices[:n_class_split])
        indices1_list.append(class_indices[n_class_split:])

    indices1 = np.concatenate(indices1_list)
    indices2 = np.concatenate(indices2_list)

    if shuffle:
        np.random.shuffle(indices1)
        np.random.shuffle(indices2)

    return indices1, indices2


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

class KFold:
    """
    K-Fold cross-validation iterator.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None
    ):
        """
        Initialize K-Fold.

        Parameters:
        -----------
        n_splits : int
            Number of folds.
        shuffle : bool
            Whether to shuffle data.
        random_state : int, optional
            Random seed.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.

        Parameters:
        -----------
        X : np.ndarray
            Data to split.

        Yields:
        -------
        train_indices : np.ndarray
            Training set indices.
        test_indices : np.ndarray
            Test set indices.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            test_indices = indices[current:current + fold_size]
            train_indices = np.concatenate([indices[:current], indices[current + fold_size:]])

            yield train_indices, test_indices

            current += fold_size

    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits


class StratifiedKFold:
    """
    Stratified K-Fold cross-validation iterator.

    Maintains class distribution in each fold.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None
    ):
        """
        Initialize Stratified K-Fold.

        Parameters:
        -----------
        n_splits : int
            Number of folds.
        shuffle : bool
            Whether to shuffle data.
        random_state : int, optional
            Random seed.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate stratified train/test indices.

        Parameters:
        -----------
        X : np.ndarray
            Data.
        y : np.ndarray
            Labels for stratification.

        Yields:
        -------
        train_indices : np.ndarray
            Training set indices.
        test_indices : np.ndarray
            Test set indices.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = len(X)
        classes = np.unique(y)

        # Collect indices for each class
        class_indices = {cls: np.where(y == cls)[0] for cls in classes}

        # Shuffle if requested
        if self.shuffle:
            for cls in classes:
                np.random.shuffle(class_indices[cls])

        # Split each class into folds
        class_folds = {}
        for cls in classes:
            indices = class_indices[cls]
            n_class = len(indices)

            fold_sizes = np.full(self.n_splits, n_class // self.n_splits, dtype=int)
            fold_sizes[:n_class % self.n_splits] += 1

            current = 0
            class_folds[cls] = []
            for fold_size in fold_sizes:
                class_folds[cls].append(indices[current:current + fold_size])
                current += fold_size

        # Combine folds from all classes
        for fold_idx in range(self.n_splits):
            test_indices = np.concatenate([class_folds[cls][fold_idx] for cls in classes])

            train_indices = []
            for cls in classes:
                for i in range(self.n_splits):
                    if i != fold_idx:
                        train_indices.append(class_folds[cls][i])
            train_indices = np.concatenate(train_indices)

            if self.shuffle:
                np.random.shuffle(train_indices)
                np.random.shuffle(test_indices)

            yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits


class GroupKFold:
    """
    K-Fold cross-validation with non-overlapping groups.

    Useful when samples from the same group should not appear in both train and test.
    """

    def __init__(self, n_splits: int = 5):
        """
        Initialize Group K-Fold.

        Parameters:
        -----------
        n_splits : int
            Number of folds.
        """
        self.n_splits = n_splits

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate group-based train/test indices.

        Parameters:
        -----------
        X : np.ndarray
            Data.
        y : np.ndarray
            Labels (not used, for API compatibility).
        groups : np.ndarray
            Group labels for each sample.

        Yields:
        -------
        train_indices : np.ndarray
            Training set indices.
        test_indices : np.ndarray
            Test set indices.
        """
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        if n_groups < self.n_splits:
            raise ValueError(f"n_splits={self.n_splits} cannot be greater than n_groups={n_groups}")

        # Split groups into folds
        fold_sizes = np.full(self.n_splits, n_groups // self.n_splits, dtype=int)
        fold_sizes[:n_groups % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            test_groups = unique_groups[current:current + fold_size]
            train_groups = np.concatenate([unique_groups[:current], unique_groups[current + fold_size:]])

            test_indices = np.where(np.isin(groups, test_groups))[0]
            train_indices = np.where(np.isin(groups, train_groups))[0]

            yield train_indices, test_indices

            current += fold_size

    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits


# ============================================================================
# TIME SERIES SPLITS
# ============================================================================

class TimeSeriesSplit:
    """
    Time series cross-validation iterator.

    Provides train/test indices respecting temporal order.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0
    ):
        """
        Initialize Time Series Split.

        Parameters:
        -----------
        n_splits : int
            Number of splits.
        test_size : int, optional
            Size of test set in each split.
        gap : int
            Number of samples to exclude between train and test.
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time series train/test indices.

        Parameters:
        -----------
        X : np.ndarray
            Time series data.

        Yields:
        -------
        train_indices : np.ndarray
            Training set indices.
        test_indices : np.ndarray
            Test set indices.
        """
        n_samples = len(X)

        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            # Training set: all data up to split point
            train_end = n_samples - (self.n_splits - i) * test_size - self.gap
            train_indices = indices[:train_end]

            # Test set: next test_size samples after gap
            test_start = train_end + self.gap
            test_end = test_start + test_size
            test_indices = indices[test_start:test_end]

            if len(test_indices) > 0 and len(train_indices) > 0:
                yield train_indices, test_indices


# ============================================================================
# EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DATA SPLITTING EXAMPLES")
    print("=" * 70)

    # Create sample dataset
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)

    # Example 1: Train-Test Split
    print("\n1. Train-Test Split")
    print("-" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    print(f"Original: {len(X)} samples")
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    print(f"Train labels: {Counter(y_train)}")
    print(f"Test labels: {Counter(y_test)}")

    # Example 2: Stratified Train-Test Split
    print("\n2. Stratified Train-Test Split")
    print("-" * 70)

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y
    )

    print(f"Original labels: {Counter(y)}")
    print(f"Train labels: {Counter(y_train_s)}")
    print(f"Test labels: {Counter(y_test_s)}")

    # Check stratification
    original_dist = np.array([Counter(y)[i] for i in range(3)]) / len(y)
    train_dist = np.array([Counter(y_train_s)[i] for i in range(3)]) / len(y_train_s)
    test_dist = np.array([Counter(y_test_s)[i] for i in range(3)]) / len(y_test_s)

    print(f"\nClass distribution:")
    print(f"Original: {original_dist}")
    print(f"Train: {train_dist}")
    print(f"Test: {test_dist}")

    # Example 3: Train-Val-Test Split
    print("\n3. Train-Val-Test Split")
    print("-" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
        shuffle=True,
        stratify=y
    )

    print(f"Original: {len(X)} samples")
    print(f"Train: {len(X_train)} samples ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"Val: {len(X_val)} samples ({len(X_val) / len(X) * 100:.1f}%)")
    print(f"Test: {len(X_test)} samples ({len(X_test) / len(X) * 100:.1f}%)")

    # Example 4: K-Fold Cross-Validation
    print("\n4. K-Fold Cross-Validation")
    print("-" * 70)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Number of splits: {kfold.get_n_splits()}")
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
        print(f"Fold {fold_idx + 1}: Train={len(train_idx)}, Test={len(test_idx)}")

    # Example 5: Stratified K-Fold
    print("\n5. Stratified K-Fold Cross-Validation")
    print("-" * 70)

    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Number of splits: {skfold.get_n_splits()}")
    for fold_idx, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]

        train_dist = np.array([Counter(y_train_fold)[i] for i in range(3)]) / len(y_train_fold)
        test_dist = np.array([Counter(y_test_fold)[i] for i in range(3)]) / len(y_test_fold)

        print(f"Fold {fold_idx + 1}: Train={len(train_idx)}, Test={len(test_idx)}")
        print(f"  Train dist: {train_dist.round(3)}")
        print(f"  Test dist:  {test_dist.round(3)}")

    # Example 6: Group K-Fold
    print("\n6. Group K-Fold Cross-Validation")
    print("-" * 70)

    # Create groups (e.g., different patients, sessions, etc.)
    groups = np.repeat(np.arange(20), 5)  # 20 groups, 5 samples each

    gkfold = GroupKFold(n_splits=4)

    print(f"Number of splits: {gkfold.get_n_splits()}")
    print(f"Total groups: {len(np.unique(groups))}")

    for fold_idx, (train_idx, test_idx) in enumerate(gkfold.split(X, y, groups)):
        train_groups = np.unique(groups[train_idx])
        test_groups = np.unique(groups[test_idx])

        print(f"Fold {fold_idx + 1}:")
        print(f"  Train: {len(train_idx)} samples, {len(train_groups)} groups")
        print(f"  Test: {len(test_idx)} samples, {len(test_groups)} groups")

        # Verify no overlap
        overlap = set(train_groups) & set(test_groups)
        print(f"  Group overlap: {len(overlap)} (should be 0)")

    # Example 7: Time Series Split
    print("\n7. Time Series Split")
    print("-" * 70)

    # Create time series data
    ts_data = np.random.randn(100, 3)

    tscv = TimeSeriesSplit(n_splits=5, gap=2)

    print(f"Number of splits: {tscv.n_splits}")
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(ts_data)):
        print(f"Fold {fold_idx + 1}:")
        print(f"  Train: indices {train_idx[0]} to {train_idx[-1]} ({len(train_idx)} samples)")
        print(f"  Test:  indices {test_idx[0]} to {test_idx[-1]} ({len(test_idx)} samples)")

    print("\n" + "=" * 70)
    print("All splitting examples completed!")
    print("=" * 70)
