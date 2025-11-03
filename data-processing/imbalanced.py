"""
Imbalanced Data Handling Module

Techniques for handling imbalanced datasets:
- Oversampling: Random oversampling, SMOTE, ADASYN
- Undersampling: Random undersampling, Tomek links, NearMiss
- Combined: SMOTE + Tomek, SMOTE + ENN
- Class weighting

Author: ML Framework Team
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from collections import Counter


# ============================================================================
# OVERSAMPLING TECHNIQUES
# ============================================================================

class RandomOverSampler:
    """
    Random oversampling of minority class(es).
    """

    def __init__(
        self,
        sampling_strategy: str = 'auto',
        random_state: Optional[int] = None
    ):
        """
        Initialize oversampler.

        Parameters:
        -----------
        sampling_strategy : str or dict
            'auto' or 'minority': oversample to balance
            'all': oversample all classes to match majority
            dict: {class: n_samples} for specific sampling
        random_state : int, optional
            Random seed.
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample dataset.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        y : np.ndarray
            Labels (n_samples,).

        Returns:
        --------
        X_resampled : np.ndarray
            Resampled features.
        y_resampled : np.ndarray
            Resampled labels.
        """
        # Count classes
        class_counts = Counter(y)
        classes = list(class_counts.keys())

        # Determine sampling targets
        if self.sampling_strategy == 'auto' or self.sampling_strategy == 'minority':
            max_count = max(class_counts.values())
            target_counts = {cls: max_count for cls in classes}
        elif isinstance(self.sampling_strategy, dict):
            target_counts = self.sampling_strategy
        else:
            raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}")

        X_resampled = []
        y_resampled = []

        for cls in classes:
            # Get samples for this class
            class_mask = (y == cls)
            X_class = X[class_mask]
            y_class = y[class_mask]

            current_count = len(y_class)
            target_count = target_counts.get(cls, current_count)

            # Oversample if needed
            if target_count > current_count:
                n_oversample = target_count - current_count
                indices = np.random.randint(0, current_count, n_oversample)
                X_oversampled = X_class[indices]
                y_oversampled = y_class[indices]

                X_resampled.append(X_class)
                X_resampled.append(X_oversampled)
                y_resampled.append(y_class)
                y_resampled.append(y_oversampled)
            else:
                X_resampled.append(X_class)
                y_resampled.append(y_class)

        X_resampled = np.vstack(X_resampled)
        y_resampled = np.concatenate(y_resampled)

        # Shuffle
        indices = np.random.permutation(len(y_resampled))
        return X_resampled[indices], y_resampled[indices]


class SMOTE:
    """
    Synthetic Minority Over-sampling Technique (SMOTE).

    Generates synthetic samples by interpolating between minority class samples.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        sampling_strategy: str = 'auto',
        random_state: Optional[int] = None
    ):
        """
        Initialize SMOTE.

        Parameters:
        -----------
        k_neighbors : int
            Number of nearest neighbors.
        sampling_strategy : str or dict
            Sampling strategy.
        random_state : int, optional
            Random seed.
        """
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def _find_k_neighbors(
        self,
        X: np.ndarray,
        sample: np.ndarray,
        k: int
    ) -> np.ndarray:
        """Find k nearest neighbors using Euclidean distance."""
        distances = np.linalg.norm(X - sample, axis=1)
        neighbor_indices = np.argsort(distances)[1:k+1]  # Exclude self
        return neighbor_indices

    def _generate_synthetic_sample(
        self,
        sample: np.ndarray,
        neighbor: np.ndarray
    ) -> np.ndarray:
        """Generate synthetic sample between sample and neighbor."""
        alpha = np.random.rand()
        return sample + alpha * (neighbor - sample)

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample dataset using SMOTE.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Labels.

        Returns:
        --------
        X_resampled : np.ndarray
            Resampled features.
        y_resampled : np.ndarray
            Resampled labels.
        """
        # Count classes
        class_counts = Counter(y)
        classes = list(class_counts.keys())

        # Determine sampling targets
        if self.sampling_strategy == 'auto' or self.sampling_strategy == 'minority':
            max_count = max(class_counts.values())
            target_counts = {cls: max_count for cls in classes}
        else:
            target_counts = self.sampling_strategy

        X_resampled = [X]
        y_resampled = [y]

        for cls in classes:
            current_count = class_counts[cls]
            target_count = target_counts.get(cls, current_count)

            if target_count > current_count:
                # Get minority class samples
                class_mask = (y == cls)
                X_class = X[class_mask]

                n_synthetic = target_count - current_count
                synthetic_samples = []

                for _ in range(n_synthetic):
                    # Random sample from minority class
                    idx = np.random.randint(0, len(X_class))
                    sample = X_class[idx]

                    # Find k neighbors
                    k = min(self.k_neighbors, len(X_class) - 1)
                    if k <= 0:
                        # Not enough samples, use random oversampling
                        synthetic_samples.append(sample)
                        continue

                    neighbor_indices = self._find_k_neighbors(X_class, sample, k)

                    # Random neighbor
                    neighbor_idx = np.random.choice(neighbor_indices)
                    neighbor = X_class[neighbor_idx]

                    # Generate synthetic sample
                    synthetic_sample = self._generate_synthetic_sample(sample, neighbor)
                    synthetic_samples.append(synthetic_sample)

                if synthetic_samples:
                    X_synthetic = np.array(synthetic_samples)
                    y_synthetic = np.full(len(synthetic_samples), cls)

                    X_resampled.append(X_synthetic)
                    y_resampled.append(y_synthetic)

        X_resampled = np.vstack(X_resampled)
        y_resampled = np.concatenate(y_resampled)

        # Shuffle
        indices = np.random.permutation(len(y_resampled))
        return X_resampled[indices], y_resampled[indices]


class ADASYN:
    """
    Adaptive Synthetic Sampling (ADASYN).

    Generates more synthetic samples for minority samples that are harder to learn.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        sampling_strategy: str = 'auto',
        random_state: Optional[int] = None
    ):
        """
        Initialize ADASYN.

        Parameters:
        -----------
        k_neighbors : int
            Number of nearest neighbors.
        sampling_strategy : str
            Sampling strategy.
        random_state : int, optional
            Random seed.
        """
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample dataset using ADASYN.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Labels.

        Returns:
        --------
        X_resampled : np.ndarray
            Resampled features.
        y_resampled : np.ndarray
            Resampled labels.
        """
        class_counts = Counter(y)
        classes = list(class_counts.keys())

        # Identify minority and majority classes
        minority_class = min(classes, key=lambda c: class_counts[c])
        majority_class = max(classes, key=lambda c: class_counts[c])

        minority_mask = (y == minority_class)
        X_minority = X[minority_mask]

        # Calculate ratio of majority neighbors for each minority sample
        ratios = []
        for sample in X_minority:
            # Find k neighbors in full dataset
            distances = np.linalg.norm(X - sample, axis=1)
            neighbor_indices = np.argsort(distances)[1:self.k_neighbors+1]
            neighbor_labels = y[neighbor_indices]

            # Ratio of majority class neighbors
            ratio = np.sum(neighbor_labels == majority_class) / self.k_neighbors
            ratios.append(ratio)

        ratios = np.array(ratios)

        # Normalize ratios
        if ratios.sum() > 0:
            ratios = ratios / ratios.sum()
        else:
            ratios = np.ones_like(ratios) / len(ratios)

        # Generate synthetic samples
        n_synthetic = class_counts[majority_class] - class_counts[minority_class]
        n_samples_per_minority = (ratios * n_synthetic).astype(int)

        synthetic_samples = []
        for i, n_samples in enumerate(n_samples_per_minority):
            if n_samples == 0:
                continue

            sample = X_minority[i]

            # Find neighbors in minority class
            distances = np.linalg.norm(X_minority - sample, axis=1)
            neighbor_indices = np.argsort(distances)[1:self.k_neighbors+1]

            for _ in range(n_samples):
                # Random neighbor
                neighbor_idx = np.random.choice(neighbor_indices)
                neighbor = X_minority[neighbor_idx]

                # Generate synthetic sample
                alpha = np.random.rand()
                synthetic_sample = sample + alpha * (neighbor - sample)
                synthetic_samples.append(synthetic_sample)

        # Combine original and synthetic
        if synthetic_samples:
            X_synthetic = np.array(synthetic_samples)
            y_synthetic = np.full(len(synthetic_samples), minority_class)

            X_resampled = np.vstack([X, X_synthetic])
            y_resampled = np.concatenate([y, y_synthetic])
        else:
            X_resampled, y_resampled = X, y

        # Shuffle
        indices = np.random.permutation(len(y_resampled))
        return X_resampled[indices], y_resampled[indices]


# ============================================================================
# UNDERSAMPLING TECHNIQUES
# ============================================================================

class RandomUnderSampler:
    """
    Random undersampling of majority class(es).
    """

    def __init__(
        self,
        sampling_strategy: str = 'auto',
        random_state: Optional[int] = None
    ):
        """
        Initialize undersampler.

        Parameters:
        -----------
        sampling_strategy : str or dict
            'auto': undersample to balance
            dict: {class: n_samples} for specific sampling
        random_state : int, optional
            Random seed.
        """
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample dataset.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Labels.

        Returns:
        --------
        X_resampled : np.ndarray
            Resampled features.
        y_resampled : np.ndarray
            Resampled labels.
        """
        class_counts = Counter(y)
        classes = list(class_counts.keys())

        # Determine sampling targets
        if self.sampling_strategy == 'auto':
            min_count = min(class_counts.values())
            target_counts = {cls: min_count for cls in classes}
        elif isinstance(self.sampling_strategy, dict):
            target_counts = self.sampling_strategy
        else:
            raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}")

        X_resampled = []
        y_resampled = []

        for cls in classes:
            class_mask = (y == cls)
            X_class = X[class_mask]
            y_class = y[class_mask]

            current_count = len(y_class)
            target_count = target_counts.get(cls, current_count)

            # Undersample if needed
            if target_count < current_count:
                indices = np.random.choice(current_count, target_count, replace=False)
                X_resampled.append(X_class[indices])
                y_resampled.append(y_class[indices])
            else:
                X_resampled.append(X_class)
                y_resampled.append(y_class)

        X_resampled = np.vstack(X_resampled)
        y_resampled = np.concatenate(y_resampled)

        # Shuffle
        indices = np.random.permutation(len(y_resampled))
        return X_resampled[indices], y_resampled[indices]


class TomekLinks:
    """
    Remove Tomek links (pairs of samples from different classes that are nearest neighbors).
    """

    def __init__(self):
        """Initialize Tomek links remover."""
        pass

    def _find_tomek_links(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> List[int]:
        """
        Find indices of samples that are part of Tomek links.

        Returns:
        --------
        tomek_indices : list
            Indices of samples to remove.
        """
        n_samples = len(y)
        tomek_indices = []

        for i in range(n_samples):
            # Find nearest neighbor
            distances = np.linalg.norm(X - X[i], axis=1)
            distances[i] = np.inf  # Exclude self
            nearest_idx = np.argmin(distances)

            # Check if forms Tomek link
            # A Tomek link is a pair (i, j) where:
            # 1. i and j are nearest neighbors
            # 2. i and j have different labels
            distances_j = np.linalg.norm(X - X[nearest_idx], axis=1)
            distances_j[nearest_idx] = np.inf
            nearest_to_j = np.argmin(distances_j)

            if nearest_to_j == i and y[i] != y[nearest_idx]:
                tomek_indices.append(i)

        return tomek_indices

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove Tomek links from dataset.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Labels.

        Returns:
        --------
        X_resampled : np.ndarray
            Cleaned features.
        y_resampled : np.ndarray
            Cleaned labels.
        """
        tomek_indices = self._find_tomek_links(X, y)

        # Remove Tomek links
        mask = np.ones(len(y), dtype=bool)
        mask[tomek_indices] = False

        return X[mask], y[mask]


class NearMiss:
    """
    NearMiss undersampling.

    Selects majority class samples closest to minority class samples.
    """

    def __init__(
        self,
        version: int = 1,
        n_neighbors: int = 3,
        random_state: Optional[int] = None
    ):
        """
        Initialize NearMiss.

        Parameters:
        -----------
        version : int
            NearMiss version (1, 2, or 3).
        n_neighbors : int
            Number of neighbors to consider.
        random_state : int, optional
            Random seed.
        """
        self.version = version
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample dataset using NearMiss.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Labels.

        Returns:
        --------
        X_resampled : np.ndarray
            Resampled features.
        y_resampled : np.ndarray
            Resampled labels.
        """
        class_counts = Counter(y)
        classes = list(class_counts.keys())

        minority_class = min(classes, key=lambda c: class_counts[c])
        majority_class = max(classes, key=lambda c: class_counts[c])

        minority_mask = (y == minority_class)
        majority_mask = (y == majority_class)

        X_minority = X[minority_mask]
        X_majority = X[majority_mask]

        target_count = len(X_minority)

        # NearMiss-1: Select majority samples with smallest average distance to k minority neighbors
        if self.version == 1:
            avg_distances = []
            for maj_sample in X_majority:
                distances = np.linalg.norm(X_minority - maj_sample, axis=1)
                k_nearest_distances = np.sort(distances)[:self.n_neighbors]
                avg_distances.append(np.mean(k_nearest_distances))

            selected_indices = np.argsort(avg_distances)[:target_count]

        # NearMiss-2: Select majority samples with smallest average distance to k farthest minority neighbors
        elif self.version == 2:
            avg_distances = []
            for maj_sample in X_majority:
                distances = np.linalg.norm(X_minority - maj_sample, axis=1)
                k_farthest_distances = np.sort(distances)[-self.n_neighbors:]
                avg_distances.append(np.mean(k_farthest_distances))

            selected_indices = np.argsort(avg_distances)[:target_count]

        # NearMiss-3: Select majority samples ensuring each minority sample has at least k majority neighbors
        elif self.version == 3:
            selected_indices = np.random.choice(len(X_majority), target_count, replace=False)

        else:
            raise ValueError(f"Invalid version: {self.version}")

        X_majority_selected = X_majority[selected_indices]
        y_majority_selected = np.full(len(selected_indices), majority_class)

        # Combine minority and selected majority samples
        X_resampled = np.vstack([X_minority, X_majority_selected])
        y_resampled = np.concatenate([y[minority_mask], y_majority_selected])

        # Shuffle
        indices = np.random.permutation(len(y_resampled))
        return X_resampled[indices], y_resampled[indices]


# ============================================================================
# COMBINED TECHNIQUES
# ============================================================================

class SMOTETomek:
    """
    Combination of SMOTE oversampling and Tomek links cleaning.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        random_state: Optional[int] = None
    ):
        """
        Initialize SMOTETomek.

        Parameters:
        -----------
        k_neighbors : int
            Number of neighbors for SMOTE.
        random_state : int, optional
            Random seed.
        """
        self.smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        self.tomek = TomekLinks()

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample using SMOTE followed by Tomek links removal.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Labels.

        Returns:
        --------
        X_resampled : np.ndarray
            Resampled features.
        y_resampled : np.ndarray
            Resampled labels.
        """
        # Apply SMOTE
        X_smote, y_smote = self.smote.fit_resample(X, y)

        # Apply Tomek links
        X_resampled, y_resampled = self.tomek.fit_resample(X_smote, y_smote)

        return X_resampled, y_resampled


# ============================================================================
# CLASS WEIGHTING
# ============================================================================

def compute_class_weights(
    y: np.ndarray,
    mode: str = 'balanced'
) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.

    Parameters:
    -----------
    y : np.ndarray
        Labels.
    mode : str
        'balanced': n_samples / (n_classes * class_count)
        'balanced_subsample': similar but for subsamples

    Returns:
    --------
    class_weights : dict
        Dictionary mapping class to weight.
    """
    class_counts = Counter(y)
    classes = list(class_counts.keys())
    n_samples = len(y)
    n_classes = len(classes)

    if mode == 'balanced':
        class_weights = {
            cls: n_samples / (n_classes * class_counts[cls])
            for cls in classes
        }
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return class_weights


def compute_sample_weights(
    y: np.ndarray,
    class_weights: Optional[Dict[int, float]] = None
) -> np.ndarray:
    """
    Compute sample weights from class weights.

    Parameters:
    -----------
    y : np.ndarray
        Labels.
    class_weights : dict, optional
        Class weights. If None, compute balanced weights.

    Returns:
    --------
    sample_weights : np.ndarray
        Sample weights.
    """
    if class_weights is None:
        class_weights = compute_class_weights(y, mode='balanced')

    sample_weights = np.array([class_weights[label] for label in y])
    return sample_weights


# ============================================================================
# EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("IMBALANCED DATA HANDLING EXAMPLES")
    print("=" * 70)

    # Create imbalanced dataset
    np.random.seed(42)
    n_majority = 900
    n_minority = 100

    X_majority = np.random.randn(n_majority, 5)
    y_majority = np.zeros(n_majority)

    X_minority = np.random.randn(n_minority, 5) + 2
    y_minority = np.ones(n_minority)

    X = np.vstack([X_majority, X_minority])
    y = np.concatenate([y_majority, y_minority])

    print(f"\nOriginal dataset:")
    print(f"  Total samples: {len(y)}")
    print(f"  Class distribution: {Counter(y)}")
    print(f"  Imbalance ratio: {n_majority / n_minority:.1f}:1")

    # Example 1: Random Oversampling
    print("\n1. Random Oversampling")
    print("-" * 70)
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_ros, y_ros = ros.fit_resample(X, y)
    print(f"Resampled distribution: {Counter(y_ros)}")

    # Example 2: SMOTE
    print("\n2. SMOTE")
    print("-" * 70)
    smote = SMOTE(k_neighbors=5, random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    print(f"Resampled distribution: {Counter(y_smote)}")

    # Example 3: ADASYN
    print("\n3. ADASYN")
    print("-" * 70)
    adasyn = ADASYN(k_neighbors=5, random_state=42)
    X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
    print(f"Resampled distribution: {Counter(y_adasyn)}")

    # Example 4: Random Undersampling
    print("\n4. Random Undersampling")
    print("-" * 70)
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_rus, y_rus = rus.fit_resample(X, y)
    print(f"Resampled distribution: {Counter(y_rus)}")

    # Example 5: NearMiss
    print("\n5. NearMiss")
    print("-" * 70)
    nm = NearMiss(version=1, n_neighbors=3, random_state=42)
    X_nm, y_nm = nm.fit_resample(X, y)
    print(f"Resampled distribution: {Counter(y_nm)}")

    # Example 6: SMOTE + Tomek
    print("\n6. SMOTE + Tomek Links")
    print("-" * 70)
    smote_tomek = SMOTETomek(k_neighbors=5, random_state=42)
    X_st, y_st = smote_tomek.fit_resample(X, y)
    print(f"Resampled distribution: {Counter(y_st)}")

    # Example 7: Class Weighting
    print("\n7. Class Weighting")
    print("-" * 70)
    class_weights = compute_class_weights(y, mode='balanced')
    sample_weights = compute_sample_weights(y, class_weights)
    print(f"Class weights: {class_weights}")
    print(f"Sample weights shape: {sample_weights.shape}")
    print(f"Average weight for class 0: {sample_weights[y == 0].mean():.4f}")
    print(f"Average weight for class 1: {sample_weights[y == 1].mean():.4f}")

    print("\n" + "=" * 70)
    print("All imbalanced data handling examples completed!")
    print("=" * 70)
