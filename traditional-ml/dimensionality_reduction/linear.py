"""
Linear Dimensionality Reduction
PCA and LDA
"""

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PCAReducer:
    """
    Principal Component Analysis

    Linear dimensionality reduction using SVD.
    Projects data onto principal components (directions of maximum variance).

    Best for:
    - Reducing feature dimensions
    - Removing multicollinearity
    - Data visualization
    - Noise reduction
    - Feature extraction

    Note: PCA is unsupervised (doesn't use labels)
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        whiten: bool = False,
        random_state: int = 42
    ):
        """
        Initialize PCA

        Args:
            n_components: Number of components (int) or variance to keep (float 0-1)
            whiten: Whether to whiten (zero mean, unit variance) the components
            random_state: Random seed
        """
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

        # Create model
        self.model = PCA(
            n_components=n_components,
            whiten=whiten,
            random_state=random_state
        )

        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray):
        """Fit the model"""
        logger.info(f"Fitting PCA...")
        self.model.fit(X)
        self.components_ = self.model.components_
        self.explained_variance_ = self.model.explained_variance_
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_
        self.mean_ = self.model.mean_

        total_variance = np.sum(self.explained_variance_ratio_)
        logger.info(f"PCA fitted. Components: {len(self.components_)}")
        logger.info(f"Total variance explained: {total_variance:.4f}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to principal components"""
        return self.model.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform back to original space"""
        return self.model.inverse_transform(X)

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get proportion of variance explained by each component"""
        if self.explained_variance_ratio_ is None:
            raise ValueError("Model not fitted yet")
        return self.explained_variance_ratio_

    def get_cumulative_variance_ratio(self) -> np.ndarray:
        """Get cumulative variance explained"""
        if self.explained_variance_ratio_ is None:
            raise ValueError("Model not fitted yet")
        return np.cumsum(self.explained_variance_ratio_)

    def get_components(self) -> np.ndarray:
        """Get principal components"""
        if self.components_ is None:
            raise ValueError("Model not fitted yet")
        return self.components_


class LDAReducer:
    """
    Linear Discriminant Analysis

    Supervised dimensionality reduction.
    Finds linear combinations of features that best separate classes.

    Best for:
    - Supervised dimensionality reduction
    - Classification preprocessing
    - When you want to maximize class separability

    Note: LDA is supervised (uses labels)
    Maximum components = n_classes - 1
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        solver: str = 'svd',
        shrinkage: Optional[Union[str, float]] = None
    ):
        """
        Initialize LDA

        Args:
            n_components: Number of components (max = n_classes - 1)
            solver: Solver ('svd', 'lsqr', 'eigen')
            shrinkage: Shrinkage parameter (None, 'auto', or float 0-1)
        """
        self.n_components = n_components
        self.solver = solver
        self.shrinkage = shrinkage

        # Create model
        self.model = LinearDiscriminantAnalysis(
            n_components=n_components,
            solver=solver,
            shrinkage=shrinkage
        )

        self.scalings_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model"""
        logger.info("Fitting LDA...")
        self.model.fit(X, y)
        self.scalings_ = self.model.scalings_
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_

        logger.info(f"LDA fitted. Components: {self.model.n_components}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        return self.model.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get variance explained by each component"""
        if self.explained_variance_ratio_ is None:
            raise ValueError("Model not fitted yet")
        return self.explained_variance_ratio_


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris, make_classification
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("Dimensionality Reduction Test")
    print("=" * 70)

    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test PCA
    print("\n1. PCA (n_components=2)")
    print("-" * 70)
    pca = PCAReducer(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Original shape: {X_scaled.shape}")
    print(f"Transformed shape: {X_pca.shape}")
    print(f"Explained variance ratio: {pca.get_explained_variance_ratio()}")
    print(f"Total variance explained: {np.sum(pca.get_explained_variance_ratio()):.4f}")

    # Test PCA with variance threshold
    print("\n2. PCA (variance threshold=0.95)")
    print("-" * 70)
    pca_var = PCAReducer(n_components=0.95)
    X_pca_var = pca_var.fit_transform(X_scaled)
    print(f"Components selected: {X_pca_var.shape[1]}")
    print(f"Total variance explained: {np.sum(pca_var.get_explained_variance_ratio()):.4f}")

    # Test LDA
    print("\n3. LDA (n_components=2)")
    print("-" * 70)
    lda = LDAReducer(n_components=2)
    X_lda = lda.fit_transform(X_scaled, y)
    print(f"Original shape: {X_scaled.shape}")
    print(f"Transformed shape: {X_lda.shape}")
    print(f"Explained variance ratio: {lda.get_explained_variance_ratio()}")

    print("\n" + "=" * 70)
    print("Dimensionality reduction tested successfully!")
