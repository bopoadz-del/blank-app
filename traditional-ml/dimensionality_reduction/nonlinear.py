"""
Non-linear Dimensionality Reduction
t-SNE and manifold learning
"""

import numpy as np
from sklearn.manifold import TSNE
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TSNEReducer:
    """
    t-Distributed Stochastic Neighbor Embedding

    Non-linear dimensionality reduction for visualization.
    Preserves local structure (nearby points stay nearby).

    Best for:
    - Visualization (2D/3D plots)
    - Exploring cluster structure
    - Understanding data relationships

    Important notes:
    - Mainly for visualization, not feature extraction
    - Computationally expensive
    - Non-deterministic
    - No inverse_transform (can't go back to original space)
    - Perplexity should be 5-50 (related to number of nearest neighbors)
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: Union[float, str] = 'auto',
        n_iter: int = 1000,
        metric: str = 'euclidean',
        init: str = 'pca',
        random_state: int = 42,
        n_jobs: Optional[int] = None,
        verbose: int = 0
    ):
        """
        Initialize t-SNE

        Args:
            n_components: Dimensions of embedded space (typically 2 or 3)
            perplexity: Related to number of nearest neighbors (5-50)
            learning_rate: Learning rate ('auto' or float 10-1000)
            n_iter: Maximum number of iterations
            metric: Distance metric
            init: Initialization ('pca' or 'random')
            random_state: Random seed
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.metric = metric
        self.init = init
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Create model
        self.model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            metric=metric,
            init=init,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose
        )

        self.embedding_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data

        Note: t-SNE doesn't have separate fit() and transform() methods
        """
        logger.info(f"Computing t-SNE embedding (n_components={self.n_components}, perplexity={self.perplexity})...")
        logger.info("This may take a while...")

        self.embedding_ = self.model.fit_transform(X)

        logger.info("t-SNE embedding complete")
        return self.embedding_

    def get_embedding(self) -> np.ndarray:
        """Get the embedding"""
        if self.embedding_ is None:
            raise ValueError("Model not fitted yet")
        return self.embedding_


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler
    import time

    print("=" * 70)
    print("t-SNE Test")
    print("=" * 70)

    # Load digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {len(np.unique(y))}")

    # Test t-SNE with default parameters
    print("\n1. t-SNE (perplexity=30)")
    print("-" * 70)
    start_time = time.time()

    tsne = TSNEReducer(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        random_state=42
    )
    X_tsne = tsne.fit_transform(X_scaled)

    elapsed = time.time() - start_time

    print(f"Original shape: {X_scaled.shape}")
    print(f"Embedded shape: {X_tsne.shape}")
    print(f"Time elapsed: {elapsed:.2f}s")

    # Test with different perplexity
    print("\n2. t-SNE (perplexity=50)")
    print("-" * 70)
    start_time = time.time()

    tsne_50 = TSNEReducer(
        n_components=2,
        perplexity=50,
        n_iter=1000,
        random_state=42
    )
    X_tsne_50 = tsne_50.fit_transform(X_scaled)

    elapsed = time.time() - start_time
    print(f"Time elapsed: {elapsed:.2f}s")

    print("\n" + "=" * 70)
    print("t-SNE tested successfully!")
