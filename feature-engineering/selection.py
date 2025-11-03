"""
Feature Selection and Dimensionality Reduction
Comprehensive utilities for feature selection and dimensionality reduction
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFromModel,
    RFE, RFECV, SequentialFeatureSelector,
    chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    VarianceThreshold
)
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, NMF, FactorAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from typing import List, Optional, Union, Dict, Tuple
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Comprehensive feature selection

    Methods:
    - Statistical: Chi2, ANOVA F-test, Mutual Information
    - Model-based: Lasso, Random Forest, Gradient Boosting
    - Wrapper: Recursive Feature Elimination (RFE)
    - Sequential: Forward/Backward selection
    - Filter: Variance threshold, correlation

    Usage:
        selector = FeatureSelector(method='mutual_info', k=10)
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_selected_features()
    """

    def __init__(
        self,
        method: str = 'mutual_info',
        k: Union[int, str] = 10,
        estimator: Optional[any] = None,
        scoring: Optional[str] = None,
        cv: int = 5
    ):
        """
        Initialize feature selector

        Args:
            method: Selection method
                - 'chi2': Chi-squared test (classification, non-negative features)
                - 'f_classif': ANOVA F-test (classification)
                - 'f_regression': F-test (regression)
                - 'mutual_info': Mutual information
                - 'variance': Variance threshold
                - 'model': Model-based selection (L1/tree-based)
                - 'rfe': Recursive Feature Elimination
                - 'rfecv': RFE with cross-validation
                - 'sequential': Sequential feature selection
            k: Number of features to select (or 'all')
            estimator: Estimator for model-based/RFE/sequential selection
            scoring: Scoring function for RFECV
            cv: Cross-validation folds for RFECV
        """
        self.method = method
        self.k = k
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv

        self.selector = None
        self.fitted = False
        self.selected_features_ = None
        self.feature_scores_ = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        """Fit feature selector"""
        is_dataframe = isinstance(X, pd.DataFrame)
        feature_names = X.columns.tolist() if is_dataframe else None

        # Create selector based on method
        if self.method == 'chi2':
            self.selector = SelectKBest(chi2, k=self.k)

        elif self.method == 'f_classif':
            self.selector = SelectKBest(f_classif, k=self.k)

        elif self.method == 'f_regression':
            self.selector = SelectKBest(f_regression, k=self.k)

        elif self.method == 'mutual_info':
            # Determine if classification or regression
            if len(np.unique(y)) < 20:  # Heuristic for classification
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression
            self.selector = SelectKBest(score_func, k=self.k)

        elif self.method == 'variance':
            # Variance threshold
            threshold = self.k if isinstance(self.k, float) else 0.0
            self.selector = VarianceThreshold(threshold=threshold)

        elif self.method == 'model':
            # Model-based selection
            if self.estimator is None:
                # Default to Lasso for regression, Random Forest for classification
                if len(np.unique(y)) < 20:
                    self.estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    self.estimator = Lasso(alpha=0.1, random_state=42)

            self.selector = SelectFromModel(self.estimator, max_features=self.k if self.k != 'all' else None)

        elif self.method == 'rfe':
            # Recursive Feature Elimination
            if self.estimator is None:
                self.estimator = RandomForestClassifier(n_estimators=100, random_state=42)

            n_features = self.k if self.k != 'all' else X.shape[1] // 2
            self.selector = RFE(self.estimator, n_features_to_select=n_features)

        elif self.method == 'rfecv':
            # RFE with cross-validation
            if self.estimator is None:
                self.estimator = RandomForestClassifier(n_estimators=100, random_state=42)

            self.selector = RFECV(
                self.estimator,
                min_features_to_select=self.k if isinstance(self.k, int) else 1,
                cv=self.cv,
                scoring=self.scoring
            )

        elif self.method == 'sequential':
            # Sequential feature selection
            if self.estimator is None:
                self.estimator = RandomForestClassifier(n_estimators=100, random_state=42)

            self.selector = SequentialFeatureSelector(
                self.estimator,
                n_features_to_select=self.k if self.k != 'all' else 'auto',
                direction='forward',
                cv=self.cv,
                scoring=self.scoring
            )

        else:
            raise ValueError(f"Unknown selection method: {self.method}")

        # Fit selector
        self.selector.fit(X, y)

        # Get selected features
        if hasattr(self.selector, 'get_support'):
            mask = self.selector.get_support()
            if feature_names is not None:
                self.selected_features_ = [feature_names[i] for i, selected in enumerate(mask) if selected]
            else:
                self.selected_features_ = [i for i, selected in enumerate(mask) if selected]

        # Get feature scores if available
        if hasattr(self.selector, 'scores_'):
            self.feature_scores_ = self.selector.scores_
        elif hasattr(self.selector, 'ranking_'):
            self.feature_scores_ = self.selector.ranking_
        elif hasattr(self.selector, 'estimator_') and hasattr(self.selector.estimator_, 'feature_importances_'):
            self.feature_scores_ = self.selector.estimator_.feature_importances_

        self.fitted = True
        logger.info(f"FeatureSelector ({self.method}) fitted - {len(self.selected_features_)} features selected")

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data"""
        if not self.fitted:
            raise ValueError("Selector not fitted. Call fit() first.")
        return self.selector.transform(X)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> np.ndarray:
        """Fit and transform data"""
        self.fit(X, y)
        return self.transform(X)

    def get_selected_features(self) -> List:
        """Get list of selected features"""
        return self.selected_features_

    def get_feature_scores(self) -> Optional[np.ndarray]:
        """Get feature importance scores"""
        return self.feature_scores_

    def plot_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        top_k: int = 20
    ):
        """Plot feature importance"""
        if self.feature_scores_ is None:
            logger.warning("Feature scores not available for this method")
            return

        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(self.feature_scores_))]

        # Get top k features
        indices = np.argsort(self.feature_scores_)[-top_k:][::-1]

        plt.figure(figsize=(10, 6))
        plt.barh(range(top_k), self.feature_scores_[indices])
        plt.yticks(range(top_k), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_k} Features ({self.method})')
        plt.tight_layout()
        plt.show()


class DimensionalityReducer:
    """
    Dimensionality reduction techniques

    Linear Methods:
    - PCA: Principal Component Analysis
    - LDA: Linear Discriminant Analysis
    - SVD: Singular Value Decomposition
    - Factor Analysis

    Non-linear Methods:
    - t-SNE: t-Distributed Stochastic Neighbor Embedding
    - Isomap: Isometric Mapping
    - LLE: Locally Linear Embedding
    - MDS: Multidimensional Scaling

    Other:
    - ICA: Independent Component Analysis
    - NMF: Non-negative Matrix Factorization

    Usage:
        reducer = DimensionalityReducer(method='pca', n_components=10)
        X_reduced = reducer.fit_transform(X)
    """

    def __init__(
        self,
        method: str = 'pca',
        n_components: Union[int, float] = 2,
        **kwargs
    ):
        """
        Initialize dimensionality reducer

        Args:
            method: Reduction method
                - 'pca': Principal Component Analysis
                - 'lda': Linear Discriminant Analysis (supervised)
                - 'svd': Truncated SVD
                - 'tsne': t-SNE
                - 'isomap': Isomap
                - 'lle': Locally Linear Embedding
                - 'mds': Multidimensional Scaling
                - 'ica': Independent Component Analysis
                - 'nmf': Non-negative Matrix Factorization
                - 'fa': Factor Analysis
            n_components: Number of components (or variance ratio for PCA)
            **kwargs: Additional arguments for the reducer
        """
        self.method = method
        self.n_components = n_components

        if method == 'pca':
            self.reducer = PCA(n_components=n_components, **kwargs)

        elif method == 'lda':
            self.reducer = LDA(n_components=n_components, **kwargs)

        elif method == 'svd':
            self.reducer = TruncatedSVD(n_components=n_components, **kwargs)

        elif method == 'tsne':
            # t-SNE parameters
            perplexity = kwargs.get('perplexity', 30)
            learning_rate = kwargs.get('learning_rate', 200)
            n_iter = kwargs.get('n_iter', 1000)

            self.reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=42
            )

        elif method == 'isomap':
            n_neighbors = kwargs.get('n_neighbors', 5)
            self.reducer = Isomap(n_components=n_components, n_neighbors=n_neighbors)

        elif method == 'lle':
            n_neighbors = kwargs.get('n_neighbors', 5)
            self.reducer = LocallyLinearEmbedding(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=42
            )

        elif method == 'mds':
            self.reducer = MDS(n_components=n_components, random_state=42, **kwargs)

        elif method == 'ica':
            self.reducer = FastICA(n_components=n_components, random_state=42, **kwargs)

        elif method == 'nmf':
            self.reducer = NMF(n_components=n_components, random_state=42, **kwargs)

        elif method == 'fa':
            self.reducer = FactorAnalysis(n_components=n_components, random_state=42, **kwargs)

        else:
            raise ValueError(f"Unknown reduction method: {method}")

        self.fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
        """Fit reducer to data"""
        # LDA requires labels
        if self.method == 'lda':
            if y is None:
                raise ValueError("LDA requires target variable y")
            self.reducer.fit(X, y)
        # t-SNE doesn't need fit
        elif self.method == 'tsne':
            pass
        else:
            self.reducer.fit(X)

        self.fitted = True
        logger.info(f"DimensionalityReducer ({self.method}) fitted - {self.n_components} components")
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data"""
        if self.method == 'tsne':
            # t-SNE transforms during fit
            logger.warning("t-SNE doesn't support transform(). Use fit_transform() instead.")
            return self.fit_transform(X)

        if not self.fitted:
            raise ValueError("Reducer not fitted. Call fit() first.")

        return self.reducer.transform(X)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform data"""
        if self.method == 'lda':
            return self.reducer.fit_transform(X, y)
        elif self.method == 'tsne':
            return self.reducer.fit_transform(X)
        else:
            return self.reducer.fit_transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform (not available for all methods)"""
        if self.method in ['tsne', 'mds', 'lda']:
            raise ValueError(f"{self.method} does not support inverse transform")

        if not self.fitted:
            raise ValueError("Reducer not fitted. Call fit() first.")

        return self.reducer.inverse_transform(X)

    def get_explained_variance(self) -> Optional[np.ndarray]:
        """Get explained variance ratio (for PCA, SVD, FA)"""
        if hasattr(self.reducer, 'explained_variance_ratio_'):
            return self.reducer.explained_variance_ratio_
        return None

    def plot_explained_variance(self, cumulative: bool = True):
        """Plot explained variance (for PCA, SVD, FA)"""
        variance = self.get_explained_variance()

        if variance is None:
            logger.warning("Explained variance not available for this method")
            return

        plt.figure(figsize=(10, 6))

        if cumulative:
            plt.plot(np.cumsum(variance), marker='o')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('Cumulative Explained Variance by Component')
        else:
            plt.bar(range(len(variance)), variance)
            plt.ylabel('Explained Variance Ratio')
            plt.title('Explained Variance by Component')

        plt.xlabel('Component')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_2d(
        self,
        X_reduced: np.ndarray,
        y: Optional[np.ndarray] = None,
        title: Optional[str] = None
    ):
        """Plot 2D projection"""
        if X_reduced.shape[1] != 2:
            logger.warning("Data must have 2 components for 2D plot")
            return

        plt.figure(figsize=(10, 8))

        if y is not None:
            scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter)
        else:
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6)

        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(title or f'{self.method.upper()} 2D Projection')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Feature Selection and Dimensionality Reduction Test")
    print("=" * 70)

    # Create sample data
    from sklearn.datasets import make_classification, make_regression

    # Classification dataset
    X_class, y_class = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=20,
        n_redundant=10,
        n_repeated=5,
        random_state=42
    )

    print(f"Classification dataset: {X_class.shape}")
    print(f"Classes: {np.unique(y_class)}")

    # Test Feature Selection
    print("\n1. Feature Selection (Mutual Information)")
    print("-" * 70)

    selector = FeatureSelector(method='mutual_info', k=10)
    X_selected = selector.fit_transform(X_class, y_class)

    print(f"Original features: {X_class.shape[1]}")
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Selected indices: {selector.get_selected_features()[:10]}")

    # Test Model-based Selection
    print("\n2. Model-based Feature Selection (Random Forest)")
    print("-" * 70)

    selector = FeatureSelector(method='model', k=15)
    X_selected = selector.fit_transform(X_class, y_class)

    print(f"Selected features: {X_selected.shape[1]}")

    # Test RFE
    print("\n3. Recursive Feature Elimination (RFE)")
    print("-" * 70)

    selector = FeatureSelector(method='rfe', k=10)
    X_selected = selector.fit_transform(X_class, y_class)

    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Feature ranking: {selector.selector.ranking_[:10]}")

    # Test PCA
    print("\n4. Principal Component Analysis (PCA)")
    print("-" * 70)

    reducer = DimensionalityReducer(method='pca', n_components=10)
    X_pca = reducer.fit_transform(X_class)

    print(f"Original dimensions: {X_class.shape[1]}")
    print(f"Reduced dimensions: {X_pca.shape[1]}")

    variance = reducer.get_explained_variance()
    if variance is not None:
        print(f"Explained variance: {variance[:5]}")
        print(f"Cumulative variance (5 components): {np.sum(variance[:5]):.4f}")

    # Test t-SNE
    print("\n5. t-SNE Dimensionality Reduction")
    print("-" * 70)

    reducer = DimensionalityReducer(method='tsne', n_components=2, perplexity=30)
    X_tsne = reducer.fit_transform(X_class[:500])  # Use subset for speed

    print(f"t-SNE output shape: {X_tsne.shape}")
    print(f"t-SNE range: [{X_tsne.min():.2f}, {X_tsne.max():.2f}]")

    # Test LDA
    print("\n6. Linear Discriminant Analysis (LDA)")
    print("-" * 70)

    reducer = DimensionalityReducer(method='lda', n_components=1)  # n_classes - 1 max
    X_lda = reducer.fit_transform(X_class, y_class)

    print(f"LDA output shape: {X_lda.shape}")

    print("\n" + "=" * 70)
    print("Feature selection and dimensionality reduction tested successfully!")
