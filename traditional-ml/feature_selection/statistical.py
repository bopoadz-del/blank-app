"""
Statistical Feature Selection
Univariate feature selection using statistical tests
"""

import numpy as np
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFpr, SelectFdr, SelectFwe,
    chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression
)
from typing import Union, Optional, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnivariateSelector:
    """
    Univariate Feature Selection

    Selects features based on univariate statistical tests.
    Each feature is independently evaluated against the target.

    Score functions:
    - Classification: 'chi2', 'f_classif', 'mutual_info_classif'
    - Regression: 'f_regression', 'mutual_info_regression'

    Selection methods:
    - SelectKBest: Select k highest scoring features
    - SelectPercentile: Select top percentile of features
    - SelectFpr/Fdr/Fwe: Select based on p-value thresholds

    Best for:
    - Quick feature reduction
    - Removing irrelevant features
    - Baseline feature selection
    """

    def __init__(
        self,
        score_func: Union[str, Callable] = 'f_classif',
        mode: str = 'k_best',
        k: int = 10,
        percentile: int = 10,
        alpha: float = 0.05
    ):
        """
        Initialize Univariate Selector

        Args:
            score_func: Scoring function name or callable
            mode: Selection mode ('k_best', 'percentile', 'fpr', 'fdr', 'fwe')
            k: Number of features to select (for k_best)
            percentile: Percentile of features to select
            alpha: Alpha level for FPR/FDR/FWE
        """
        self.score_func = score_func
        self.mode = mode
        self.k = k
        self.percentile = percentile
        self.alpha = alpha

        # Map string names to functions
        score_func_map = {
            'chi2': chi2,
            'f_classif': f_classif,
            'f_regression': f_regression,
            'mutual_info_classif': mutual_info_classif,
            'mutual_info_regression': mutual_info_regression
        }

        if isinstance(score_func, str):
            if score_func not in score_func_map:
                raise ValueError(f"Unknown score function: {score_func}")
            score_func_callable = score_func_map[score_func]
        else:
            score_func_callable = score_func

        # Create selector based on mode
        if mode == 'k_best':
            self.model = SelectKBest(score_func=score_func_callable, k=k)
        elif mode == 'percentile':
            self.model = SelectPercentile(score_func=score_func_callable, percentile=percentile)
        elif mode == 'fpr':
            self.model = SelectFpr(score_func=score_func_callable, alpha=alpha)
        elif mode == 'fdr':
            self.model = SelectFdr(score_func=score_func_callable, alpha=alpha)
        elif mode == 'fwe':
            self.model = SelectFwe(score_func=score_func_callable, alpha=alpha)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.scores_ = None
        self.pvalues_ = None
        self.selected_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the selector"""
        logger.info(f"Fitting univariate selector ({self.score_func}, {self.mode})...")
        self.model.fit(X, y)
        self.scores_ = self.model.scores_
        self.pvalues_ = getattr(self.model, 'pvalues_', None)
        self.selected_features_ = self.model.get_support(indices=True)

        logger.info(f"Selected {len(self.selected_features_)} features")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        return self.model.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get selected feature mask or indices"""
        return self.model.get_support(indices=indices)

    def get_scores(self) -> np.ndarray:
        """Get feature scores"""
        if self.scores_ is None:
            raise ValueError("Selector not fitted yet")
        return self.scores_

    def get_feature_names_out(self, feature_names: Optional[list] = None) -> list:
        """Get names of selected features"""
        if feature_names is None:
            return list(self.selected_features_)

        return [feature_names[i] for i in self.selected_features_]


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Create dataset with irrelevant features
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=10,
        n_redundant=5,
        n_repeated=0,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("=" * 70)
    print("Univariate Feature Selection Test")
    print("=" * 70)
    print(f"\nOriginal features: {X_train.shape[1]}")

    # Test SelectKBest
    print("\n1. SelectKBest (k=10)")
    print("-" * 70)
    selector_k = UnivariateSelector(score_func='f_classif', mode='k_best', k=10)
    X_train_selected = selector_k.fit_transform(X_train, y_train)
    X_test_selected = selector_k.transform(X_test)

    print(f"Selected features: {X_train_selected.shape[1]}")
    print(f"Feature indices: {selector_k.get_support(indices=True)[:10]}...")

    # Train model on selected features
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_selected, y_train)
    y_pred = rf.predict(X_test_selected)
    print(f"Accuracy (selected): {accuracy_score(y_test, y_pred):.4f}")

    # Compare with all features
    rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_all.fit(X_train, y_train)
    y_pred_all = rf_all.predict(X_test)
    print(f"Accuracy (all): {accuracy_score(y_test, y_pred_all):.4f}")

    # Test SelectPercentile
    print("\n2. SelectPercentile (percentile=20)")
    print("-" * 70)
    selector_p = UnivariateSelector(score_func='f_classif', mode='percentile', percentile=20)
    X_train_p = selector_p.fit_transform(X_train, y_train)
    print(f"Selected features: {X_train_p.shape[1]}")

    # Test Mutual Information
    print("\n3. Mutual Information (k=10)")
    print("-" * 70)
    selector_mi = UnivariateSelector(score_func='mutual_info_classif', mode='k_best', k=10)
    X_train_mi = selector_mi.fit_transform(X_train, y_train)
    print(f"Selected features: {X_train_mi.shape[1]}")

    # Show top features by score
    print("\nTop 5 features by F-score:")
    scores = selector_k.get_scores()
    top_idx = np.argsort(scores)[::-1][:5]
    for i, idx in enumerate(top_idx, 1):
        print(f"  {i}. Feature {idx}: {scores[idx]:.2f}")

    print("\n" + "=" * 70)
    print("Univariate feature selection tested successfully!")
