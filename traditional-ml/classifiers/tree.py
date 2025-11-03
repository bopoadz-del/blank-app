"""
Decision Tree Classifier
CART algorithm implementation
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecisionTreeClassifierWrapper:
    """
    Decision Tree Classifier

    Tree-based model that learns decision rules from features.
    Uses CART (Classification and Regression Trees) algorithm.

    Best for:
    - Interpretability (can visualize tree)
    - Mixed data types
    - Non-linear relationships
    - Feature interactions
    - Quick baseline models

    Cons:
    - Prone to overfitting
    - Unstable (small changes in data = different tree)
    - Biased toward features with more levels
    """

    def __init__(
        self,
        criterion: str = 'gini',
        splitter: str = 'best',
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Optional[Union[int, float, str]] = None,
        class_weight: Optional[Union[str, dict]] = None,
        random_state: int = 42,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0
    ):
        """
        Initialize Decision Tree Classifier

        Args:
            criterion: Split quality measure ('gini' or 'entropy')
            splitter: Split strategy ('best' or 'random')
            max_depth: Maximum depth of tree (None = unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            max_features: Number of features to consider for best split
            class_weight: Weights for classes ('balanced' or dict)
            random_state: Random seed
            max_leaf_nodes: Maximum number of leaf nodes
            min_impurity_decrease: Minimum impurity decrease to split
        """
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

        # Create model
        self.model = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease
        )

        self.feature_importances_ = None
        self.classes_ = None
        self.tree_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        logger.info("Training Decision Tree...")
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        self.classes_ = self.model.classes_
        self.tree_ = self.model.tree_
        logger.info(f"Training complete. Tree depth: {self.model.get_depth()}")
        logger.info(f"Number of leaves: {self.model.get_n_leaves()}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances"""
        if self.feature_importances_ is None:
            raise ValueError("Model not fitted yet")
        return self.feature_importances_

    def get_depth(self) -> int:
        """Get depth of the tree"""
        return self.model.get_depth()

    def get_n_leaves(self) -> int:
        """Get number of leaves"""
        return self.model.get_n_leaves()

    def export_text(self, feature_names: Optional[list] = None) -> str:
        """
        Export tree as text

        Args:
            feature_names: Names of features

        Returns:
            Text representation of tree
        """
        from sklearn.tree import export_text as sklearn_export_text
        return sklearn_export_text(self.model, feature_names=feature_names)

    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters"""
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """Set model parameters"""
        self.model.set_params(**params)
        return self


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Create dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("=" * 70)
    print("Decision Tree Classifiers Test")
    print("=" * 70)

    # Test Gini criterion
    print("\n1. Gini Criterion (unpruned)")
    print("-" * 70)
    dt_gini = DecisionTreeClassifierWrapper(criterion='gini')
    dt_gini.fit(X_train, y_train)
    y_pred = dt_gini.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Tree depth: {dt_gini.get_depth()}")
    print(f"Number of leaves: {dt_gini.get_n_leaves()}")

    # Test Entropy criterion
    print("\n2. Entropy Criterion (unpruned)")
    print("-" * 70)
    dt_entropy = DecisionTreeClassifierWrapper(criterion='entropy')
    dt_entropy.fit(X_train, y_train)
    y_pred = dt_entropy.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Tree depth: {dt_entropy.get_depth()}")

    # Test with pruning
    print("\n3. With Pruning (max_depth=5)")
    print("-" * 70)
    dt_pruned = DecisionTreeClassifierWrapper(
        criterion='gini',
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5
    )
    dt_pruned.fit(X_train, y_train)
    y_pred = dt_pruned.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Tree depth: {dt_pruned.get_depth()}")
    print(f"Number of leaves: {dt_pruned.get_n_leaves()}")

    # Feature importance
    print("\n4. Feature Importance")
    print("-" * 70)
    importance = dt_gini.get_feature_importance()
    top_features = np.argsort(importance)[::-1][:5]
    print("Top 5 features:")
    for i, feat_idx in enumerate(top_features, 1):
        print(f"  {i}. Feature {feat_idx}: {importance[feat_idx]:.4f}")

    print("\n" + "=" * 70)
    print("Decision Tree tested successfully!")
