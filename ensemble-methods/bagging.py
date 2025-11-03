"""
Bagging Ensemble Methods

This module implements Bootstrap Aggregating (Bagging) ensemble methods for both
classification and regression tasks. Bagging reduces variance by training multiple
models on random subsets of the training data with replacement.

Key Features:
- BaggingClassifier: Classification with bootstrap sampling
- BaggingRegressor: Regression with bootstrap sampling
- RandomForestEnsemble: Optimized Random Forest implementation
- ExtraTreesEnsemble: Extremely Randomized Trees
- Parallel training support
- Out-of-bag (OOB) score estimation
- Feature importance aggregation

Author: ML Framework Team
"""

import numpy as np
from typing import List, Optional, Union, Callable, Any
from sklearn.ensemble import (
    BaggingClassifier as SKBaggingClassifier,
    BaggingRegressor as SKBaggingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import BaseEstimator, clone
import warnings


class BaggingEnsemble:
    """
    Bootstrap Aggregating (Bagging) Ensemble

    Bagging trains multiple base estimators on random subsets of the original dataset
    (with replacement) and aggregates their predictions. This reduces variance and
    helps prevent overfitting.

    Parameters:
    -----------
    base_estimator : estimator object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a Decision Tree.
    n_estimators : int, default=10
        The number of base estimators in the ensemble.
    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator.
        - If int, draw max_samples samples.
        - If float, draw max_samples * X.shape[0] samples.
    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.
        - If int, draw max_features features.
        - If float, draw max_features * X.shape[1] features.
    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.
    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization error.
    n_jobs : int, default=None
        The number of jobs to run in parallel for both fit and predict.
        None means 1, -1 means using all processors.
    random_state : int, default=None
        Controls the random resampling of the original dataset.
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Attributes:
    -----------
    estimators_ : list of estimators
        The collection of fitted base estimators.
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
    oob_prediction_ : ndarray of shape (n_samples,)
        Prediction computed with out-of-bag estimate on the training set.

    Example:
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    >>>
    >>> # Create bagging ensemble
    >>> bagging = BaggingEnsemble(
    ...     base_estimator=DecisionTreeClassifier(max_depth=5),
    ...     n_estimators=50,
    ...     max_samples=0.8,
    ...     max_features=0.8,
    ...     bootstrap=True,
    ...     oob_score=True,
    ...     n_jobs=-1,
    ...     random_state=42
    ... )
    >>>
    >>> # Train the ensemble
    >>> bagging.fit(X, y, task='classification')
    >>>
    >>> # Make predictions
    >>> predictions = bagging.predict(X)
    >>> probabilities = bagging.predict_proba(X)
    >>>
    >>> # Check OOB score
    >>> print(f"OOB Score: {bagging.get_oob_score():.4f}")
    """

    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        n_estimators: int = 10,
        max_samples: Union[int, float] = 1.0,
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.ensemble_ = None
        self.task = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = 'classification',
        sample_weight: Optional[np.ndarray] = None
    ):
        """
        Fit the bagging ensemble.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        task : str, default='classification'
            The task type: 'classification' or 'regression'.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        self.task = task

        # Select base estimator if not provided
        if self.base_estimator is None:
            if task == 'classification':
                base_estimator = DecisionTreeClassifier()
            else:
                base_estimator = DecisionTreeRegressor()
        else:
            base_estimator = self.base_estimator

        # Create sklearn bagging ensemble
        if task == 'classification':
            self.ensemble_ = SKBaggingClassifier(
                estimator=base_estimator,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose
            )
        else:
            self.ensemble_ = SKBaggingRegressor(
                estimator=base_estimator,
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose
            )

        # Fit the ensemble
        self.ensemble_.fit(X, y, sample_weight=sample_weight)

        if self.verbose > 0:
            print(f"Bagging ensemble fitted with {self.n_estimators} estimators")
            if self.oob_score:
                print(f"OOB Score: {self.ensemble_.oob_score_:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels or regression values for X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns:
        --------
        predictions : ndarray of shape (n_samples,)
            Predicted class labels or regression values.
        """
        if self.ensemble_ is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        return self.ensemble_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X (classification only).

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns:
        --------
        probabilities : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        if self.ensemble_ is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        if self.task != 'classification':
            raise ValueError("predict_proba() only available for classification tasks")

        return self.ensemble_.predict_proba(X)

    def get_oob_score(self) -> float:
        """
        Get the out-of-bag score.

        Returns:
        --------
        oob_score : float
            OOB score of the training dataset.
        """
        if not self.oob_score:
            raise ValueError("OOB score not available. Set oob_score=True when creating ensemble.")

        if self.ensemble_ is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        return self.ensemble_.oob_score_

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get aggregated feature importance from all base estimators.

        Returns:
        --------
        feature_importance : ndarray or None
            Feature importance values, or None if base estimators don't support it.
        """
        if self.ensemble_ is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Check if base estimators have feature_importances_
        if hasattr(self.ensemble_.estimators_[0], 'feature_importances_'):
            importances = np.array([
                estimator.feature_importances_
                for estimator in self.ensemble_.estimators_
            ])
            return np.mean(importances, axis=0)
        else:
            warnings.warn("Base estimators do not support feature importance")
            return None


class RandomForestEnsemble:
    """
    Random Forest Ensemble

    Random Forest is an extension of bagging that also randomly selects a subset
    of features at each split in the decision tree. This creates more diverse trees
    and often improves performance.

    Parameters:
    -----------
    n_estimators : int, default=100
        The number of trees in the forest.
    criterion : {"gini", "entropy", "log_loss"} for classification or
                {"squared_error", "absolute_error", "friedman_mse", "poisson"} for regression
        The function to measure the quality of a split.
    max_depth : int, default=None
        The maximum depth of the tree. If None, nodes are expanded until
        all leaves are pure or contain less than min_samples_split samples.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
    max_features : {"sqrt", "log2", None} or int or float, default="sqrt"
        The number of features to consider when looking for the best split.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization error.
    n_jobs : int, default=None
        The number of jobs to run in parallel. -1 means using all processors.
    random_state : int, default=None
        Controls randomness of the estimator.
    verbose : int, default=0
        Controls verbosity of the building process.
    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes (classification only).

    Attributes:
    -----------
    estimators_ : list of DecisionTreeClassifier or DecisionTreeRegressor
        The collection of fitted sub-estimators.
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    Example:
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    >>>
    >>> # Create Random Forest ensemble
    >>> rf = RandomForestEnsemble(
    ...     n_estimators=100,
    ...     max_depth=10,
    ...     max_features='sqrt',
    ...     oob_score=True,
    ...     n_jobs=-1,
    ...     random_state=42
    ... )
    >>>
    >>> # Train the forest
    >>> rf.fit(X, y, task='classification')
    >>>
    >>> # Make predictions
    >>> predictions = rf.predict(X)
    >>>
    >>> # Get feature importance
    >>> importance = rf.get_feature_importance()
    >>> print(f"Top feature: {np.argmax(importance)}")
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = 'gini',
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Union[str, int, float] = 'sqrt',
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
        class_weight: Optional[Union[dict, str]] = None
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight
        self.model_ = None
        self.task = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = 'classification',
        sample_weight: Optional[np.ndarray] = None
    ):
        """
        Fit the Random Forest ensemble.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        task : str, default='classification'
            The task type: 'classification' or 'regression'.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        self.task = task

        if task == 'classification':
            self.model_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                class_weight=self.class_weight
            )
        else:
            self.model_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose
            )

        self.model_.fit(X, y, sample_weight=sample_weight)

        if self.verbose > 0:
            print(f"Random Forest fitted with {self.n_estimators} trees")
            if self.oob_score:
                print(f"OOB Score: {self.model_.oob_score_:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels or regression values for X."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X (classification only)."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.task != 'classification':
            raise ValueError("predict_proba() only available for classification tasks")
        return self.model_.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from the forest."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.feature_importances_

    def get_oob_score(self) -> float:
        """Get the out-of-bag score."""
        if not self.oob_score:
            raise ValueError("OOB score not available. Set oob_score=True when creating ensemble.")
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.oob_score_


class ExtraTreesEnsemble:
    """
    Extremely Randomized Trees (Extra Trees) Ensemble

    Extra Trees is similar to Random Forest but uses random thresholds for each
    feature rather than searching for the best threshold. This adds more randomness
    and can sometimes improve performance while reducing computation time.

    Parameters are similar to RandomForestEnsemble.

    Example:
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    >>>
    >>> # Create Extra Trees ensemble
    >>> et = ExtraTreesEnsemble(
    ...     n_estimators=100,
    ...     max_depth=10,
    ...     n_jobs=-1,
    ...     random_state=42
    ... )
    >>>
    >>> # Train the ensemble
    >>> et.fit(X, y, task='classification')
    >>>
    >>> # Make predictions
    >>> predictions = et.predict(X)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = 'gini',
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Union[str, int, float] = 'sqrt',
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
        class_weight: Optional[Union[dict, str]] = None
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight
        self.model_ = None
        self.task = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = 'classification',
        sample_weight: Optional[np.ndarray] = None
    ):
        """Fit the Extra Trees ensemble."""
        self.task = task

        if task == 'classification':
            self.model_ = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                class_weight=self.class_weight
            )
        else:
            self.model_ = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose
            )

        self.model_.fit(X, y, sample_weight=sample_weight)

        if self.verbose > 0:
            print(f"Extra Trees fitted with {self.n_estimators} trees")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels or regression values for X."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X (classification only)."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.task != 'classification':
            raise ValueError("predict_proba() only available for classification tasks")
        return self.model_.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from the ensemble."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.feature_importances_


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("=" * 70)
    print("BAGGING ENSEMBLE EXAMPLE")
    print("=" * 70)

    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Example 1: Basic Bagging
    print("\n1. Basic Bagging Ensemble")
    print("-" * 70)
    bagging = BaggingEnsemble(
        n_estimators=50,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    bagging.fit(X_train, y_train, task='classification')
    y_pred = bagging.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"OOB Score: {bagging.get_oob_score():.4f}")

    # Example 2: Random Forest
    print("\n2. Random Forest Ensemble")
    print("-" * 70)
    rf = RandomForestEnsemble(
        n_estimators=100,
        max_depth=10,
        max_features='sqrt',
        oob_score=True,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    rf.fit(X_train, y_train, task='classification')
    y_pred_rf = rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Test Accuracy: {accuracy_rf:.4f}")
    print(f"OOB Score: {rf.get_oob_score():.4f}")

    # Get feature importance
    importance = rf.get_feature_importance()
    print(f"\nTop 5 important features: {np.argsort(importance)[-5:][::-1]}")

    # Example 3: Extra Trees
    print("\n3. Extra Trees Ensemble")
    print("-" * 70)
    et = ExtraTreesEnsemble(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    et.fit(X_train, y_train, task='classification')
    y_pred_et = et.predict(X_test)
    accuracy_et = accuracy_score(y_test, y_pred_et)
    print(f"Test Accuracy: {accuracy_et:.4f}")

    print("\n" + "=" * 70)
    print("All bagging examples completed successfully!")
    print("=" * 70)
