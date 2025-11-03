"""
Boosting Ensemble Methods

This module implements various Boosting ensemble methods. Boosting builds models
sequentially, where each new model focuses on correcting the errors made by
previous models. This reduces bias and can create very powerful ensembles.

Key Features:
- AdaBoost (Adaptive Boosting)
- Gradient Boosting (GBDT)
- XGBoost (Extreme Gradient Boosting)
- LightGBM (Light Gradient Boosting Machine)
- CatBoost (Categorical Boosting)
- HistGradient Boosting
- Early stopping support
- Feature importance analysis

Author: ML Framework Team
"""

import numpy as np
from typing import Optional, Union, Dict, Any
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import warnings

# Optional imports for advanced boosting libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not available. Install with: pip install catboost")


class AdaBoostEnsemble:
    """
    AdaBoost (Adaptive Boosting) Ensemble

    AdaBoost combines multiple weak learners (typically decision stumps) into a
    strong learner. It adjusts the weights of incorrectly classified samples,
    forcing subsequent learners to focus on difficult cases.

    Parameters:
    -----------
    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
    learning_rate : float, default=1.0
        Weight applied to each classifier at each boosting iteration.
    algorithm : {'SAMME', 'SAMME.R'}, default='SAMME.R'
        SAMME.R uses predicted probabilities (requires predict_proba).
        SAMME uses predicted class labels.
    random_state : int, default=None
        Controls the random seed.

    Example:
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>>
    >>> ada = AdaBoostEnsemble(n_estimators=100, learning_rate=1.0)
    >>> ada.fit(X, y, task='classification')
    >>> predictions = ada.predict(X)
    >>> importance = ada.get_feature_importance()
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        algorithm: str = 'SAMME.R',
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state
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
        Fit the AdaBoost ensemble.

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
            self.model_ = AdaBoostClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                algorithm=self.algorithm,
                random_state=self.random_state
            )
        else:
            self.model_ = AdaBoostRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )

        self.model_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels or regression values."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.task != 'classification':
            raise ValueError("predict_proba() only available for classification")
        return self.model_.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.feature_importances_


class GradientBoostingEnsemble:
    """
    Gradient Boosting Decision Tree (GBDT) Ensemble

    Gradient Boosting builds an ensemble of trees sequentially, where each tree
    is trained to predict the residuals (errors) of the previous trees. It uses
    gradient descent in function space.

    Parameters:
    -----------
    n_estimators : int, default=100
        The number of boosting stages.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree.
    max_depth : int, default=3
        Maximum depth of individual trees.
    min_samples_split : int or float, default=2
        Minimum number of samples required to split a node.
    min_samples_leaf : int or float, default=1
        Minimum number of samples required at a leaf node.
    subsample : float, default=1.0
        Fraction of samples used for fitting individual trees.
    max_features : str, int or float, default=None
        Number of features to consider for best split.
    validation_fraction : float, default=0.1
        Fraction of training data for early stopping.
    n_iter_no_change : int, default=None
        Number of iterations with no improvement for early stopping.
    random_state : int, default=None
        Controls randomness.

    Example:
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>>
    >>> gb = GradientBoostingEnsemble(
    ...     n_estimators=100,
    ...     learning_rate=0.1,
    ...     max_depth=3,
    ...     subsample=0.8
    ... )
    >>> gb.fit(X, y, task='classification')
    >>> predictions = gb.predict(X)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        subsample: float = 1.0,
        max_features: Optional[Union[str, int, float]] = None,
        validation_fraction: float = 0.1,
        n_iter_no_change: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.model_ = None
        self.task = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = 'classification',
        sample_weight: Optional[np.ndarray] = None
    ):
        """Fit the Gradient Boosting ensemble."""
        self.task = task

        if task == 'classification':
            self.model_ = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                subsample=self.subsample,
                max_features=self.max_features,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                random_state=self.random_state
            )
        else:
            self.model_ = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                subsample=self.subsample,
                max_features=self.max_features,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                random_state=self.random_state
            )

        self.model_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels or regression values."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.task != 'classification':
            raise ValueError("predict_proba() only available for classification")
        return self.model_.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.feature_importances_


class XGBoostEnsemble:
    """
    XGBoost (Extreme Gradient Boosting) Ensemble

    XGBoost is an optimized implementation of gradient boosting with additional
    features like regularization, parallel processing, and handling missing values.

    Parameters:
    -----------
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.1
        Step size shrinkage used to prevent overfitting.
    max_depth : int, default=6
        Maximum tree depth.
    min_child_weight : float, default=1
        Minimum sum of instance weight needed in a child.
    gamma : float, default=0
        Minimum loss reduction required to make a split.
    subsample : float, default=1.0
        Subsample ratio of training instances.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    reg_alpha : float, default=0
        L1 regularization term on weights.
    reg_lambda : float, default=1
        L2 regularization term on weights.
    early_stopping_rounds : int, default=None
        Validation metric needs to improve for early stopping.
    random_state : int, default=None
        Random seed.

    Example:
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    >>>
    >>> xgb_model = XGBoostEnsemble(
    ...     n_estimators=100,
    ...     learning_rate=0.1,
    ...     max_depth=6,
    ...     early_stopping_rounds=10
    ... )
    >>> xgb_model.fit(X_train, y_train, task='classification',
    ...               eval_set=[(X_val, y_val)])
    >>> predictions = xgb_model.predict(X_val)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_child_weight: float = 1,
        gamma: float = 0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        early_stopping_rounds: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.model_ = None
        self.task = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = 'classification',
        eval_set: Optional[list] = None,
        verbose: bool = True
    ):
        """
        Fit the XGBoost ensemble.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        task : str, default='classification'
            The task type: 'classification' or 'regression'.
        eval_set : list of (X, y) tuples, default=None
            Evaluation sets for early stopping.
        verbose : bool, default=True
            Print training progress.

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        self.task = task

        if task == 'classification':
            self.model_ = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                eval_metric='logloss'
            )
        else:
            self.model_ = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                eval_metric='rmse'
            )

        # Fit with optional early stopping
        fit_params = {}
        if eval_set is not None and self.early_stopping_rounds is not None:
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = self.early_stopping_rounds
            fit_params['verbose'] = verbose

        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels or regression values."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.task != 'classification':
            raise ValueError("predict_proba() only available for classification")
        return self.model_.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.feature_importances_


class LightGBMEnsemble:
    """
    LightGBM (Light Gradient Boosting Machine) Ensemble

    LightGBM uses a leaf-wise tree growth algorithm (instead of level-wise like
    most other implementations), which can be more efficient and accurate.

    Parameters:
    -----------
    n_estimators : int, default=100
        Number of boosting iterations.
    learning_rate : float, default=0.1
        Boosting learning rate.
    max_depth : int, default=-1
        Maximum tree depth. -1 means no limit.
    num_leaves : int, default=31
        Maximum number of leaves in one tree.
    min_child_samples : int, default=20
        Minimum number of data points in a leaf.
    subsample : float, default=1.0
        Subsample ratio of training instances.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    reg_alpha : float, default=0
        L1 regularization.
    reg_lambda : float, default=0
        L2 regularization.
    early_stopping_rounds : int, default=None
        Early stopping rounds.
    random_state : int, default=None
        Random seed.

    Example:
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>>
    >>> lgbm = LightGBMEnsemble(
    ...     n_estimators=100,
    ...     learning_rate=0.1,
    ...     num_leaves=31
    ... )
    >>> lgbm.fit(X, y, task='classification')
    >>> predictions = lgbm.predict(X)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0,
        reg_lambda: float = 0,
        early_stopping_rounds: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.model_ = None
        self.task = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = 'classification',
        eval_set: Optional[list] = None,
        verbose: bool = True
    ):
        """Fit the LightGBM ensemble."""
        self.task = task

        if task == 'classification':
            self.model_ = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                verbose=-1 if not verbose else 0
            )
        else:
            self.model_ = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                verbose=-1 if not verbose else 0
            )

        # Fit with optional early stopping
        fit_params = {}
        if eval_set is not None and self.early_stopping_rounds is not None:
            fit_params['eval_set'] = eval_set
            fit_params['callbacks'] = [lgb.early_stopping(self.early_stopping_rounds)]

        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels or regression values."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.task != 'classification':
            raise ValueError("predict_proba() only available for classification")
        return self.model_.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.feature_importances_


class CatBoostEnsemble:
    """
    CatBoost (Categorical Boosting) Ensemble

    CatBoost is specifically designed to handle categorical features efficiently
    without preprocessing. It uses ordered boosting and can achieve high accuracy.

    Parameters:
    -----------
    n_estimators : int, default=100
        Number of boosting iterations.
    learning_rate : float, default=0.1
        Learning rate.
    depth : int, default=6
        Depth of the trees.
    l2_leaf_reg : float, default=3.0
        L2 regularization coefficient.
    early_stopping_rounds : int, default=None
        Early stopping rounds.
    random_state : int, default=None
        Random seed.
    verbose : bool, default=False
        Print training progress.

    Example:
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>>
    >>> cat = CatBoostEnsemble(
    ...     n_estimators=100,
    ...     learning_rate=0.1,
    ...     depth=6
    ... )
    >>> cat.fit(X, y, task='classification')
    >>> predictions = cat.predict(X)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        early_stopping_rounds: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Install with: pip install catboost")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.verbose = verbose
        self.model_ = None
        self.task = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = 'classification',
        eval_set: Optional[tuple] = None,
        cat_features: Optional[list] = None
    ):
        """
        Fit the CatBoost ensemble.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        task : str, default='classification'
            The task type: 'classification' or 'regression'.
        eval_set : tuple of (X, y), default=None
            Evaluation set for early stopping.
        cat_features : list of int or str, default=None
            Indices or names of categorical features.

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        self.task = task

        if task == 'classification':
            self.model_ = CatBoostClassifier(
                iterations=self.n_estimators,
                learning_rate=self.learning_rate,
                depth=self.depth,
                l2_leaf_reg=self.l2_leaf_reg,
                random_state=self.random_state,
                verbose=self.verbose
            )
        else:
            self.model_ = CatBoostRegressor(
                iterations=self.n_estimators,
                learning_rate=self.learning_rate,
                depth=self.depth,
                l2_leaf_reg=self.l2_leaf_reg,
                random_state=self.random_state,
                verbose=self.verbose
            )

        # Fit with optional early stopping
        fit_params = {}
        if cat_features is not None:
            fit_params['cat_features'] = cat_features
        if eval_set is not None and self.early_stopping_rounds is not None:
            fit_params['eval_set'] = [eval_set]
            fit_params['early_stopping_rounds'] = self.early_stopping_rounds

        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels or regression values."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.task != 'classification':
            raise ValueError("predict_proba() only available for classification")
        return self.model_.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.feature_importances_


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("=" * 70)
    print("BOOSTING ENSEMBLE EXAMPLES")
    print("=" * 70)

    # Generate data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Example 1: AdaBoost
    print("\n1. AdaBoost Ensemble")
    print("-" * 70)
    ada = AdaBoostEnsemble(n_estimators=100, learning_rate=1.0)
    ada.fit(X_train, y_train, task='classification')
    y_pred = ada.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Example 2: Gradient Boosting
    print("\n2. Gradient Boosting Ensemble")
    print("-" * 70)
    gb = GradientBoostingEnsemble(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8
    )
    gb.fit(X_train, y_train, task='classification')
    y_pred = gb.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Example 3: XGBoost (if available)
    if XGBOOST_AVAILABLE:
        print("\n3. XGBoost Ensemble")
        print("-" * 70)
        xgb_model = XGBoostEnsemble(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6
        )
        xgb_model.fit(X_train, y_train, task='classification', verbose=False)
        y_pred = xgb_model.predict(X_test)
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Example 4: LightGBM (if available)
    if LIGHTGBM_AVAILABLE:
        print("\n4. LightGBM Ensemble")
        print("-" * 70)
        lgbm = LightGBMEnsemble(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31
        )
        lgbm.fit(X_train, y_train, task='classification', verbose=False)
        y_pred = lgbm.predict(X_test)
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Example 5: CatBoost (if available)
    if CATBOOST_AVAILABLE:
        print("\n5. CatBoost Ensemble")
        print("-" * 70)
        cat = CatBoostEnsemble(
            n_estimators=100,
            learning_rate=0.1,
            depth=6,
            verbose=False
        )
        cat.fit(X_train, y_train, task='classification')
        y_pred = cat.predict(X_test)
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\n" + "=" * 70)
    print("All boosting examples completed!")
    print("=" * 70)
