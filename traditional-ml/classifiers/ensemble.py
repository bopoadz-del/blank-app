"""
Ensemble Classification Algorithms
Random Forest, XGBoost, LightGBM, Gradient Boosting
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import Optional, Union, Dict, Any
import logging

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestClassifierWrapper:
    """
    Random Forest Classifier

    Ensemble of decision trees trained on random subsets of data and features.
    Reduces overfitting compared to single decision trees.

    Best for:
    - High-dimensional data
    - Feature importance analysis
    - Balanced datasets
    - When interpretability is needed
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = 'sqrt',
        bootstrap: bool = True,
        n_jobs: int = -1,
        random_state: int = 42,
        class_weight: Optional[Union[str, dict]] = None,
        verbose: int = 0
    ):
        """
        Initialize Random Forest Classifier

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap samples
            n_jobs: Number of parallel jobs (-1 = use all cores)
            random_state: Random seed
            class_weight: Weights for classes ('balanced' or dict)
            verbose: Verbosity level
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.class_weight = class_weight
        self.verbose = verbose

        # Create model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight=class_weight,
            verbose=verbose
        )

        self.feature_importances_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        logger.info(f"Training Random Forest with {self.n_estimators} trees...")
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        self.classes_ = self.model.classes_
        logger.info("Training complete")
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

    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters"""
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """Set model parameters"""
        self.model.set_params(**params)
        return self


class XGBoostClassifierWrapper:
    """
    XGBoost Classifier

    Extreme Gradient Boosting with advanced regularization.
    State-of-the-art performance on structured data.

    Best for:
    - Structured/tabular data
    - Competitions (Kaggle)
    - When accuracy is priority
    - Imbalanced datasets (with scale_pos_weight)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        n_jobs: int = -1,
        random_state: int = 42,
        early_stopping_rounds: Optional[int] = None,
        eval_metric: str = 'logloss'
    ):
        """
        Initialize XGBoost Classifier

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            reg_alpha: L1 regularization (lasso)
            reg_lambda: L2 regularization (ridge)
            min_child_weight: Minimum sum of instance weight in a child
            gamma: Minimum loss reduction for split
            n_jobs: Number of parallel threads
            random_state: Random seed
            early_stopping_rounds: Early stopping rounds
            eval_metric: Evaluation metric
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

        # Create model
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            n_jobs=n_jobs,
            random_state=random_state,
            eval_metric=eval_metric,
            use_label_encoder=False
        )

        self.feature_importances_ = None
        self.classes_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        verbose: bool = False
    ):
        """Train the model"""
        logger.info(f"Training XGBoost with {self.n_estimators} rounds...")

        fit_params = {'verbose': verbose}

        if eval_set is not None and self.early_stopping_rounds is not None:
            fit_params['early_stopping_rounds'] = self.early_stopping_rounds
            fit_params['eval_set'] = eval_set

        self.model.fit(X, y, **fit_params)
        self.feature_importances_ = self.model.feature_importances_
        self.classes_ = self.model.classes_
        logger.info("Training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        return self.model.predict_proba(X)

    def get_feature_importance(self, importance_type: str = 'weight') -> np.ndarray:
        """
        Get feature importances

        Args:
            importance_type: 'weight', 'gain', 'cover', or 'total_gain'
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not fitted yet")

        if importance_type != 'weight':
            booster = self.model.get_booster()
            importance_dict = booster.get_score(importance_type=importance_type)
            # Convert to array
            n_features = len(self.feature_importances_)
            importances = np.zeros(n_features)
            for i in range(n_features):
                key = f'f{i}'
                if key in importance_dict:
                    importances[i] = importance_dict[key]
            return importances

        return self.feature_importances_

    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters"""
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """Set model parameters"""
        self.model.set_params(**params)
        return self


class LightGBMClassifierWrapper:
    """
    LightGBM Classifier

    Fast gradient boosting with histogram-based learning.
    Very efficient for large datasets.

    Best for:
    - Large datasets (>10k samples)
    - When training speed matters
    - Categorical features (native support)
    - Memory-constrained environments
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        min_child_samples: int = 20,
        n_jobs: int = -1,
        random_state: int = 42,
        early_stopping_rounds: Optional[int] = None,
        categorical_feature: Optional[list] = None
    ):
        """
        Initialize LightGBM Classifier

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (-1 = no limit)
            learning_rate: Boosting learning rate
            num_leaves: Max number of leaves in one tree
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            min_child_samples: Minimum number of data in one leaf
            n_jobs: Number of parallel threads
            random_state: Random seed
            early_stopping_rounds: Early stopping rounds
            categorical_feature: List of categorical feature indices
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.categorical_feature = categorical_feature

        # Create model
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_samples=min_child_samples,
            n_jobs=n_jobs,
            random_state=random_state
        )

        self.feature_importances_ = None
        self.classes_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        verbose: bool = False
    ):
        """Train the model"""
        logger.info(f"Training LightGBM with {self.n_estimators} rounds...")

        fit_params = {'verbose': verbose}

        if self.categorical_feature is not None:
            fit_params['categorical_feature'] = self.categorical_feature

        if eval_set is not None and self.early_stopping_rounds is not None:
            fit_params['early_stopping_rounds'] = self.early_stopping_rounds
            fit_params['eval_set'] = eval_set

        self.model.fit(X, y, **fit_params)
        self.feature_importances_ = self.model.feature_importances_
        self.classes_ = self.model.classes_
        logger.info("Training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        return self.model.predict_proba(X)

    def get_feature_importance(self, importance_type: str = 'split') -> np.ndarray:
        """
        Get feature importances

        Args:
            importance_type: 'split' or 'gain'
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not fitted yet")

        if importance_type == 'gain':
            booster = self.model.booster_
            return booster.feature_importance(importance_type='gain')

        return self.feature_importances_

    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters"""
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """Set model parameters"""
        self.model.set_params(**params)
        return self


class GradientBoostingClassifierWrapper:
    """
    Gradient Boosting Classifier (Scikit-learn)

    Sequential ensemble of weak learners (usually decision trees).
    Good baseline before trying XGBoost/LightGBM.

    Best for:
    - Small to medium datasets
    - When you want pure sklearn implementation
    - Baseline comparisons
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        max_features: Optional[Union[str, int, float]] = None,
        random_state: int = 42,
        verbose: int = 0
    ):
        """Initialize Gradient Boosting Classifier"""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state
        self.verbose = verbose

        # Create model
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose
        )

        self.feature_importances_ = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        logger.info(f"Training Gradient Boosting with {self.n_estimators} rounds...")
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        self.classes_ = self.model.classes_
        logger.info("Training complete")
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
    from sklearn.metrics import accuracy_score, classification_report

    # Create dataset
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

    print("=" * 70)
    print("Ensemble Classifiers Comparison")
    print("=" * 70)

    # Test Random Forest
    print("\n1. Random Forest")
    print("-" * 70)
    rf = RandomForestClassifierWrapper(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Test XGBoost
    if XGBOOST_AVAILABLE:
        print("\n2. XGBoost")
        print("-" * 70)
        xgb_clf = XGBoostClassifierWrapper(n_estimators=100)
        xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = xgb_clf.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Test LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\n3. LightGBM")
        print("-" * 70)
        lgb_clf = LightGBMClassifierWrapper(n_estimators=100)
        lgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = lgb_clf.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Test Gradient Boosting
    print("\n4. Gradient Boosting")
    print("-" * 70)
    gb = GradientBoostingClassifierWrapper(n_estimators=100)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\n" + "=" * 70)
    print("All ensemble classifiers tested successfully!")
