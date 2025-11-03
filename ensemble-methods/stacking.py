"""
Stacking Ensemble Methods

This module implements Stacking (Stacked Generalization) ensemble methods.
Stacking combines multiple base models (level-0) by training a meta-model (level-1)
on their predictions. This can capture complex patterns and improve performance.

Key Features:
- StackingClassifier: Multi-layer stacking for classification
- StackingRegressor: Multi-layer stacking for regression
- Cross-validation predictions for base models
- Support for multiple meta-learners
- Multi-level stacking (deep stacking)
- Feature augmentation (original features + predictions)

Author: ML Framework Team
"""

import numpy as np
from typing import List, Optional, Union, Tuple
from sklearn.ensemble import StackingClassifier as SKStackingClassifier
from sklearn.ensemble import StackingRegressor as SKStackingRegressor
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression, Ridge
import warnings


class StackingEnsemble:
    """
    Stacking Ensemble (Stacked Generalization)

    Stacking combines multiple diverse base models by training a meta-model on
    their out-of-fold predictions. The base models are trained on the full dataset,
    and the meta-model learns to combine their predictions optimally.

    Process:
    1. Split training data into K folds
    2. For each base model:
       - Train on K-1 folds, predict on the remaining fold (repeat for all folds)
       - Train on full dataset for final predictions
    3. Train meta-model on out-of-fold predictions from all base models
    4. Final prediction: meta-model(base_model_predictions)

    Parameters:
    -----------
    estimators : list of (str, estimator) tuples
        Base estimators which will be stacked together.
    final_estimator : estimator, default=None
        The final estimator (meta-model) trained on the base estimators' predictions.
        If None, LogisticRegression (classification) or Ridge (regression) is used.
    cv : int or cross-validation generator, default=5
        Cross-validation strategy for generating out-of-fold predictions.
    stack_method : {'auto', 'predict_proba', 'decision_function', 'predict'}, default='auto'
        Method called for each base estimator:
        - 'predict_proba': use predicted probabilities (classification)
        - 'decision_function': use decision function
        - 'predict': use predictions directly
        - 'auto': automatically select based on estimator
    passthrough : bool, default=False
        When True, the original features are concatenated with the predictions
        of the base estimators.
    n_jobs : int, default=None
        Number of jobs to run in parallel. -1 means using all processors.
    verbose : int, default=0
        Verbosity level.

    Attributes:
    -----------
    estimators_ : list of estimators
        The fitted base estimators.
    final_estimator_ : estimator
        The fitted meta-model.
    stack_method_ : str
        The method used to call the base estimators.

    Example:
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import SVC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>>
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>>
    >>> # Define base models
    >>> base_models = [
    ...     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ...     ('svc', SVC(probability=True, random_state=42))
    ... ]
    >>>
    >>> # Create stacking ensemble
    >>> stacking = StackingEnsemble(
    ...     estimators=base_models,
    ...     final_estimator=LogisticRegression(),
    ...     cv=5,
    ...     stack_method='auto'
    ... )
    >>>
    >>> # Train the stacked model
    >>> stacking.fit(X, y, task='classification')
    >>>
    >>> # Make predictions
    >>> predictions = stacking.predict(X)
    >>> probabilities = stacking.predict_proba(X)
    """

    def __init__(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        final_estimator: Optional[BaseEstimator] = None,
        cv: Union[int, object] = 5,
        stack_method: str = 'auto',
        passthrough: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0
    ):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.passthrough = passthrough
        self.n_jobs = n_jobs
        self.verbose = verbose
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
        Fit the stacking ensemble.

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

        # Set default final estimator if not provided
        if self.final_estimator is None:
            if task == 'classification':
                final_estimator = LogisticRegression()
            else:
                final_estimator = Ridge()
        else:
            final_estimator = self.final_estimator

        # Create sklearn stacking ensemble
        if task == 'classification':
            self.model_ = SKStackingClassifier(
                estimators=self.estimators,
                final_estimator=final_estimator,
                cv=self.cv,
                stack_method=self.stack_method,
                passthrough=self.passthrough,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
        else:
            self.model_ = SKStackingRegressor(
                estimators=self.estimators,
                final_estimator=final_estimator,
                cv=self.cv,
                passthrough=self.passthrough,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )

        # Fit the stacking ensemble
        if sample_weight is not None:
            self.model_.fit(X, y, sample_weight=sample_weight)
        else:
            self.model_.fit(X, y)

        if self.verbose > 0:
            print(f"Stacking ensemble fitted with {len(self.estimators)} base models")

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
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.predict(X)

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
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.task != 'classification':
            raise ValueError("predict_proba() only available for classification tasks")
        return self.model_.predict_proba(X)

    def get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from all base estimators.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns:
        --------
        base_predictions : ndarray of shape (n_samples, n_base_models)
            Predictions from each base model.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        base_predictions = []
        for name, estimator in self.model_.estimators_:
            if self.task == 'classification' and hasattr(estimator, 'predict_proba'):
                pred = estimator.predict_proba(X)
            else:
                pred = estimator.predict(X)
                if len(pred.shape) == 1:
                    pred = pred.reshape(-1, 1)
            base_predictions.append(pred)

        return np.hstack(base_predictions)


class MultiLevelStacking:
    """
    Multi-Level Stacking Ensemble (Deep Stacking)

    This class implements multi-level stacking where multiple layers of models
    are stacked on top of each other. Each level uses predictions from the
    previous level as features.

    Process:
    1. Level 0: Train multiple diverse base models
    2. Level 1: Train models on Level 0 predictions
    3. Level 2+: Continue stacking as needed
    4. Final level: Meta-model combines all predictions

    Parameters:
    -----------
    levels : list of lists of (str, estimator) tuples
        Each list contains estimators for one level.
        Example: [
            [('rf', RandomForest()), ('svm', SVC())],  # Level 0
            [('gb', GradientBoosting())],              # Level 1
        ]
    final_estimator : estimator, default=None
        The final meta-model.
    cv : int, default=5
        Number of cross-validation folds.
    passthrough : bool, default=False
        Whether to pass original features through all levels.
    verbose : int, default=0
        Verbosity level.

    Example:
    --------
    >>> from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    >>> from sklearn.svm import SVC
    >>> from sklearn.linear_model import LogisticRegression
    >>>
    >>> # Define multi-level architecture
    >>> levels = [
    ...     # Level 0: Diverse base models
    ...     [
    ...         ('rf', RandomForestClassifier(n_estimators=10)),
    ...         ('svm', SVC(probability=True))
    ...     ],
    ...     # Level 1: Second layer models
    ...     [
    ...         ('gb', GradientBoostingClassifier(n_estimators=10))
    ...     ]
    ... ]
    >>>
    >>> # Create multi-level stacking
    >>> multi_stack = MultiLevelStacking(
    ...     levels=levels,
    ...     final_estimator=LogisticRegression(),
    ...     cv=5
    ... )
    >>>
    >>> # Train
    >>> multi_stack.fit(X, y, task='classification')
    >>> predictions = multi_stack.predict(X)
    """

    def __init__(
        self,
        levels: List[List[Tuple[str, BaseEstimator]]],
        final_estimator: Optional[BaseEstimator] = None,
        cv: int = 5,
        passthrough: bool = False,
        verbose: int = 0
    ):
        self.levels = levels
        self.final_estimator = final_estimator
        self.cv = cv
        self.passthrough = passthrough
        self.verbose = verbose
        self.level_models_ = []
        self.final_model_ = None
        self.task = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = 'classification'
    ):
        """
        Fit the multi-level stacking ensemble.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        task : str, default='classification'
            The task type: 'classification' or 'regression'.

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        self.task = task
        self.level_models_ = []

        X_current = X.copy()

        # Train each level
        for level_idx, level_estimators in enumerate(self.levels):
            if self.verbose > 0:
                print(f"\nTraining Level {level_idx}...")
                print(f"Number of models: {len(level_estimators)}")
                print(f"Input shape: {X_current.shape}")

            # Create stacking ensemble for this level
            if task == 'classification':
                final_est = LogisticRegression() if self.final_estimator is None else clone(self.final_estimator)
            else:
                final_est = Ridge() if self.final_estimator is None else clone(self.final_estimator)

            level_model = StackingEnsemble(
                estimators=level_estimators,
                final_estimator=final_est,
                cv=self.cv,
                passthrough=self.passthrough,
                verbose=self.verbose
            )

            level_model.fit(X_current, y, task=task)
            self.level_models_.append(level_model)

            # Get predictions for next level
            if task == 'classification':
                X_level_preds = level_model.get_base_predictions(X_current)
            else:
                X_level_preds = level_model.get_base_predictions(X_current)

            if self.passthrough:
                X_current = np.hstack([X_current, X_level_preds])
            else:
                X_current = X_level_preds

        # Train final meta-model
        if self.verbose > 0:
            print(f"\nTraining Final Meta-Model...")
            print(f"Input shape: {X_current.shape}")

        if self.final_estimator is None:
            if task == 'classification':
                self.final_model_ = LogisticRegression()
            else:
                self.final_model_ = Ridge()
        else:
            self.final_model_ = clone(self.final_estimator)

        self.final_model_.fit(X_current, y)

        if self.verbose > 0:
            print(f"\nMulti-level stacking complete!")
            print(f"Total levels: {len(self.levels)}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the multi-level stacking ensemble.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns:
        --------
        predictions : ndarray of shape (n_samples,)
            Predicted values.
        """
        if not self.level_models_:
            raise ValueError("Model not fitted. Call fit() first.")

        X_current = X.copy()

        # Pass through each level
        for level_model in self.level_models_:
            X_level_preds = level_model.get_base_predictions(X_current)

            if self.passthrough:
                X_current = np.hstack([X_current, X_level_preds])
            else:
                X_current = X_level_preds

        # Final prediction
        return self.final_model_.predict(X_current)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns:
        --------
        probabilities : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        if self.task != 'classification':
            raise ValueError("predict_proba() only available for classification tasks")

        if not self.level_models_:
            raise ValueError("Model not fitted. Call fit() first.")

        X_current = X.copy()

        # Pass through each level
        for level_model in self.level_models_:
            X_level_preds = level_model.get_base_predictions(X_current)

            if self.passthrough:
                X_current = np.hstack([X_current, X_level_preds])
            else:
                X_current = X_level_preds

        # Final prediction
        if hasattr(self.final_model_, 'predict_proba'):
            return self.final_model_.predict_proba(X_current)
        else:
            raise ValueError("Final estimator does not support predict_proba()")


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    print("=" * 70)
    print("STACKING ENSEMBLE EXAMPLES")
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

    # Example 1: Basic Stacking
    print("\n1. Basic Stacking Ensemble")
    print("-" * 70)

    base_models = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]

    stacking = StackingEnsemble(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=5,
        stack_method='auto',
        verbose=1
    )

    stacking.fit(X_train, y_train, task='classification')
    y_pred = stacking.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Example 2: Stacking with Passthrough
    print("\n2. Stacking with Passthrough (original features + predictions)")
    print("-" * 70)

    stacking_passthrough = StackingEnsemble(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=5,
        passthrough=True,  # Include original features
        verbose=1
    )

    stacking_passthrough.fit(X_train, y_train, task='classification')
    y_pred = stacking_passthrough.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Example 3: Multi-Level Stacking
    print("\n3. Multi-Level Stacking (Deep Stacking)")
    print("-" * 70)

    levels = [
        # Level 0: Diverse base models
        [
            ('rf', RandomForestClassifier(n_estimators=30, random_state=42)),
            ('svc', SVC(probability=True, random_state=42))
        ],
        # Level 1: Second layer
        [
            ('gb', GradientBoostingClassifier(n_estimators=30, random_state=42))
        ]
    ]

    multi_stack = MultiLevelStacking(
        levels=levels,
        final_estimator=LogisticRegression(),
        cv=5,
        verbose=1
    )

    multi_stack.fit(X_train, y_train, task='classification')
    y_pred = multi_stack.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Get base model predictions
    print("\n4. Analyzing Base Model Predictions")
    print("-" * 70)
    base_preds = stacking.get_base_predictions(X_test[:5])
    print(f"Base predictions shape: {base_preds.shape}")
    print(f"First 5 samples base predictions:\n{base_preds}")

    print("\n" + "=" * 70)
    print("All stacking examples completed!")
    print("=" * 70)
