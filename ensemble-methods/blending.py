"""
Blending Ensemble Methods

This module implements Blending ensemble methods. Blending is similar to stacking
but uses a holdout validation set instead of cross-validation to generate
meta-features. This is simpler and faster than stacking but uses less training data.

Key Features:
- BlendingEnsemble: Basic blending with holdout validation
- MultiLayerBlending: Deep blending with multiple layers
- Automatic blend weight optimization
- Support for both classification and regression
- Feature augmentation (original features + predictions)

Author: ML Framework Team
"""

import numpy as np
from typing import List, Optional, Union, Tuple
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error


class BlendingEnsemble:
    """
    Blending Ensemble

    Blending is an ensemble technique similar to stacking but simpler:

    1. Split training data into train and holdout (blend) sets
    2. Train base models on the train set
    3. Make predictions on the holdout set using base models
    4. Train meta-model on holdout predictions
    5. For final predictions: meta-model(base_model_predictions on test data)

    Differences from Stacking:
    - Stacking uses cross-validation (all data used for training)
    - Blending uses a single holdout set (simpler, faster, but less data for training)
    - Blending is easier to implement and understand
    - Stacking typically performs better but takes longer

    Parameters:
    -----------
    estimators : list of (str, estimator) tuples
        Base estimators to blend.
    meta_estimator : estimator, default=None
        Meta-learner trained on base predictions. If None, LogisticRegression
        (classification) or Ridge (regression) is used.
    test_size : float, default=0.2
        Proportion of data to use for blending (holdout set).
    passthrough : bool, default=False
        If True, original features are concatenated with base predictions.
    random_state : int, default=None
        Random seed for train/blend split.
    verbose : int, default=0
        Verbosity level.

    Attributes:
    -----------
    base_estimators_ : list of fitted estimators
        Fitted base models.
    meta_estimator_ : fitted estimator
        Fitted meta-model.

    Example:
    --------
    >>> from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    >>> from sklearn.svm import SVC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>>
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>>
    >>> # Define base models
    >>> base_models = [
    ...     ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ...     ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ...     ('svc', SVC(probability=True, random_state=42))
    ... ]
    >>>
    >>> # Create blending ensemble
    >>> blending = BlendingEnsemble(
    ...     estimators=base_models,
    ...     meta_estimator=LogisticRegression(),
    ...     test_size=0.2,
    ...     random_state=42
    ... )
    >>>
    >>> # Fit (automatically splits into train/blend sets)
    >>> blending.fit(X, y, task='classification')
    >>>
    >>> # Predict
    >>> predictions = blending.predict(X)
    >>> probabilities = blending.predict_proba(X)
    """

    def __init__(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        meta_estimator: Optional[BaseEstimator] = None,
        test_size: float = 0.2,
        passthrough: bool = False,
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        self.estimators = estimators
        self.meta_estimator = meta_estimator
        self.test_size = test_size
        self.passthrough = passthrough
        self.random_state = random_state
        self.verbose = verbose
        self.base_estimators_ = []
        self.meta_estimator_ = None
        self.task = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = 'classification'
    ):
        """
        Fit the blending ensemble.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        task : str, default='classification'
            Task type: 'classification' or 'regression'.

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        self.task = task

        # Split data into train and blend sets
        X_train, X_blend, y_train, y_blend = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if task == 'classification' else None
        )

        if self.verbose > 0:
            print(f"Training data shape: {X_train.shape}")
            print(f"Blending data shape: {X_blend.shape}")
            print(f"\nTraining {len(self.estimators)} base models...")

        # Train base models and collect blend predictions
        blend_predictions = []

        for i, (name, estimator) in enumerate(self.estimators):
            if self.verbose > 0:
                print(f"  [{i+1}/{len(self.estimators)}] Training {name}...")

            # Clone and fit on train set
            est = clone(estimator)
            est.fit(X_train, y_train)
            self.base_estimators_.append((name, est))

            # Predict on blend set
            if task == 'classification' and hasattr(est, 'predict_proba'):
                blend_pred = est.predict_proba(X_blend)
            else:
                blend_pred = est.predict(X_blend).reshape(-1, 1)

            blend_predictions.append(blend_pred)

        # Stack blend predictions
        X_blend_meta = np.hstack(blend_predictions)

        if self.verbose > 0:
            print(f"\nMeta-features shape: {X_blend_meta.shape}")

        # Add original features if passthrough
        if self.passthrough:
            X_blend_meta = np.hstack([X_blend, X_blend_meta])
            if self.verbose > 0:
                print(f"With passthrough: {X_blend_meta.shape}")

        # Train meta-model
        if self.meta_estimator is None:
            if task == 'classification':
                self.meta_estimator_ = LogisticRegression()
            else:
                self.meta_estimator_ = Ridge()
        else:
            self.meta_estimator_ = clone(self.meta_estimator)

        if self.verbose > 0:
            print(f"\nTraining meta-model: {type(self.meta_estimator_).__name__}")

        self.meta_estimator_.fit(X_blend_meta, y_blend)

        if self.verbose > 0:
            # Evaluate on blend set
            y_pred = self.meta_estimator_.predict(X_blend_meta)
            if task == 'classification':
                score = accuracy_score(y_blend, y_pred)
                print(f"Meta-model blend accuracy: {score:.4f}")
            else:
                score = mean_squared_error(y_blend, y_pred, squared=False)
                print(f"Meta-model blend RMSE: {score:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the blending ensemble.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns:
        --------
        predictions : ndarray of shape (n_samples,)
            Predicted values.
        """
        if not self.base_estimators_:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get base model predictions
        base_predictions = []
        for name, estimator in self.base_estimators_:
            if self.task == 'classification' and hasattr(estimator, 'predict_proba'):
                pred = estimator.predict_proba(X)
            else:
                pred = estimator.predict(X).reshape(-1, 1)
            base_predictions.append(pred)

        X_meta = np.hstack(base_predictions)

        # Add original features if passthrough
        if self.passthrough:
            X_meta = np.hstack([X, X_meta])

        # Meta-model prediction
        return self.meta_estimator_.predict(X_meta)

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
            Predicted class probabilities.
        """
        if self.task != 'classification':
            raise ValueError("predict_proba() only available for classification")

        if not hasattr(self.meta_estimator_, 'predict_proba'):
            raise ValueError("Meta-estimator does not support predict_proba()")

        # Get base model predictions
        base_predictions = []
        for name, estimator in self.base_estimators_:
            if hasattr(estimator, 'predict_proba'):
                pred = estimator.predict_proba(X)
            else:
                pred = estimator.predict(X).reshape(-1, 1)
            base_predictions.append(pred)

        X_meta = np.hstack(base_predictions)

        # Add original features if passthrough
        if self.passthrough:
            X_meta = np.hstack([X, X_meta])

        # Meta-model probability prediction
        return self.meta_estimator_.predict_proba(X_meta)

    def get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from all base models.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns:
        --------
        base_predictions : ndarray
            Stacked predictions from base models.
        """
        if not self.base_estimators_:
            raise ValueError("Model not fitted. Call fit() first.")

        base_predictions = []
        for name, estimator in self.base_estimators_:
            if self.task == 'classification' and hasattr(estimator, 'predict_proba'):
                pred = estimator.predict_proba(X)
            else:
                pred = estimator.predict(X).reshape(-1, 1)
            base_predictions.append(pred)

        return np.hstack(base_predictions)


class MultiLayerBlending:
    """
    Multi-Layer Blending Ensemble

    Extends blending to multiple layers, similar to deep stacking but using
    holdout sets instead of cross-validation.

    Parameters:
    -----------
    layers : list of lists of (str, estimator) tuples
        Each inner list contains estimators for one layer.
    meta_estimator : estimator, default=None
        Final meta-learner.
    test_size : float, default=0.2
        Proportion for blend set at each layer.
    passthrough : bool, default=False
        Pass original features through all layers.
    random_state : int, default=None
        Random seed.
    verbose : int, default=0
        Verbosity level.

    Example:
    --------
    >>> from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    >>> from sklearn.svm import SVC
    >>> from sklearn.linear_model import LogisticRegression
    >>>
    >>> # Define layers
    >>> layers = [
    ...     # Layer 0: Base models
    ...     [
    ...         ('rf', RandomForestClassifier(n_estimators=30)),
    ...         ('svc', SVC(probability=True))
    ...     ],
    ...     # Layer 1: Second layer
    ...     [
    ...         ('gb', GradientBoostingClassifier(n_estimators=30))
    ...     ]
    ... ]
    >>>
    >>> # Create multi-layer blending
    >>> multi_blend = MultiLayerBlending(
    ...     layers=layers,
    ...     meta_estimator=LogisticRegression(),
    ...     test_size=0.2
    ... )
    >>>
    >>> multi_blend.fit(X, y, task='classification')
    >>> predictions = multi_blend.predict(X)
    """

    def __init__(
        self,
        layers: List[List[Tuple[str, BaseEstimator]]],
        meta_estimator: Optional[BaseEstimator] = None,
        test_size: float = 0.2,
        passthrough: bool = False,
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        self.layers = layers
        self.meta_estimator = meta_estimator
        self.test_size = test_size
        self.passthrough = passthrough
        self.random_state = random_state
        self.verbose = verbose
        self.layer_models_ = []
        self.meta_estimator_ = None
        self.task = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = 'classification'
    ):
        """
        Fit the multi-layer blending ensemble.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        task : str, default='classification'
            Task type.

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        self.task = task
        self.layer_models_ = []

        X_current = X.copy()
        y_current = y.copy()

        # Train each layer
        for layer_idx, layer_estimators in enumerate(self.layers):
            if self.verbose > 0:
                print(f"\n{'='*70}")
                print(f"Training Layer {layer_idx}")
                print(f"{'='*70}")

            # Use blending for this layer
            layer_blending = BlendingEnsemble(
                estimators=layer_estimators,
                meta_estimator=None,  # Will be set later
                test_size=self.test_size,
                passthrough=False,  # Handle passthrough manually
                random_state=self.random_state,
                verbose=self.verbose
            )

            # Split for this layer
            X_train, X_blend, y_train, y_blend = train_test_split(
                X_current, y_current,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_current if task == 'classification' else None
            )

            # Train base models
            layer_base_models = []
            blend_predictions = []

            for name, estimator in layer_estimators:
                if self.verbose > 0:
                    print(f"  Training {name}...")

                est = clone(estimator)
                est.fit(X_train, y_train)
                layer_base_models.append((name, est))

                # Predict on blend set
                if task == 'classification' and hasattr(est, 'predict_proba'):
                    blend_pred = est.predict_proba(X_blend)
                else:
                    blend_pred = est.predict(X_blend).reshape(-1, 1)

                blend_predictions.append(blend_pred)

            # Store layer models
            self.layer_models_.append(layer_base_models)

            # Prepare data for next layer
            X_blend_next = np.hstack(blend_predictions)

            if self.passthrough:
                X_blend_next = np.hstack([X_blend, X_blend_next])

            # Update for next layer
            X_current = X_blend_next
            y_current = y_blend

        # Train final meta-model
        if self.verbose > 0:
            print(f"\n{'='*70}")
            print("Training Final Meta-Model")
            print(f"{'='*70}")
            print(f"Meta-features shape: {X_current.shape}")

        if self.meta_estimator is None:
            if task == 'classification':
                self.meta_estimator_ = LogisticRegression()
            else:
                self.meta_estimator_ = Ridge()
        else:
            self.meta_estimator_ = clone(self.meta_estimator)

        self.meta_estimator_.fit(X_current, y_current)

        if self.verbose > 0:
            y_pred = self.meta_estimator_.predict(X_current)
            if task == 'classification':
                score = accuracy_score(y_current, y_pred)
                print(f"Final meta-model accuracy: {score:.4f}")
            else:
                score = mean_squared_error(y_current, y_pred, squared=False)
                print(f"Final meta-model RMSE: {score:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the multi-layer blending ensemble.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns:
        --------
        predictions : ndarray of shape (n_samples,)
            Predicted values.
        """
        if not self.layer_models_:
            raise ValueError("Model not fitted. Call fit() first.")

        X_current = X.copy()

        # Pass through each layer
        for layer_idx, layer_models in enumerate(self.layer_models_):
            layer_predictions = []

            for name, estimator in layer_models:
                if self.task == 'classification' and hasattr(estimator, 'predict_proba'):
                    pred = estimator.predict_proba(X_current)
                else:
                    pred = estimator.predict(X_current).reshape(-1, 1)
                layer_predictions.append(pred)

            X_layer = np.hstack(layer_predictions)

            if self.passthrough:
                X_current = np.hstack([X_current, X_layer])
            else:
                X_current = X_layer

        # Final prediction
        return self.meta_estimator_.predict(X_current)

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
            Predicted class probabilities.
        """
        if self.task != 'classification':
            raise ValueError("predict_proba() only available for classification")

        if not hasattr(self.meta_estimator_, 'predict_proba'):
            raise ValueError("Meta-estimator does not support predict_proba()")

        X_current = X.copy()

        # Pass through each layer
        for layer_idx, layer_models in enumerate(self.layer_models_):
            layer_predictions = []

            for name, estimator in layer_models:
                if hasattr(estimator, 'predict_proba'):
                    pred = estimator.predict_proba(X_current)
                else:
                    pred = estimator.predict(X_current).reshape(-1, 1)
                layer_predictions.append(pred)

            X_layer = np.hstack(layer_predictions)

            if self.passthrough:
                X_current = np.hstack([X_current, X_layer])
            else:
                X_current = X_layer

        # Final probability prediction
        return self.meta_estimator_.predict_proba(X_current)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    print("=" * 70)
    print("BLENDING ENSEMBLE EXAMPLES")
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

    # Define base models
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]

    # Example 1: Basic Blending
    print("\n1. Basic Blending Ensemble")
    print("-" * 70)
    blending = BlendingEnsemble(
        estimators=base_models,
        meta_estimator=LogisticRegression(),
        test_size=0.2,
        random_state=42,
        verbose=1
    )
    blending.fit(X_train, y_train, task='classification')
    y_pred = blending.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Example 2: Blending with Passthrough
    print("\n2. Blending with Passthrough")
    print("-" * 70)
    blending_passthrough = BlendingEnsemble(
        estimators=base_models,
        meta_estimator=LogisticRegression(),
        test_size=0.2,
        passthrough=True,
        random_state=42,
        verbose=1
    )
    blending_passthrough.fit(X_train, y_train, task='classification')
    y_pred = blending_passthrough.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Example 3: Multi-Layer Blending
    print("\n3. Multi-Layer Blending")
    print("-" * 70)

    layers = [
        # Layer 0
        [
            ('rf', RandomForestClassifier(n_estimators=30, random_state=42)),
            ('svc', SVC(probability=True, random_state=42))
        ],
        # Layer 1
        [
            ('gb', GradientBoostingClassifier(n_estimators=30, random_state=42))
        ]
    ]

    multi_blend = MultiLayerBlending(
        layers=layers,
        meta_estimator=LogisticRegression(),
        test_size=0.2,
        random_state=42,
        verbose=1
    )
    multi_blend.fit(X_train, y_train, task='classification')
    y_pred = multi_blend.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Example 4: Get Base Predictions
    print("\n4. Analyzing Base Model Predictions")
    print("-" * 70)
    base_preds = blending.get_base_predictions(X_test[:5])
    print(f"Base predictions shape: {base_preds.shape}")
    print(f"First sample base predictions:\n{base_preds[0]}")

    print("\n" + "=" * 70)
    print("All blending examples completed!")
    print("=" * 70)
