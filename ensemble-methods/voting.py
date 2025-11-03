"""
Voting Ensemble Methods

This module implements Voting ensemble methods that combine predictions from
multiple models through majority voting (hard voting) or weighted averaging
of probabilities (soft voting).

Key Features:
- VotingClassifier: Hard and soft voting for classification
- VotingRegressor: Weighted averaging for regression
- Customizable voting weights
- Support for heterogeneous estimators
- Probability calibration support

Author: ML Framework Team
"""

import numpy as np
from typing import List, Optional, Union, Tuple
from sklearn.ensemble import VotingClassifier as SKVotingClassifier
from sklearn.ensemble import VotingRegressor as SKVotingRegressor
from sklearn.base import BaseEstimator, clone
import warnings


class VotingEnsemble:
    """
    Voting Ensemble for Classification and Regression

    Voting combines predictions from multiple diverse models:

    For Classification:
    - Hard Voting: Majority vote (mode of predicted classes)
    - Soft Voting: Weighted average of predicted probabilities

    For Regression:
    - Weighted average of predictions

    Hard voting is useful when models have similar performance, while soft voting
    can leverage probability information and typically performs better when models
    are well-calibrated.

    Parameters:
    -----------
    estimators : list of (str, estimator) tuples
        List of (name, estimator) tuples. Estimators must have fit and predict
        methods. For soft voting, classification estimators must also have
        predict_proba method.
    voting : {'hard', 'soft'}, default='hard'
        For classification:
        - 'hard': Use predicted class labels for majority vote.
        - 'soft': Predict class label based on argmax of sums of predicted probabilities.
        For regression, this parameter is ignored.
    weights : array-like of shape (n_estimators,), default=None
        Sequence of weights (float or int) to weight the occurrences of predicted
        class labels (hard voting) or class probabilities (soft voting). If None,
        uniform weights are assumed.
    n_jobs : int, default=None
        The number of jobs to run in parallel for fit. -1 means using all processors.
    flatten_transform : bool, default=True
        Affects shape of transform output when voting='soft'. Not used if voting='hard'.
    verbose : int, default=0
        Verbosity level.

    Attributes:
    -----------
    estimators_ : list of estimators
        The fitted estimators.
    named_estimators_ : dict
        Dictionary to access fitted estimators by name.

    Example:
    --------
    >>> from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    >>> from sklearn.svm import SVC
    >>> from sklearn.datasets import make_classification
    >>>
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>>
    >>> # Define base models
    >>> estimators = [
    ...     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ...     ('gb', GradientBoostingClassifier(n_estimators=10, random_state=42)),
    ...     ('svc', SVC(probability=True, random_state=42))
    ... ]
    >>>
    >>> # Hard Voting
    >>> voting_hard = VotingEnsemble(estimators=estimators, voting='hard')
    >>> voting_hard.fit(X, y, task='classification')
    >>> predictions = voting_hard.predict(X)
    >>>
    >>> # Soft Voting (usually better performance)
    >>> voting_soft = VotingEnsemble(estimators=estimators, voting='soft')
    >>> voting_soft.fit(X, y, task='classification')
    >>> predictions = voting_soft.predict(X)
    >>>
    >>> # Weighted Voting (give more weight to better models)
    >>> voting_weighted = VotingEnsemble(
    ...     estimators=estimators,
    ...     voting='soft',
    ...     weights=[2, 2, 1]  # RF and GB get 2x weight
    ... )
    >>> voting_weighted.fit(X, y, task='classification')
    """

    def __init__(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        voting: str = 'hard',
        weights: Optional[np.ndarray] = None,
        n_jobs: Optional[int] = None,
        flatten_transform: bool = True,
        verbose: int = 0
    ):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
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
        Fit the voting ensemble.

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
            # Validate voting parameter
            if self.voting not in ['hard', 'soft']:
                raise ValueError(f"voting must be 'hard' or 'soft', got {self.voting}")

            # For soft voting, check that all estimators support predict_proba
            if self.voting == 'soft':
                for name, estimator in self.estimators:
                    if not hasattr(estimator, 'predict_proba'):
                        raise ValueError(
                            f"Estimator {name} does not support predict_proba. "
                            "Use voting='hard' or an estimator with predict_proba."
                        )

            self.model_ = SKVotingClassifier(
                estimators=self.estimators,
                voting=self.voting,
                weights=self.weights,
                n_jobs=self.n_jobs,
                flatten_transform=self.flatten_transform,
                verbose=self.verbose
            )
        else:
            # For regression, voting parameter is ignored
            self.model_ = SKVotingRegressor(
                estimators=self.estimators,
                weights=self.weights,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )

        # Fit the ensemble
        if sample_weight is not None:
            self.model_.fit(X, y, sample_weight=sample_weight)
        else:
            self.model_.fit(X, y)

        if self.verbose > 0:
            print(f"Voting ensemble fitted with {len(self.estimators)} estimators")
            if task == 'classification':
                print(f"Voting strategy: {self.voting}")
            if self.weights is not None:
                print(f"Weights: {self.weights}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels or regression values for X.

        For hard voting classification: returns mode of predicted labels.
        For soft voting classification: returns argmax of averaged probabilities.
        For regression: returns weighted average of predictions.

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
        Predict class probabilities for X (classification with soft voting only).

        Returns averaged probabilities from all estimators.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns:
        --------
        probabilities : ndarray of shape (n_samples, n_classes)
            Averaged class probabilities.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.task != 'classification':
            raise ValueError("predict_proba() only available for classification tasks")

        if self.voting != 'soft':
            raise ValueError("predict_proba() only available with voting='soft'")

        return self.model_.predict_proba(X)

    def get_individual_predictions(self, X: np.ndarray) -> dict:
        """
        Get predictions from each individual estimator.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns:
        --------
        predictions : dict
            Dictionary mapping estimator names to their predictions.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = {}
        for name, estimator in self.model_.named_estimators_.items():
            predictions[name] = estimator.predict(X)

        return predictions

    def get_individual_probabilities(self, X: np.ndarray) -> dict:
        """
        Get probability predictions from each individual estimator (classification only).

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns:
        --------
        probabilities : dict
            Dictionary mapping estimator names to their predicted probabilities.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.task != 'classification':
            raise ValueError("get_individual_probabilities() only available for classification")

        probabilities = {}
        for name, estimator in self.model_.named_estimators_.items():
            if hasattr(estimator, 'predict_proba'):
                probabilities[name] = estimator.predict_proba(X)
            else:
                warnings.warn(f"Estimator {name} does not support predict_proba")

        return probabilities

    def set_weights(self, weights: np.ndarray):
        """
        Update voting weights after fitting.

        Parameters:
        -----------
        weights : array-like of shape (n_estimators,)
            New weights for the estimators.

        Returns:
        --------
        self : object
            Estimator with updated weights.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if len(weights) != len(self.estimators):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of estimators ({len(self.estimators)})"
            )

        self.weights = weights
        self.model_.weights = weights

        return self


class WeightedVotingEnsemble:
    """
    Weighted Voting Ensemble with Automatic Weight Optimization

    This class extends basic voting by automatically learning optimal weights
    for each base estimator based on their individual performance.

    Parameters:
    -----------
    estimators : list of (str, estimator) tuples
        Base estimators for the ensemble.
    voting : {'hard', 'soft'}, default='soft'
        Voting strategy.
    weight_optimization : {'accuracy', 'f1', 'roc_auc', 'uniform'}, default='accuracy'
        Method to compute weights:
        - 'accuracy': Weight by accuracy on validation set
        - 'f1': Weight by F1 score on validation set
        - 'roc_auc': Weight by ROC AUC on validation set
        - 'uniform': Equal weights (same as standard voting)
    n_jobs : int, default=None
        Number of parallel jobs.
    verbose : int, default=0
        Verbosity level.

    Example:
    --------
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    >>>
    >>> weighted_voting = WeightedVotingEnsemble(
    ...     estimators=estimators,
    ...     voting='soft',
    ...     weight_optimization='accuracy'
    ... )
    >>>
    >>> # Fit with validation data to learn weights
    >>> weighted_voting.fit(X_train, y_train, X_val, y_val, task='classification')
    >>> print(f"Learned weights: {weighted_voting.weights_}")
    """

    def __init__(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        voting: str = 'soft',
        weight_optimization: str = 'accuracy',
        n_jobs: Optional[int] = None,
        verbose: int = 0
    ):
        self.estimators = estimators
        self.voting = voting
        self.weight_optimization = weight_optimization
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.voting_ensemble_ = None
        self.weights_ = None
        self.task = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        task: str = 'classification'
    ):
        """
        Fit the weighted voting ensemble.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Training target.
        X_val : array-like of shape (n_val_samples, n_features), default=None
            Validation data for weight optimization.
        y_val : array-like of shape (n_val_samples,), default=None
            Validation target for weight optimization.
        task : str, default='classification'
            Task type.

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        self.task = task

        # Compute weights if validation data provided
        if X_val is not None and y_val is not None and self.weight_optimization != 'uniform':
            self.weights_ = self._compute_weights(X, y, X_val, y_val, task)
        else:
            self.weights_ = None

        # Create and fit voting ensemble
        self.voting_ensemble_ = VotingEnsemble(
            estimators=self.estimators,
            voting=self.voting,
            weights=self.weights_,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )

        self.voting_ensemble_.fit(X, y, task=task)

        if self.verbose > 0:
            print(f"Weighted voting ensemble fitted")
            if self.weights_ is not None:
                for (name, _), weight in zip(self.estimators, self.weights_):
                    print(f"  {name}: weight = {weight:.4f}")

        return self

    def _compute_weights(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        task: str
    ) -> np.ndarray:
        """Compute optimal weights based on validation performance."""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        weights = []

        for name, estimator in self.estimators:
            # Clone and fit estimator
            est = clone(estimator)
            est.fit(X_train, y_train)

            # Evaluate on validation set
            if task == 'classification':
                if self.weight_optimization == 'accuracy':
                    y_pred = est.predict(X_val)
                    score = accuracy_score(y_val, y_pred)
                elif self.weight_optimization == 'f1':
                    y_pred = est.predict(X_val)
                    score = f1_score(y_val, y_pred, average='weighted')
                elif self.weight_optimization == 'roc_auc':
                    if hasattr(est, 'predict_proba'):
                        y_pred_proba = est.predict_proba(X_val)
                        score = roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='weighted')
                    else:
                        # Fall back to accuracy if no predict_proba
                        y_pred = est.predict(X_val)
                        score = accuracy_score(y_val, y_pred)
                else:
                    score = 1.0  # Uniform
            else:
                from sklearn.metrics import r2_score
                y_pred = est.predict(X_val)
                score = r2_score(y_val, y_pred)

            weights.append(max(score, 0.0001))  # Avoid zero weights

            if self.verbose > 0:
                print(f"  {name}: validation score = {score:.4f}")

        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / np.sum(weights) * len(weights)

        return weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the weighted voting ensemble."""
        if self.voting_ensemble_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.voting_ensemble_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.voting_ensemble_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.voting_ensemble_.predict_proba(X)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    print("=" * 70)
    print("VOTING ENSEMBLE EXAMPLES")
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

    # Define base estimators
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ('svc', SVC(probability=True, random_state=42)),
        ('lr', LogisticRegression(random_state=42))
    ]

    # Example 1: Hard Voting
    print("\n1. Hard Voting (Majority Vote)")
    print("-" * 70)
    voting_hard = VotingEnsemble(estimators=estimators, voting='hard', verbose=1)
    voting_hard.fit(X_train, y_train, task='classification')
    y_pred = voting_hard.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Example 2: Soft Voting
    print("\n2. Soft Voting (Averaged Probabilities)")
    print("-" * 70)
    voting_soft = VotingEnsemble(estimators=estimators, voting='soft', verbose=1)
    voting_soft.fit(X_train, y_train, task='classification')
    y_pred = voting_soft.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Example 3: Weighted Voting
    print("\n3. Weighted Voting (Custom Weights)")
    print("-" * 70)
    weights = [2, 2, 1, 1]  # Give more weight to RF and GB
    voting_weighted = VotingEnsemble(
        estimators=estimators,
        voting='soft',
        weights=weights,
        verbose=1
    )
    voting_weighted.fit(X_train, y_train, task='classification')
    y_pred = voting_weighted.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Example 4: Automatic Weight Optimization
    print("\n4. Automatic Weight Optimization")
    print("-" * 70)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    weighted_auto = WeightedVotingEnsemble(
        estimators=estimators,
        voting='soft',
        weight_optimization='accuracy',
        verbose=1
    )
    weighted_auto.fit(X_tr, y_tr, X_val, y_val, task='classification')
    y_pred = weighted_auto.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Example 5: Individual Predictions
    print("\n5. Analyzing Individual Estimator Predictions")
    print("-" * 70)
    individual_preds = voting_soft.get_individual_predictions(X_test[:5])
    print("Predictions from each estimator (first 5 samples):")
    for name, preds in individual_preds.items():
        print(f"  {name}: {preds}")

    print("\n" + "=" * 70)
    print("All voting examples completed!")
    print("=" * 70)
