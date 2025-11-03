"""
Hyperparameter Tuning
GridSearch, RandomSearch, Bayesian Optimization with Optuna, and cross-validation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV,
    cross_val_score, cross_validate,
    KFold, StratifiedKFold, TimeSeriesSplit
)
from sklearn.metrics import make_scorer
from typing import Dict, List, Optional, Union, Callable, Any
import logging
import json
from pathlib import Path
import time

try:
    import optuna
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    from optuna.samplers import TPESampler, RandomSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Install with: pip install optuna")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GridSearchTuner:
    """
    Grid Search for hyperparameter tuning

    Exhaustively searches through specified parameter combinations.

    Best for:
    - Small parameter spaces
    - When you want to try every combination
    - Understanding parameter interactions

    Features:
    - Exhaustive search
    - Cross-validation
    - Parallel processing
    - Results analysis
    """

    def __init__(
        self,
        estimator: Any,
        param_grid: Dict[str, List],
        scoring: Optional[Union[str, Callable]] = None,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 1,
        refit: bool = True
    ):
        """
        Initialize grid search tuner

        Args:
            estimator: Model to tune
            param_grid: Dictionary of parameters to search
            scoring: Scoring function
            cv: Cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            refit: Refit best model on full dataset
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.refit = refit

        self.grid_search = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.results_ = None

    def search(self, X, y):
        """
        Perform grid search

        Args:
            X: Features
            y: Target
        """
        logger.info("=" * 70)
        logger.info("Starting Grid Search")
        logger.info("=" * 70)
        logger.info(f"Parameter grid: {self.param_grid}")
        logger.info(f"Cross-validation folds: {self.cv}")

        start_time = time.time()

        self.grid_search = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            refit=self.refit,
            return_train_score=True
        )

        self.grid_search.fit(X, y)

        elapsed_time = time.time() - start_time

        # Extract results
        self.best_params_ = self.grid_search.best_params_
        self.best_score_ = self.grid_search.best_score_
        self.best_estimator_ = self.grid_search.best_estimator_
        self.results_ = pd.DataFrame(self.grid_search.cv_results_)

        logger.info("=" * 70)
        logger.info("Grid Search Complete")
        logger.info("=" * 70)
        logger.info(f"Time elapsed: {elapsed_time:.2f}s")
        logger.info(f"Best score: {self.best_score_:.6f}")
        logger.info(f"Best parameters: {self.best_params_}")
        logger.info("=" * 70)

        return self

    def get_results(self) -> pd.DataFrame:
        """Get detailed results"""
        return self.results_

    def plot_results(self, param_name: str):
        """Plot results for a specific parameter"""
        if self.results_ is None:
            raise ValueError("No results available. Run search() first.")

        import matplotlib.pyplot as plt

        # Filter results for varying param_name
        param_col = f'param_{param_name}'
        if param_col not in self.results_.columns:
            raise ValueError(f"Parameter {param_name} not found in grid")

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            self.results_[param_col],
            self.results_['mean_test_score'],
            yerr=self.results_['std_test_score'],
            marker='o'
        )
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title(f'Grid Search Results: {param_name}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class RandomSearchTuner:
    """
    Random Search for hyperparameter tuning

    Samples random parameter combinations from specified distributions.

    Best for:
    - Large parameter spaces
    - When grid search is too expensive
    - Continuous parameters

    Features:
    - Random sampling
    - More efficient than grid search
    - Good for continuous spaces
    """

    def __init__(
        self,
        estimator: Any,
        param_distributions: Dict[str, Any],
        n_iter: int = 100,
        scoring: Optional[Union[str, Callable]] = None,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: int = 42,
        refit: bool = True
    ):
        """
        Initialize random search tuner

        Args:
            estimator: Model to tune
            param_distributions: Dictionary of parameter distributions
            n_iter: Number of iterations
            scoring: Scoring function
            cv: Cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            random_state: Random seed
            refit: Refit best model on full dataset
        """
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.refit = refit

        self.random_search = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.results_ = None

    def search(self, X, y):
        """
        Perform random search

        Args:
            X: Features
            y: Target
        """
        logger.info("=" * 70)
        logger.info("Starting Random Search")
        logger.info("=" * 70)
        logger.info(f"Parameter distributions: {self.param_distributions}")
        logger.info(f"Number of iterations: {self.n_iter}")
        logger.info(f"Cross-validation folds: {self.cv}")

        start_time = time.time()

        self.random_search = RandomizedSearchCV(
            estimator=self.estimator,
            param_distributions=self.param_distributions,
            n_iter=self.n_iter,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            refit=self.refit,
            return_train_score=True
        )

        self.random_search.fit(X, y)

        elapsed_time = time.time() - start_time

        # Extract results
        self.best_params_ = self.random_search.best_params_
        self.best_score_ = self.random_search.best_score_
        self.best_estimator_ = self.random_search.best_estimator_
        self.results_ = pd.DataFrame(self.random_search.cv_results_)

        logger.info("=" * 70)
        logger.info("Random Search Complete")
        logger.info("=" * 70)
        logger.info(f"Time elapsed: {elapsed_time:.2f}s")
        logger.info(f"Best score: {self.best_score_:.6f}")
        logger.info(f"Best parameters: {self.best_params_}")
        logger.info("=" * 70)

        return self

    def get_results(self) -> pd.DataFrame:
        """Get detailed results"""
        return self.results_


class OptunaTuner:
    """
    Bayesian Optimization with Optuna

    Uses Tree-structured Parzen Estimator (TPE) for intelligent parameter search.

    Best for:
    - Large parameter spaces
    - Expensive model training
    - When you want the best results with fewer trials
    - Complex optimization objectives

    Features:
    - Bayesian optimization
    - Pruning (early stopping of bad trials)
    - Multi-objective optimization
    - Visualization tools
    """

    def __init__(
        self,
        objective_fn: Callable,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        direction: str = 'maximize',
        sampler: Optional[str] = 'tpe',
        pruner: Optional[str] = 'median',
        study_name: Optional[str] = None,
        storage: Optional[str] = None
    ):
        """
        Initialize Optuna tuner

        Args:
            objective_fn: Objective function to optimize (receives trial object)
            n_trials: Number of trials
            timeout: Timeout in seconds
            direction: 'maximize' or 'minimize'
            sampler: Sampling algorithm ('tpe', 'random')
            pruner: Pruning algorithm ('median', 'halving', None)
            study_name: Name for the study
            storage: Database URL for persistent storage
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed. Install with: pip install optuna")

        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction
        self.study_name = study_name or f'study_{int(time.time())}'
        self.storage = storage

        # Create sampler
        if sampler == 'tpe':
            self.sampler = TPESampler(seed=42)
        elif sampler == 'random':
            self.sampler = RandomSampler(seed=42)
        else:
            self.sampler = None

        # Create pruner
        if pruner == 'median':
            self.pruner = MedianPruner()
        elif pruner == 'halving':
            self.pruner = SuccessiveHalvingPruner()
        else:
            self.pruner = None

        self.study = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_trial_ = None

    def optimize(self, X=None, y=None, **kwargs):
        """
        Run optimization

        Args:
            X: Features (optional, passed to objective function)
            y: Target (optional, passed to objective function)
            **kwargs: Additional arguments for objective function
        """
        logger.info("=" * 70)
        logger.info("Starting Optuna Optimization")
        logger.info("=" * 70)
        logger.info(f"Number of trials: {self.n_trials}")
        logger.info(f"Direction: {self.direction}")
        logger.info(f"Sampler: {type(self.sampler).__name__ if self.sampler else 'Default'}")
        logger.info(f"Pruner: {type(self.pruner).__name__ if self.pruner else 'None'}")

        start_time = time.time()

        # Create study
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True
        )

        # Create wrapped objective
        def wrapped_objective(trial):
            return self.objective_fn(trial, X=X, y=y, **kwargs)

        # Optimize
        self.study.optimize(
            wrapped_objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        elapsed_time = time.time() - start_time

        # Extract results
        self.best_trial_ = self.study.best_trial
        self.best_params_ = self.study.best_params
        self.best_score_ = self.study.best_value

        logger.info("=" * 70)
        logger.info("Optuna Optimization Complete")
        logger.info("=" * 70)
        logger.info(f"Time elapsed: {elapsed_time:.2f}s")
        logger.info(f"Best score: {self.best_score_:.6f}")
        logger.info(f"Best parameters: {self.best_params_}")
        logger.info(f"Number of trials: {len(self.study.trials)}")
        logger.info("=" * 70)

        return self

    def get_results(self) -> pd.DataFrame:
        """Get trial results as DataFrame"""
        return self.study.trials_dataframe()

    def plot_optimization_history(self):
        """Plot optimization history"""
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")

        fig = optuna.visualization.plot_optimization_history(self.study)
        fig.show()

    def plot_param_importances(self):
        """Plot parameter importances"""
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")

        fig = optuna.visualization.plot_param_importances(self.study)
        fig.show()

    def plot_slice(self):
        """Plot parameter slice"""
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")

        fig = optuna.visualization.plot_slice(self.study)
        fig.show()


class CrossValidator:
    """
    Cross-validation utilities

    Supports:
    - K-Fold
    - Stratified K-Fold
    - Time Series Split
    - Leave-One-Out
    - Custom splits
    """

    def __init__(
        self,
        cv_type: str = 'kfold',
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42
    ):
        """
        Initialize cross-validator

        Args:
            cv_type: Type of cross-validation
                - 'kfold': K-Fold
                - 'stratified': Stratified K-Fold (for classification)
                - 'timeseries': Time Series Split
            n_splits: Number of folds
            shuffle: Whether to shuffle data
            random_state: Random seed
        """
        self.cv_type = cv_type
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

        if cv_type == 'kfold':
            self.cv = KFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
        elif cv_type == 'stratified':
            self.cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
        elif cv_type == 'timeseries':
            self.cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            raise ValueError(f"Unknown cv_type: {cv_type}")

    def evaluate(
        self,
        estimator: Any,
        X,
        y,
        scoring: Optional[Union[str, List[str], Dict[str, Callable]]] = None,
        return_train_score: bool = True,
        n_jobs: int = -1,
        verbose: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate model using cross-validation

        Args:
            estimator: Model to evaluate
            X: Features
            y: Target
            scoring: Scoring metric(s)
            return_train_score: Return training scores
            n_jobs: Number of parallel jobs
            verbose: Verbosity level

        Returns:
            Dictionary of scores
        """
        logger.info(f"Running {self.n_splits}-fold cross-validation ({self.cv_type})")

        scores = cross_validate(
            estimator=estimator,
            X=X,
            y=y,
            cv=self.cv,
            scoring=scoring,
            return_train_score=return_train_score,
            n_jobs=n_jobs,
            verbose=verbose
        )

        # Print results
        for key, values in scores.items():
            mean = values.mean()
            std = values.std()
            logger.info(f"{key}: {mean:.6f} (+/- {std:.6f})")

        return scores


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Hyperparameter Tuning Test")
    print("=" * 70)

    # Create sample data
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )

    print(f"Dataset shape: {X.shape}")

    # Test Grid Search
    print("\n1. Grid Search")
    print("-" * 70)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5]
    }

    grid_tuner = GridSearchTuner(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=0
    )

    grid_tuner.search(X, y)

    # Test Random Search
    print("\n2. Random Search")
    print("-" * 70)

    from scipy.stats import randint

    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 10)
    }

    random_tuner = RandomSearchTuner(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=20,
        scoring='accuracy',
        cv=3,
        verbose=0
    )

    random_tuner.search(X, y)

    # Test Optuna (if available)
    if OPTUNA_AVAILABLE:
        print("\n3. Optuna Bayesian Optimization")
        print("-" * 70)

        def objective(trial, X=None, y=None):
            # Define parameter search space
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

            # Create and evaluate model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )

            # Cross-validation
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')

            return scores.mean()

        optuna_tuner = OptunaTuner(
            objective_fn=objective,
            n_trials=20,
            direction='maximize',
            sampler='tpe',
            pruner='median'
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        optuna_tuner.optimize(X=X, y=y)

    else:
        print("\n3. Optuna: Not available")

    # Test Cross-Validation
    print("\n4. Cross-Validation")
    print("-" * 70)

    cv = CrossValidator(cv_type='stratified', n_splits=5)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    scores = cv.evaluate(
        estimator=model,
        X=X,
        y=y,
        scoring=['accuracy', 'f1_weighted'],
        verbose=0
    )

    print("\n" + "=" * 70)
    print("Hyperparameter tuning tested successfully!")
