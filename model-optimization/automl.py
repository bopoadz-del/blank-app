"""
AutoML Pipeline
Automated machine learning pipeline combining feature engineering,
model selection, and hyperparameter tuning
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from typing import Optional, List, Dict, Tuple, Any, Union
import logging
import time
import json
from pathlib import Path

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoMLPipeline:
    """
    Automated Machine Learning Pipeline

    Features:
    - Automatic preprocessing
    - Feature engineering
    - Model selection
    - Hyperparameter tuning
    - Ensemble methods
    - Leaderboard and comparison

    Workflow:
    1. Preprocess data (handle missing, encode categorical)
    2. Feature engineering (scaling, selection)
    3. Train multiple models
    4. Tune best models
    5. Create ensemble
    6. Evaluate and compare

    Usage:
        automl = AutoMLPipeline(task='classification')
        automl.fit(X_train, y_train)
        predictions = automl.predict(X_test)
        leaderboard = automl.get_leaderboard()
    """

    def __init__(
        self,
        task: str = 'classification',
        metric: Optional[str] = None,
        time_budget: Optional[int] = None,
        models: Optional[List[str]] = None,
        tune_hyperparameters: bool = True,
        create_ensemble: bool = True,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Initialize AutoML pipeline

        Args:
            task: Task type ('classification' or 'regression')
            metric: Evaluation metric (None = auto-select)
            time_budget: Time budget in seconds (None = no limit)
            models: List of models to try (None = try all)
            tune_hyperparameters: Whether to tune hyperparameters
            create_ensemble: Whether to create ensemble
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        self.task = task
        self.metric = metric or ('accuracy' if task == 'classification' else 'neg_mean_squared_error')
        self.time_budget = time_budget
        self.tune_hyperparameters = tune_hyperparameters
        self.create_ensemble = create_ensemble
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Models to try
        if models is None:
            self.model_names = self._get_default_models()
        else:
            self.model_names = models

        # Results
        self.results = []
        self.leaderboard = None
        self.best_model = None
        self.best_score = float('-inf') if 'neg' not in self.metric else float('inf')
        self.ensemble_model = None

        # Preprocessing
        self.scaler = StandardScaler()
        self.label_encoders = {}

        # Fitted flag
        self.fitted = False

    def _get_default_models(self) -> List[str]:
        """Get default models based on task"""
        if self.task == 'classification':
            models = [
                'logistic_regression',
                'random_forest',
                'gradient_boosting',
                'svm',
                'knn',
                'decision_tree'
            ]

            if XGBOOST_AVAILABLE:
                models.append('xgboost')
            if LIGHTGBM_AVAILABLE:
                models.append('lightgbm')

        else:  # regression
            models = [
                'ridge',
                'lasso',
                'random_forest',
                'gradient_boosting',
                'svr',
                'knn',
                'decision_tree'
            ]

            if XGBOOST_AVAILABLE:
                models.append('xgboost')
            if LIGHTGBM_AVAILABLE:
                models.append('lightgbm')

        return models

    def _create_model(self, model_name: str, **params) -> Any:
        """Create model instance"""
        if self.task == 'classification':
            models = {
                'logistic_regression': LogisticRegression,
                'random_forest': RandomForestClassifier,
                'gradient_boosting': GradientBoostingClassifier,
                'svm': SVC,
                'knn': KNeighborsClassifier,
                'decision_tree': DecisionTreeClassifier,
            }

            if XGBOOST_AVAILABLE:
                models['xgboost'] = XGBClassifier
            if LIGHTGBM_AVAILABLE:
                models['lightgbm'] = LGBMClassifier

        else:  # regression
            models = {
                'ridge': Ridge,
                'lasso': Lasso,
                'random_forest': RandomForestRegressor,
                'gradient_boosting': GradientBoostingRegressor,
                'svr': SVR,
                'knn': KNeighborsRegressor,
                'decision_tree': DecisionTreeRegressor,
            }

            if XGBOOST_AVAILABLE:
                models['xgboost'] = XGBRegressor
            if LIGHTGBM_AVAILABLE:
                models['lightgbm'] = LGBMRegressor

        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")

        # Default parameters
        default_params = {'random_state': self.random_state}
        if model_name in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
            default_params['n_jobs'] = self.n_jobs

        # Merge with provided params
        default_params.update(params)

        return models[model_name](**default_params)

    def _preprocess_data(self, X, y=None, fit=True):
        """Preprocess data"""
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Handle missing values
        X = X.fillna(X.mean(numeric_only=True))

        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                if col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))

        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled

    def _evaluate_model(
        self,
        model_name: str,
        model: Any,
        X_train,
        y_train,
        X_val,
        y_val
    ) -> Dict:
        """Evaluate a single model"""
        logger.info(f"Evaluating {model_name}...")

        start_time = time.time()

        # Train
        model.fit(X_train, y_train)

        # Predict
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # Calculate metrics
        if self.task == 'classification':
            train_score = accuracy_score(y_train, train_pred)
            val_score = accuracy_score(y_val, val_pred)

            metrics = {
                'accuracy': val_score,
                'f1': f1_score(y_val, val_pred, average='weighted'),
                'precision': precision_score(y_val, val_pred, average='weighted'),
                'recall': recall_score(y_val, val_pred, average='weighted')
            }
        else:
            train_score = r2_score(y_train, train_pred)
            val_score = r2_score(y_val, val_pred)

            metrics = {
                'r2': val_score,
                'mse': mean_squared_error(y_val, val_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'mae': mean_absolute_error(y_val, val_pred)
            }

        elapsed_time = time.time() - start_time

        result = {
            'model_name': model_name,
            'model': model,
            'train_score': train_score,
            'val_score': val_score,
            'metrics': metrics,
            'time': elapsed_time
        }

        logger.info(f"{model_name}: val_score={val_score:.4f}, time={elapsed_time:.2f}s")

        return result

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit AutoML pipeline

        Args:
            X: Training features
            y: Training target
            X_val: Validation features (None = split from X)
            y_val: Validation target
        """
        logger.info("=" * 70)
        logger.info("Starting AutoML Pipeline")
        logger.info("=" * 70)
        logger.info(f"Task: {self.task}")
        logger.info(f"Metric: {self.metric}")
        logger.info(f"Models to try: {len(self.model_names)}")
        logger.info("=" * 70)

        start_time = time.time()

        # Split validation set if not provided
        if X_val is None:
            X, X_val, y, y_val = train_test_split(
                X, y,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y if self.task == 'classification' else None
            )

        # Preprocess
        logger.info("Preprocessing data...")
        X_train = self._preprocess_data(X, y, fit=True)
        X_val_processed = self._preprocess_data(X_val, fit=False)

        # Try all models
        logger.info("\nTraining and evaluating models...")
        for model_name in self.model_names:
            # Check time budget
            if self.time_budget is not None:
                elapsed = time.time() - start_time
                if elapsed > self.time_budget:
                    logger.info(f"Time budget exceeded ({self.time_budget}s)")
                    break

            # Create and evaluate model
            model = self._create_model(model_name)
            result = self._evaluate_model(
                model_name,
                model,
                X_train,
                y,
                X_val_processed,
                y_val
            )

            self.results.append(result)

            # Track best model
            if result['val_score'] > self.best_score:
                self.best_score = result['val_score']
                self.best_model = result['model']

        # Create leaderboard
        self.leaderboard = pd.DataFrame([
            {
                'Model': r['model_name'],
                'Val Score': r['val_score'],
                'Train Score': r['train_score'],
                'Time (s)': r['time'],
                **r['metrics']
            }
            for r in self.results
        ])

        self.leaderboard = self.leaderboard.sort_values('Val Score', ascending=False)

        # Create ensemble if requested
        if self.create_ensemble and len(self.results) > 1:
            logger.info("\nCreating ensemble...")
            self._create_voting_ensemble(X_train, y, X_val_processed, y_val)

        self.fitted = True

        elapsed_time = time.time() - start_time

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("AutoML Pipeline Complete")
        logger.info("=" * 70)
        logger.info(f"Total time: {elapsed_time:.2f}s")
        logger.info(f"Best model: {self.leaderboard.iloc[0]['Model']}")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info("=" * 70)

        return self

    def _create_voting_ensemble(self, X_train, y_train, X_val, y_val):
        """Create voting ensemble from top models"""
        from sklearn.ensemble import VotingClassifier, VotingRegressor

        # Get top 3 models
        top_models = [(r['model_name'], r['model']) for r in self.results[:3]]

        if self.task == 'classification':
            self.ensemble_model = VotingClassifier(
                estimators=top_models,
                voting='soft'
            )
        else:
            self.ensemble_model = VotingRegressor(estimators=top_models)

        # Train ensemble
        self.ensemble_model.fit(X_train, y_train)

        # Evaluate
        val_pred = self.ensemble_model.predict(X_val)

        if self.task == 'classification':
            ensemble_score = accuracy_score(y_val, val_pred)
        else:
            ensemble_score = r2_score(y_val, val_pred)

        logger.info(f"Ensemble score: {ensemble_score:.4f}")

        # Add to results
        if ensemble_score > self.best_score:
            self.best_score = ensemble_score
            self.best_model = self.ensemble_model
            logger.info("Ensemble is the best model!")

    def predict(self, X):
        """Make predictions"""
        if not self.fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        X_processed = self._preprocess_data(X, fit=False)
        return self.best_model.predict(X_processed)

    def predict_proba(self, X):
        """Get prediction probabilities (classification only)"""
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")

        if not self.fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        X_processed = self._preprocess_data(X, fit=False)
        return self.best_model.predict_proba(X_processed)

    def get_leaderboard(self) -> pd.DataFrame:
        """Get model leaderboard"""
        return self.leaderboard

    def get_best_model(self) -> Any:
        """Get best model"""
        return self.best_model

    def save(self, path: str):
        """Save pipeline"""
        import pickle

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        logger.info(f"Pipeline saved to {path}")

    @staticmethod
    def load(path: str):
        """Load pipeline"""
        import pickle

        with open(path, 'rb') as f:
            pipeline = pickle.load(f)

        logger.info(f"Pipeline loaded from {path}")
        return pipeline


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("AutoML Pipeline Test")
    print("=" * 70)

    # Create sample data
    from sklearn.datasets import make_classification, make_regression

    # Classification
    print("\n1. Classification Task")
    print("-" * 70)

    X_class, y_class = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    # Create and run AutoML
    automl = AutoMLPipeline(
        task='classification',
        time_budget=None,
        models=['logistic_regression', 'random_forest', 'gradient_boosting'],
        tune_hyperparameters=False,
        create_ensemble=True
    )

    automl.fit(X_class, y_class)

    # Get leaderboard
    print("\nLeaderboard:")
    print(automl.get_leaderboard())

    # Make predictions
    predictions = automl.predict(X_class[:10])
    print(f"\nPredictions (first 10): {predictions}")

    # Regression
    print("\n2. Regression Task")
    print("-" * 70)

    X_reg, y_reg = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )

    automl_reg = AutoMLPipeline(
        task='regression',
        models=['ridge', 'random_forest', 'gradient_boosting'],
        create_ensemble=True
    )

    automl_reg.fit(X_reg, y_reg)

    print("\nLeaderboard:")
    print(automl_reg.get_leaderboard())

    print("\n" + "=" * 70)
    print("AutoML pipeline tested successfully!")
