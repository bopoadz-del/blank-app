"""
Comprehensive Examples for Ensemble Methods

This module provides complete, working examples demonstrating all ensemble
techniques implemented in this framework.

Examples:
1. Bagging Ensemble Comparison
2. Boosting Methods Comparison
3. Stacking and Multi-Level Stacking
4. Voting Ensemble (Hard vs Soft)
5. Blending Ensemble
6. Complete Ensemble Comparison

Author: ML Framework Team
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_classification, make_regression, load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Base models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier

# Import our ensemble methods
from bagging import BaggingEnsemble, RandomForestEnsemble, ExtraTreesEnsemble
from boosting import (
    AdaBoostEnsemble,
    GradientBoostingEnsemble,
    XGBoostEnsemble,
    LightGBMEnsemble,
    CatBoostEnsemble,
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    CATBOOST_AVAILABLE
)
from stacking import StackingEnsemble, MultiLevelStacking
from voting import VotingEnsemble, WeightedVotingEnsemble
from blending import BlendingEnsemble, MultiLayerBlending

import time


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n{title}")
    print("-" * 80)


def example_1_bagging_comparison():
    """
    Example 1: Bagging Ensemble Comparison

    Compare different bagging methods:
    - Basic Bagging with Decision Trees
    - Random Forest
    - Extra Trees
    """
    print_section("EXAMPLE 1: BAGGING ENSEMBLE COMPARISON")

    # Generate data
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

    results = {}

    # 1. Basic Bagging
    print_subsection("1.1 Basic Bagging Ensemble")
    start_time = time.time()
    bagging = BaggingEnsemble(
        base_estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=50,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42
    )
    bagging.fit(X_train, y_train, task='classification')
    y_pred = bagging.predict(X_test)
    train_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    oob_score = bagging.get_oob_score()
    results['Bagging'] = {'accuracy': accuracy, 'oob_score': oob_score, 'time': train_time}

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"OOB Score: {oob_score:.4f}")
    print(f"Training Time: {train_time:.2f}s")

    # 2. Random Forest
    print_subsection("1.2 Random Forest Ensemble")
    start_time = time.time()
    rf = RandomForestEnsemble(
        n_estimators=50,
        max_depth=10,
        max_features='sqrt',
        oob_score=True,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train, task='classification')
    y_pred = rf.predict(X_test)
    train_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    oob_score = rf.get_oob_score()
    importance = rf.get_feature_importance()
    results['Random Forest'] = {'accuracy': accuracy, 'oob_score': oob_score, 'time': train_time}

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"OOB Score: {oob_score:.4f}")
    print(f"Training Time: {train_time:.2f}s")
    print(f"Top 5 Important Features: {np.argsort(importance)[-5:][::-1]}")

    # 3. Extra Trees
    print_subsection("1.3 Extra Trees Ensemble")
    start_time = time.time()
    et = ExtraTreesEnsemble(
        n_estimators=50,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )
    et.fit(X_train, y_train, task='classification')
    y_pred = et.predict(X_test)
    train_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    results['Extra Trees'] = {'accuracy': accuracy, 'time': train_time}

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Training Time: {train_time:.2f}s")

    # Summary
    print_subsection("1.4 Summary")
    print(f"{'Method':<20} {'Accuracy':<12} {'OOB Score':<12} {'Time (s)':<10}")
    print("-" * 80)
    for method, metrics in results.items():
        oob = metrics.get('oob_score', 'N/A')
        oob_str = f"{oob:.4f}" if isinstance(oob, float) else oob
        print(f"{method:<20} {metrics['accuracy']:<12.4f} {oob_str:<12} {metrics['time']:<10.2f}")


def example_2_boosting_comparison():
    """
    Example 2: Boosting Methods Comparison

    Compare different boosting algorithms:
    - AdaBoost
    - Gradient Boosting
    - XGBoost (if available)
    - LightGBM (if available)
    - CatBoost (if available)
    """
    print_section("EXAMPLE 2: BOOSTING METHODS COMPARISON")

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

    results = {}

    # 1. AdaBoost
    print_subsection("2.1 AdaBoost")
    start_time = time.time()
    ada = AdaBoostEnsemble(n_estimators=100, learning_rate=1.0, random_state=42)
    ada.fit(X_train, y_train, task='classification')
    y_pred = ada.predict(X_test)
    train_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    results['AdaBoost'] = {'accuracy': accuracy, 'time': train_time}
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Training Time: {train_time:.2f}s")

    # 2. Gradient Boosting
    print_subsection("2.2 Gradient Boosting")
    start_time = time.time()
    gb = GradientBoostingEnsemble(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train, y_train, task='classification')
    y_pred = gb.predict(X_test)
    train_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    results['Gradient Boosting'] = {'accuracy': accuracy, 'time': train_time}
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Training Time: {train_time:.2f}s")

    # 3. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        print_subsection("2.3 XGBoost")
        start_time = time.time()
        xgb_model = XGBoostEnsemble(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        xgb_model.fit(X_train, y_train, task='classification', verbose=False)
        y_pred = xgb_model.predict(X_test)
        train_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        results['XGBoost'] = {'accuracy': accuracy, 'time': train_time}
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Training Time: {train_time:.2f}s")

    # 4. LightGBM (if available)
    if LIGHTGBM_AVAILABLE:
        print_subsection("2.4 LightGBM")
        start_time = time.time()
        lgbm = LightGBMEnsemble(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42
        )
        lgbm.fit(X_train, y_train, task='classification', verbose=False)
        y_pred = lgbm.predict(X_test)
        train_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        results['LightGBM'] = {'accuracy': accuracy, 'time': train_time}
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Training Time: {train_time:.2f}s")

    # 5. CatBoost (if available)
    if CATBOOST_AVAILABLE:
        print_subsection("2.5 CatBoost")
        start_time = time.time()
        cat = CatBoostEnsemble(
            n_estimators=100,
            learning_rate=0.1,
            depth=6,
            verbose=False,
            random_state=42
        )
        cat.fit(X_train, y_train, task='classification')
        y_pred = cat.predict(X_test)
        train_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        results['CatBoost'] = {'accuracy': accuracy, 'time': train_time}
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Training Time: {train_time:.2f}s")

    # Summary
    print_subsection("2.6 Summary")
    print(f"{'Method':<20} {'Accuracy':<12} {'Time (s)':<10}")
    print("-" * 80)
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['accuracy']:<12.4f} {metrics['time']:<10.2f}")


def example_3_stacking():
    """
    Example 3: Stacking and Multi-Level Stacking

    Demonstrate:
    - Basic Stacking
    - Stacking with Passthrough
    - Multi-Level Stacking
    """
    print_section("EXAMPLE 3: STACKING ENSEMBLE")

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
        ('rf', RandomForestClassifier(n_estimators=30, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=30, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]

    results = {}

    # 1. Basic Stacking
    print_subsection("3.1 Basic Stacking")
    start_time = time.time()
    stacking = StackingEnsemble(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=5,
        stack_method='auto'
    )
    stacking.fit(X_train, y_train, task='classification')
    y_pred = stacking.predict(X_test)
    train_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    results['Stacking'] = {'accuracy': accuracy, 'time': train_time}
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Training Time: {train_time:.2f}s")

    # 2. Stacking with Passthrough
    print_subsection("3.2 Stacking with Passthrough")
    start_time = time.time()
    stacking_pass = StackingEnsemble(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=5,
        passthrough=True
    )
    stacking_pass.fit(X_train, y_train, task='classification')
    y_pred = stacking_pass.predict(X_test)
    train_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    results['Stacking + Passthrough'] = {'accuracy': accuracy, 'time': train_time}
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Training Time: {train_time:.2f}s")

    # 3. Multi-Level Stacking
    print_subsection("3.3 Multi-Level Stacking")
    levels = [
        [
            ('rf', RandomForestClassifier(n_estimators=20, random_state=42)),
            ('svc', SVC(probability=True, random_state=42))
        ],
        [
            ('gb', GradientBoostingClassifier(n_estimators=20, random_state=42))
        ]
    ]

    start_time = time.time()
    multi_stack = MultiLevelStacking(
        levels=levels,
        final_estimator=LogisticRegression(),
        cv=5
    )
    multi_stack.fit(X_train, y_train, task='classification')
    y_pred = multi_stack.predict(X_test)
    train_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    results['Multi-Level Stacking'] = {'accuracy': accuracy, 'time': train_time}
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Training Time: {train_time:.2f}s")

    # Summary
    print_subsection("3.4 Summary")
    print(f"{'Method':<25} {'Accuracy':<12} {'Time (s)':<10}")
    print("-" * 80)
    for method, metrics in results.items():
        print(f"{method:<25} {metrics['accuracy']:<12.4f} {metrics['time']:<10.2f}")


def example_4_voting():
    """
    Example 4: Voting Ensemble (Hard vs Soft)

    Compare:
    - Hard Voting
    - Soft Voting
    - Weighted Voting
    - Automatic Weight Optimization
    """
    print_section("EXAMPLE 4: VOTING ENSEMBLE")

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

    # Define estimators
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=30, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=30, random_state=42)),
        ('svc', SVC(probability=True, random_state=42)),
        ('lr', LogisticRegression(random_state=42))
    ]

    results = {}

    # 1. Hard Voting
    print_subsection("4.1 Hard Voting (Majority Vote)")
    start_time = time.time()
    voting_hard = VotingEnsemble(estimators=estimators, voting='hard')
    voting_hard.fit(X_train, y_train, task='classification')
    y_pred = voting_hard.predict(X_test)
    train_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    results['Hard Voting'] = {'accuracy': accuracy, 'time': train_time}
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Training Time: {train_time:.2f}s")

    # 2. Soft Voting
    print_subsection("4.2 Soft Voting (Averaged Probabilities)")
    start_time = time.time()
    voting_soft = VotingEnsemble(estimators=estimators, voting='soft')
    voting_soft.fit(X_train, y_train, task='classification')
    y_pred = voting_soft.predict(X_test)
    train_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    results['Soft Voting'] = {'accuracy': accuracy, 'time': train_time}
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Training Time: {train_time:.2f}s")

    # 3. Weighted Voting
    print_subsection("4.3 Weighted Voting (Custom Weights)")
    weights = [2, 2, 1, 1]  # Give more weight to RF and GB
    start_time = time.time()
    voting_weighted = VotingEnsemble(
        estimators=estimators,
        voting='soft',
        weights=weights
    )
    voting_weighted.fit(X_train, y_train, task='classification')
    y_pred = voting_weighted.predict(X_test)
    train_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    results['Weighted Voting'] = {'accuracy': accuracy, 'time': train_time}
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Weights: {weights}")
    print(f"Training Time: {train_time:.2f}s")

    # 4. Automatic Weight Optimization
    print_subsection("4.4 Automatic Weight Optimization")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    start_time = time.time()
    weighted_auto = WeightedVotingEnsemble(
        estimators=estimators,
        voting='soft',
        weight_optimization='accuracy'
    )
    weighted_auto.fit(X_tr, y_tr, X_val, y_val, task='classification')
    y_pred = weighted_auto.predict(X_test)
    train_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    results['Auto-Weighted Voting'] = {'accuracy': accuracy, 'time': train_time}
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Learned Weights: {weighted_auto.weights_}")
    print(f"Training Time: {train_time:.2f}s")

    # Summary
    print_subsection("4.5 Summary")
    print(f"{'Method':<25} {'Accuracy':<12} {'Time (s)':<10}")
    print("-" * 80)
    for method, metrics in results.items():
        print(f"{method:<25} {metrics['accuracy']:<12.4f} {metrics['time']:<10.2f}")


def example_5_blending():
    """
    Example 5: Blending Ensemble

    Demonstrate:
    - Basic Blending
    - Blending with Passthrough
    - Multi-Layer Blending
    """
    print_section("EXAMPLE 5: BLENDING ENSEMBLE")

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
        ('rf', RandomForestClassifier(n_estimators=30, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=30, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]

    results = {}

    # 1. Basic Blending
    print_subsection("5.1 Basic Blending")
    start_time = time.time()
    blending = BlendingEnsemble(
        estimators=base_models,
        meta_estimator=LogisticRegression(),
        test_size=0.2,
        random_state=42
    )
    blending.fit(X_train, y_train, task='classification')
    y_pred = blending.predict(X_test)
    train_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    results['Blending'] = {'accuracy': accuracy, 'time': train_time}
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Training Time: {train_time:.2f}s")

    # 2. Blending with Passthrough
    print_subsection("5.2 Blending with Passthrough")
    start_time = time.time()
    blending_pass = BlendingEnsemble(
        estimators=base_models,
        meta_estimator=LogisticRegression(),
        test_size=0.2,
        passthrough=True,
        random_state=42
    )
    blending_pass.fit(X_train, y_train, task='classification')
    y_pred = blending_pass.predict(X_test)
    train_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    results['Blending + Passthrough'] = {'accuracy': accuracy, 'time': train_time}
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Training Time: {train_time:.2f}s")

    # 3. Multi-Layer Blending
    print_subsection("5.3 Multi-Layer Blending")
    layers = [
        [
            ('rf', RandomForestClassifier(n_estimators=20, random_state=42)),
            ('svc', SVC(probability=True, random_state=42))
        ],
        [
            ('gb', GradientBoostingClassifier(n_estimators=20, random_state=42))
        ]
    ]

    start_time = time.time()
    multi_blend = MultiLayerBlending(
        layers=layers,
        meta_estimator=LogisticRegression(),
        test_size=0.2,
        random_state=42
    )
    multi_blend.fit(X_train, y_train, task='classification')
    y_pred = multi_blend.predict(X_test)
    train_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    results['Multi-Layer Blending'] = {'accuracy': accuracy, 'time': train_time}
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Training Time: {train_time:.2f}s")

    # Summary
    print_subsection("5.4 Summary")
    print(f"{'Method':<25} {'Accuracy':<12} {'Time (s)':<10}")
    print("-" * 80)
    for method, metrics in results.items():
        print(f"{method:<25} {metrics['accuracy']:<12.4f} {metrics['time']:<10.2f}")


def example_6_complete_comparison():
    """
    Example 6: Complete Ensemble Comparison

    Compare all ensemble methods on the same dataset.
    """
    print_section("EXAMPLE 6: COMPLETE ENSEMBLE COMPARISON")

    # Generate data
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X.shape[1]}")

    results = []

    # Base models for meta-ensembles
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=30, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=30, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]

    # 1. Bagging
    print_subsection("Training: Bagging")
    start = time.time()
    model = BaggingEnsemble(n_estimators=50, oob_score=True, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train, task='classification')
    acc = accuracy_score(y_test, model.predict(X_test))
    results.append({'Method': 'Bagging', 'Accuracy': acc, 'Time': time.time() - start})

    # 2. Random Forest
    print_subsection("Training: Random Forest")
    start = time.time()
    model = RandomForestEnsemble(n_estimators=50, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train, task='classification')
    acc = accuracy_score(y_test, model.predict(X_test))
    results.append({'Method': 'Random Forest', 'Accuracy': acc, 'Time': time.time() - start})

    # 3. AdaBoost
    print_subsection("Training: AdaBoost")
    start = time.time()
    model = AdaBoostEnsemble(n_estimators=50, random_state=42)
    model.fit(X_train, y_train, task='classification')
    acc = accuracy_score(y_test, model.predict(X_test))
    results.append({'Method': 'AdaBoost', 'Accuracy': acc, 'Time': time.time() - start})

    # 4. Gradient Boosting
    print_subsection("Training: Gradient Boosting")
    start = time.time()
    model = GradientBoostingEnsemble(n_estimators=50, random_state=42)
    model.fit(X_train, y_train, task='classification')
    acc = accuracy_score(y_test, model.predict(X_test))
    results.append({'Method': 'Gradient Boosting', 'Accuracy': acc, 'Time': time.time() - start})

    # 5. Voting (Soft)
    print_subsection("Training: Voting Ensemble")
    start = time.time()
    model = VotingEnsemble(estimators=base_models, voting='soft')
    model.fit(X_train, y_train, task='classification')
    acc = accuracy_score(y_test, model.predict(X_test))
    results.append({'Method': 'Voting', 'Accuracy': acc, 'Time': time.time() - start})

    # 6. Stacking
    print_subsection("Training: Stacking")
    start = time.time()
    model = StackingEnsemble(estimators=base_models, cv=5)
    model.fit(X_train, y_train, task='classification')
    acc = accuracy_score(y_test, model.predict(X_test))
    results.append({'Method': 'Stacking', 'Accuracy': acc, 'Time': time.time() - start})

    # 7. Blending
    print_subsection("Training: Blending")
    start = time.time()
    model = BlendingEnsemble(estimators=base_models, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, task='classification')
    acc = accuracy_score(y_test, model.predict(X_test))
    results.append({'Method': 'Blending', 'Accuracy': acc, 'Time': time.time() - start})

    # Summary
    print_subsection("Final Comparison")
    print(f"\n{'Method':<20} {'Accuracy':<12} {'Time (s)':<10}")
    print("-" * 80)
    for result in sorted(results, key=lambda x: x['Accuracy'], reverse=True):
        print(f"{result['Method']:<20} {result['Accuracy']:<12.4f} {result['Time']:<10.2f}")

    # Find best method
    best = max(results, key=lambda x: x['Accuracy'])
    fastest = min(results, key=lambda x: x['Time'])
    print(f"\nBest Accuracy: {best['Method']} ({best['Accuracy']:.4f})")
    print(f"Fastest Method: {fastest['Method']} ({fastest['Time']:.2f}s)")


if __name__ == "__main__":
    print("\n")
    print("*" * 80)
    print("COMPREHENSIVE ENSEMBLE METHODS EXAMPLES".center(80))
    print("*" * 80)

    # Run all examples
    example_1_bagging_comparison()
    example_2_boosting_comparison()
    example_3_stacking()
    example_4_voting()
    example_5_blending()
    example_6_complete_comparison()

    print("\n")
    print("*" * 80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!".center(80))
    print("*" * 80)
    print("\n")
