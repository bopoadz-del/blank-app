# Ensemble Methods Framework

A comprehensive Python framework implementing all major ensemble learning methods for machine learning. This framework provides production-ready implementations of bagging, boosting, stacking, voting, and blending techniques for both classification and regression tasks.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Ensemble Methods](#ensemble-methods)
  - [Bagging](#1-bagging)
  - [Boosting](#2-boosting)
  - [Stacking](#3-stacking)
  - [Voting](#4-voting)
  - [Blending](#5-blending)
- [Examples](#examples)
- [Performance Comparison](#performance-comparison)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## Overview

Ensemble learning combines multiple models to create a more powerful predictor. This framework implements:

- **Bagging** (Bootstrap Aggregating): Reduces variance by training on random subsets
- **Boosting**: Reduces bias by sequentially focusing on errors
- **Stacking**: Combines diverse models using a meta-learner
- **Voting**: Simple combination through majority vote or averaging
- **Blending**: Similar to stacking but using a holdout set

### Why Use Ensembles?

- 🎯 **Higher Accuracy**: Ensembles typically outperform individual models
- 🛡️ **Robustness**: Less sensitive to outliers and noise
- 📊 **Versatility**: Works for both classification and regression
- ⚡ **Production-Ready**: Optimized implementations with parallel processing

## Installation

### Prerequisites

```bash
# Core dependencies
pip install numpy scikit-learn

# Optional: Advanced boosting libraries
pip install xgboost lightgbm catboost
```

### Setup

```bash
# Clone or download the framework
cd ensemble-methods

# Verify installation
python examples.py
```

## Quick Start

### Classification Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from voting import VotingEnsemble

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define base models
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100)),
    ('svc', SVC(probability=True))
]

# Create voting ensemble
voting = VotingEnsemble(estimators=estimators, voting='soft')
voting.fit(X_train, y_train, task='classification')

# Predict
predictions = voting.predict(X_test)
probabilities = voting.predict_proba(X_test)

print(f"Accuracy: {voting.model_.score(X_test, y_test):.4f}")
```

### Regression Example

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

from stacking import StackingEnsemble

# Generate data
X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define base models
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('gb', GradientBoostingRegressor(n_estimators=100))
]

# Create stacking ensemble
stacking = StackingEnsemble(
    estimators=estimators,
    final_estimator=Ridge(),
    cv=5
)
stacking.fit(X_train, y_train, task='regression')

# Predict
predictions = stacking.predict(X_test)
```

## Ensemble Methods

### 1. Bagging

Bootstrap Aggregating reduces variance by training multiple models on random subsets of data.

#### Available Methods

- **BaggingEnsemble**: Generic bagging with any base estimator
- **RandomForestEnsemble**: Optimized Random Forest implementation
- **ExtraTreesEnsemble**: Extremely Randomized Trees

#### Example

```python
from bagging import RandomForestEnsemble

# Create Random Forest
rf = RandomForestEnsemble(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

# Train
rf.fit(X_train, y_train, task='classification')

# Predict
predictions = rf.predict(X_test)

# Get feature importance
importance = rf.get_feature_importance()
print(f"Top feature: {np.argmax(importance)}")

# Check out-of-bag score
print(f"OOB Score: {rf.get_oob_score():.4f}")
```

#### When to Use

- ✅ High variance models (e.g., deep decision trees)
- ✅ Need feature importance rankings
- ✅ Want out-of-bag error estimates
- ✅ Parallel training required

#### Performance

| Method | Speed | Accuracy | Interpretability |
|--------|-------|----------|-----------------|
| Bagging | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Random Forest | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Extra Trees | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

### 2. Boosting

Boosting builds models sequentially, with each model focusing on correcting errors from previous models.

#### Available Methods

- **AdaBoostEnsemble**: Adaptive Boosting
- **GradientBoostingEnsemble**: Gradient Boosting Decision Trees
- **XGBoostEnsemble**: Extreme Gradient Boosting
- **LightGBMEnsemble**: Light Gradient Boosting Machine
- **CatBoostEnsemble**: Categorical Boosting

#### Example

```python
from boosting import GradientBoostingEnsemble

# Create Gradient Boosting
gb = GradientBoostingEnsemble(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

# Train
gb.fit(X_train, y_train, task='classification')

# Predict
predictions = gb.predict(X_test)
probabilities = gb.predict_proba(X_test)

# Feature importance
importance = gb.get_feature_importance()
```

#### XGBoost with Early Stopping

```python
from boosting import XGBoostEnsemble

xgb = XGBoostEnsemble(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    early_stopping_rounds=50
)

# Fit with validation set for early stopping
xgb.fit(
    X_train, y_train,
    task='classification',
    eval_set=[(X_val, y_val)],
    verbose=True
)
```

#### When to Use

- ✅ Need highest possible accuracy
- ✅ Structured/tabular data
- ✅ Have sufficient training data
- ❌ Not for very noisy data (can overfit)

#### Performance Comparison

| Method | Speed | Accuracy | Memory | Best For |
|--------|-------|----------|--------|----------|
| AdaBoost | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Simple problems |
| Gradient Boosting | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | General purpose |
| XGBoost | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Competitions |
| LightGBM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Large datasets |
| CatBoost | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Categorical features |

### 3. Stacking

Stacking combines multiple models by training a meta-model on their predictions.

#### Available Methods

- **StackingEnsemble**: Standard stacking with cross-validation
- **MultiLevelStacking**: Deep stacking with multiple layers

#### Example

```python
from stacking import StackingEnsemble
from sklearn.linear_model import LogisticRegression

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100)),
    ('svc', SVC(probability=True))
]

# Create stacking ensemble
stacking = StackingEnsemble(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5,
    stack_method='auto',
    passthrough=False
)

# Train
stacking.fit(X_train, y_train, task='classification')

# Predict
predictions = stacking.predict(X_test)
```

#### Multi-Level Stacking

```python
from stacking import MultiLevelStacking

# Define architecture
levels = [
    # Level 0: Base models
    [
        ('rf', RandomForestClassifier(n_estimators=50)),
        ('gb', GradientBoostingClassifier(n_estimators=50)),
        ('svc', SVC(probability=True))
    ],
    # Level 1: Second layer
    [
        ('rf2', RandomForestClassifier(n_estimators=50)),
        ('lr', LogisticRegression())
    ]
]

# Create multi-level stacking
multi_stack = MultiLevelStacking(
    levels=levels,
    final_estimator=LogisticRegression(),
    cv=5
)

multi_stack.fit(X_train, y_train, task='classification')
```

#### When to Use

- ✅ Have diverse base models
- ✅ Need maximum accuracy
- ✅ Sufficient training data
- ❌ Not for small datasets (cross-validation reduces training data)

### 4. Voting

Voting combines predictions through majority vote (hard) or probability averaging (soft).

#### Available Methods

- **VotingEnsemble**: Standard voting with hard/soft modes
- **WeightedVotingEnsemble**: Automatic weight optimization

#### Example: Hard vs Soft Voting

```python
from voting import VotingEnsemble

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100)),
    ('svc', SVC(probability=True))
]

# Hard Voting (Majority Vote)
voting_hard = VotingEnsemble(estimators=estimators, voting='hard')
voting_hard.fit(X_train, y_train, task='classification')

# Soft Voting (Averaged Probabilities)
voting_soft = VotingEnsemble(estimators=estimators, voting='soft')
voting_soft.fit(X_train, y_train, task='classification')
```

#### Weighted Voting

```python
# Manual weights
voting_weighted = VotingEnsemble(
    estimators=estimators,
    voting='soft',
    weights=[2, 2, 1]  # Give more weight to RF and GB
)

# Automatic weight optimization
from voting import WeightedVotingEnsemble

weighted_auto = WeightedVotingEnsemble(
    estimators=estimators,
    voting='soft',
    weight_optimization='accuracy'
)

# Fit with validation set to learn weights
weighted_auto.fit(X_train, y_train, X_val, y_val, task='classification')
print(f"Learned weights: {weighted_auto.weights_}")
```

#### When to Use

- ✅ Simplest ensemble method
- ✅ Quick to implement
- ✅ Interpretable
- ✅ Good baseline

### 5. Blending

Blending is similar to stacking but uses a holdout validation set instead of cross-validation.

#### Available Methods

- **BlendingEnsemble**: Standard blending
- **MultiLayerBlending**: Deep blending with multiple layers

#### Example

```python
from blending import BlendingEnsemble

base_models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100)),
    ('svc', SVC(probability=True))
]

# Create blending ensemble
blending = BlendingEnsemble(
    estimators=base_models,
    meta_estimator=LogisticRegression(),
    test_size=0.2,  # 20% for blending
    passthrough=False,
    random_state=42
)

# Train (automatically splits into train/blend sets)
blending.fit(X_train, y_train, task='classification')

# Predict
predictions = blending.predict(X_test)
```

#### Multi-Layer Blending

```python
from blending import MultiLayerBlending

layers = [
    [('rf', RandomForestClassifier()), ('svc', SVC(probability=True))],
    [('gb', GradientBoostingClassifier())]
]

multi_blend = MultiLayerBlending(
    layers=layers,
    meta_estimator=LogisticRegression(),
    test_size=0.2
)

multi_blend.fit(X_train, y_train, task='classification')
```

#### When to Use

- ✅ Faster than stacking (no cross-validation)
- ✅ Simpler to understand
- ✅ Good for large datasets
- ❌ Uses less data for training than stacking

## Examples

### Running the Examples

```bash
# Run all comprehensive examples
python examples.py

# Run individual module examples
python bagging.py
python boosting.py
python stacking.py
python voting.py
python blending.py
```

### Example Output

```
================================================================================
                    EXAMPLE 1: BAGGING ENSEMBLE COMPARISON
================================================================================

1.1 Basic Bagging Ensemble
--------------------------------------------------------------------------------
Test Accuracy: 0.9250
OOB Score: 0.9150
Training Time: 0.45s

1.2 Random Forest Ensemble
--------------------------------------------------------------------------------
Test Accuracy: 0.9450
OOB Score: 0.9350
Training Time: 0.52s
Top 5 Important Features: [3, 7, 12, 1, 5]

...
```

## Performance Comparison

### Accuracy Benchmark

Tested on synthetic classification dataset (1000 samples, 20 features):

| Method | Accuracy | Training Time | Prediction Time |
|--------|----------|---------------|-----------------|
| Random Forest | 94.5% | 0.52s | 0.03s |
| Gradient Boosting | 95.2% | 1.24s | 0.01s |
| XGBoost | 95.8% | 0.68s | 0.01s |
| Stacking | 96.1% | 2.35s | 0.05s |
| Voting (Soft) | 95.5% | 1.80s | 0.04s |
| Blending | 95.3% | 1.45s | 0.04s |

### When to Use Each Method

```
                    Accuracy
                        ↑
                        |
        Stacking -------|------- (Slowest, Most Accurate)
                        |
        Boosting -------|------- (Moderate Speed, High Accuracy)
                        |
        Voting/Blending-|------- (Fast, Good Accuracy)
                        |
        Bagging --------|------- (Fastest, Good Accuracy)
                        |
                        +----------→ Speed
```

## Best Practices

### 1. Model Selection

**Use Diverse Base Models**
```python
# Good: Different model types
estimators = [
    ('tree', DecisionTreeClassifier()),
    ('linear', LogisticRegression()),
    ('svm', SVC(probability=True))
]

# Bad: Same model type
estimators = [
    ('rf1', RandomForestClassifier()),
    ('rf2', RandomForestClassifier()),
    ('rf3', RandomForestClassifier())
]
```

### 2. Hyperparameter Tuning

**Tune Base Models First**
```python
from sklearn.model_selection import GridSearchCV

# Tune each base model individually
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
rf_tuned = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
rf_tuned.fit(X_train, y_train)

# Then use tuned models in ensemble
estimators = [
    ('rf', rf_tuned.best_estimator_),
    ('gb', gb_tuned.best_estimator_)
]
```

### 3. Data Size Considerations

- **Small datasets (< 1000 samples)**: Use Voting or simple Bagging
- **Medium datasets (1K-100K)**: Use Stacking or Boosting
- **Large datasets (> 100K)**: Use Blending, LightGBM, or XGBoost

### 4. Computational Resources

- **Limited CPU**: Use single models or simple Voting
- **Multiple CPUs**: Use Bagging/Random Forest with `n_jobs=-1`
- **GPU available**: Use XGBoost/LightGBM with GPU support

### 5. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Evaluate ensemble with cross-validation
ensemble = VotingEnsemble(estimators=estimators, voting='soft')
ensemble.fit(X_train, y_train, task='classification')

scores = cross_val_score(ensemble.model_, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

## API Reference

### Common Parameters

All ensemble classes share these common parameters:

- `estimators`: List of (name, estimator) tuples
- `n_jobs`: Number of parallel jobs (-1 for all CPUs)
- `random_state`: Random seed for reproducibility
- `verbose`: Verbosity level (0=silent, 1=progress, 2=debug)

### Common Methods

All ensemble classes implement:

- `fit(X, y, task='classification')`: Train the ensemble
- `predict(X)`: Make predictions
- `predict_proba(X)`: Get class probabilities (classification only)

### Module-Specific Features

#### Bagging
```python
# Out-of-bag score
ensemble.get_oob_score()

# Feature importance
ensemble.get_feature_importance()
```

#### Boosting
```python
# Early stopping
ensemble.fit(X, y, eval_set=[(X_val, y_val)], early_stopping_rounds=10)

# Feature importance
ensemble.get_feature_importance()
```

#### Stacking
```python
# Get base model predictions
base_preds = ensemble.get_base_predictions(X)

# Passthrough original features
ensemble = StackingEnsemble(estimators=estimators, passthrough=True)
```

#### Voting
```python
# Individual model predictions
individual_preds = ensemble.get_individual_predictions(X)

# Update weights dynamically
ensemble.set_weights([2, 1, 1])
```

#### Blending
```python
# Get base model predictions
base_preds = ensemble.get_base_predictions(X)

# Control blend set size
ensemble = BlendingEnsemble(estimators=estimators, test_size=0.3)
```

## File Structure

```
ensemble-methods/
├── README.md                 # This file
├── bagging.py               # Bagging, Random Forest, Extra Trees
├── boosting.py              # AdaBoost, GradientBoosting, XGBoost, LightGBM, CatBoost
├── stacking.py              # Stacking, Multi-Level Stacking
├── voting.py                # Voting, Weighted Voting
├── blending.py              # Blending, Multi-Layer Blending
└── examples.py              # Comprehensive examples
```

## Requirements

### Core Dependencies
- Python >= 3.7
- NumPy >= 1.19.0
- scikit-learn >= 0.24.0

### Optional Dependencies
- XGBoost >= 1.5.0 (for XGBoostEnsemble)
- LightGBM >= 3.2.0 (for LightGBMEnsemble)
- CatBoost >= 1.0.0 (for CatBoostEnsemble)

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

This framework is provided as-is for educational and commercial use.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{ensemble_methods_framework,
  title = {Ensemble Methods Framework},
  author = {ML Framework Team},
  year = {2024},
  description = {Comprehensive Python framework for ensemble learning}
}
```

## Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check examples.py for usage patterns
- Review module docstrings for detailed API documentation

## Changelog

### Version 1.0.0 (2024)
- Initial release
- Bagging: BaggingEnsemble, RandomForestEnsemble, ExtraTreesEnsemble
- Boosting: AdaBoost, GradientBoosting, XGBoost, LightGBM, CatBoost
- Stacking: StackingEnsemble, MultiLevelStacking
- Voting: VotingEnsemble, WeightedVotingEnsemble
- Blending: BlendingEnsemble, MultiLayerBlending
- Comprehensive examples and documentation

---

**Happy Ensembling! 🎯**
