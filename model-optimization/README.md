# Model Optimization

Comprehensive model optimization utilities including hyperparameter tuning, pruning, quantization, and AutoML.

## Features

### Hyperparameter Tuning (`hyperparameter_tuning.py`)

#### GridSearchTuner
- Exhaustive search through parameter grid
- Cross-validation
- Parallel processing
- Results visualization

#### RandomSearchTuner
- Random sampling from parameter distributions
- More efficient than grid search
- Good for large parameter spaces

#### OptunaTuner
- Bayesian optimization with Tree-structured Parzen Estimator (TPE)
- Intelligent parameter search
- Pruning (early stopping of bad trials)
- Multi-objective optimization
- Visualization tools

#### CrossValidator
- K-Fold cross-validation
- Stratified K-Fold
- Time Series Split
- Multiple metrics support

### Model Optimization (`optimization.py`)

#### ModelPruner
- **Magnitude pruning**: Remove smallest weights
- **Structured pruning**: Remove entire channels/neurons
- **Global pruning**: Prune across all layers
- **Iterative pruning**: Gradually increase sparsity
- Benefits: 30-90% model size reduction, faster inference

#### ModelQuantizer
- **Dynamic quantization**: INT8 weights, FP32 activations
- **Static quantization**: INT8 weights and activations
- **Quantization-aware training (QAT)**: Train with fake quantization
- Benefits: 4x smaller, 2-4x faster inference

#### KnowledgeDistiller
- Train small student model to mimic large teacher
- Soft target distillation
- Temperature scaling
- Benefits: Small model with large model performance

### AutoML (`automl.py`)

#### AutoMLPipeline
- Automated machine learning pipeline
- Automatic preprocessing
- Feature engineering
- Model selection
- Hyperparameter tuning
- Ensemble creation
- Leaderboard and comparison

## Usage

### Grid Search

```python
from hyperparameter_tuning import GridSearchTuner

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5]
}

tuner = GridSearchTuner(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5
)

tuner.search(X, y)
print(f"Best params: {tuner.best_params_}")
print(f"Best score: {tuner.best_score_}")
```

### Optuna Bayesian Optimization

```python
from hyperparameter_tuning import OptunaTuner

def objective(trial, X=None, y=None):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    scores = cross_val_score(model, X, y, cv=5)
    return scores.mean()

tuner = OptunaTuner(
    objective_fn=objective,
    n_trials=100,
    direction='maximize'
)

tuner.optimize(X=X, y=y)
tuner.plot_optimization_history()
tuner.plot_param_importances()
```

### Model Pruning

```python
from optimization import ModelPruner

# Create pruner
pruner = ModelPruner(model, amount=0.3)

# Apply pruning
pruner.prune_global()

# Fine-tune
# ... train model ...

# Make permanent
pruner.make_permanent()

# Check size
size_info = pruner.get_model_size()
print(f"Sparsity: {size_info['sparsity'] * 100:.2f}%")
```

### Model Quantization

```python
from optimization import ModelQuantizer

quantizer = ModelQuantizer(model)

# Dynamic quantization (easiest)
quantized_model = quantizer.quantize_dynamic()

# Static quantization (best compression)
quantized_model = quantizer.quantize_static(calibration_loader)

# Quantization-aware training (best accuracy)
quantized_model = quantizer.quantize_qat(train_fn)
```

### Knowledge Distillation

```python
from optimization import KnowledgeDistiller

# Create distiller
distiller = KnowledgeDistiller(
    teacher=large_model,
    student=small_model,
    temperature=4.0,
    alpha=0.7
)

# Train student
optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
distiller.train(
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    epochs=100
)
```

### AutoML Pipeline

```python
from automl import AutoMLPipeline

# Create AutoML pipeline
automl = AutoMLPipeline(
    task='classification',
    time_budget=3600,  # 1 hour
    create_ensemble=True
)

# Fit on data
automl.fit(X_train, y_train)

# Get leaderboard
leaderboard = automl.get_leaderboard()
print(leaderboard)

# Make predictions
predictions = automl.predict(X_test)

# Save pipeline
automl.save('automl_pipeline.pkl')
```

## Requirements

- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- torch >= 2.0.0
- optuna >= 3.0.0 (optional, for Bayesian optimization)
- xgboost >= 1.7.0 (optional, for AutoML)
- lightgbm >= 3.3.0 (optional, for AutoML)

## Model Compression Results

Typical results from optimization:

| Method | Size Reduction | Speed Improvement | Accuracy Loss |
|--------|----------------|-------------------|---------------|
| Pruning (70%) | 70% | 1.5-2x | < 1% |
| Quantization (INT8) | 75% | 2-4x | < 1% |
| Distillation | 90% | 5-10x | 2-5% |
| Combined | 95% | 10-20x | 3-7% |
