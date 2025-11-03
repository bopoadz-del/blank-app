# Model Optimization Quick Start Guide

Quick reference for all model optimization techniques.

## 📊 Hyperparameter Tuning

### Grid Search (Exhaustive)
```python
from hyperparameter_tuning import GridSearchTuner

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}

tuner = GridSearchTuner(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5
)

tuner.search(X, y)
print(tuner.best_params_)
print(tuner.best_score_)
```

### Random Search (Efficient)
```python
from hyperparameter_tuning import RandomSearchTuner
from scipy.stats import randint

param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20)
}

tuner = RandomSearchTuner(
    estimator=RandomForestClassifier(),
    param_distributions=param_distributions,
    n_iter=100,
    cv=5
)

tuner.search(X, y)
```

### Optuna (Bayesian Optimization)
```python
from hyperparameter_tuning import OptunaTuner

def objective(trial, X=None, y=None):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    scores = cross_val_score(model, X, y, cv=5)
    return scores.mean()

tuner = OptunaTuner(objective_fn=objective, n_trials=100, direction='maximize')
tuner.optimize(X=X, y=y)

# Visualize
tuner.plot_optimization_history()
tuner.plot_param_importances()
```

## ✂️ Model Pruning

### Basic Pruning (30% sparsity)
```python
from optimization import ModelPruner

pruner = ModelPruner(model, amount=0.3)
pruner.prune_global()

# Check results
size_info = pruner.get_model_size()
print(f"Sparsity: {size_info['sparsity'] * 100:.1f}%")

# Make permanent
pruner.make_permanent()
```

### Iterative Pruning with Fine-tuning
```python
def train_fn(model):
    # Your training code here
    for epoch in range(5):
        # ... training loop
        pass

pruner = ModelPruner(model, amount=0.7)
pruner.prune_iterative(
    train_fn=train_fn,
    num_iterations=5,
    initial_sparsity=0.2,
    final_sparsity=0.7
)
```

## 🔢 Model Quantization

### Dynamic Quantization (INT8)
```python
from optimization import ModelQuantizer

quantizer = ModelQuantizer(model)
quantized_model = quantizer.quantize_dynamic(dtype=torch.qint8)

# 4x smaller, 2-4x faster inference
```

### Static Quantization (Best Compression)
```python
quantized_model = quantizer.quantize_static(
    calibration_loader=calib_loader,
    backend='fbgemm'  # 'fbgemm' for x86, 'qnnpack' for ARM
)
```

### Quantization-Aware Training (Best Accuracy)
```python
def train_fn(model):
    # Your training code
    pass

quantized_model = quantizer.quantize_qat(train_fn, backend='fbgemm')
```

## 🎓 Knowledge Distillation

### Basic Distillation
```python
from optimization import KnowledgeDistiller

# Create teacher (large) and student (small) models
teacher = LargeModel()  # Pre-trained
student = SmallModel()

distiller = KnowledgeDistiller(
    teacher=teacher,
    student=student,
    temperature=4.0,
    alpha=0.7
)

optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

distiller.train(
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    epochs=100
)

# Student model is now trained to mimic teacher
# Typically 10x+ faster with only 2-5% accuracy loss
```

## 🤖 AutoML

### Complete AutoML Pipeline
```python
from automl import AutoMLPipeline

# Create pipeline
automl = AutoMLPipeline(
    task='classification',  # or 'regression'
    models=['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting'],
    tune_hyperparameters=True,
    create_ensemble=True,
    time_budget=3600  # 1 hour
)

# Fit and predict
automl.fit(X_train, y_train)
predictions = automl.predict(X_test)

# Get leaderboard
leaderboard = automl.get_leaderboard()
print(leaderboard)

# Save/load
automl.save('automl_pipeline.pkl')
loaded = AutoMLPipeline.load('automl_pipeline.pkl')
```

## 🎯 Cross-Validation

### K-Fold Cross-Validation
```python
from hyperparameter_tuning import CrossValidator

cv = CrossValidator(cv_type='kfold', n_splits=5)
scores = cv.evaluate(
    estimator=model,
    X=X,
    y=y,
    scoring=['accuracy', 'f1', 'precision', 'recall']
)
```

### Stratified K-Fold (Classification)
```python
cv = CrossValidator(cv_type='stratified', n_splits=5)
scores = cv.evaluate(estimator=model, X=X, y=y, scoring='accuracy')
```

### Time Series Split
```python
cv = CrossValidator(cv_type='timeseries', n_splits=5)
scores = cv.evaluate(estimator=model, X=X, y=y, scoring='neg_mean_squared_error')
```

## 📈 Performance Comparison

| Technique | Size Reduction | Speed Improvement | Accuracy Impact |
|-----------|----------------|-------------------|-----------------|
| **Pruning (70%)** | 70% | 1.5-2x | < 1% loss |
| **Dynamic Quantization** | 75% | 2-3x | < 1% loss |
| **Static Quantization** | 75% | 3-4x | 1-2% loss |
| **Quantization-Aware Training** | 75% | 3-4x | < 0.5% loss |
| **Knowledge Distillation** | 90%+ | 5-10x | 2-5% loss |
| **Pruning + Quantization** | 95% | 5-8x | 2-3% loss |
| **Full Compression** | 97% | 10-20x | 3-7% loss |

## 🔥 Best Practices

### When to Use Each Technique

**Grid Search:**
- Small parameter space (< 1000 combinations)
- Need to try every combination
- Understanding parameter interactions

**Random Search:**
- Large parameter space
- Continuous parameters
- Time-constrained

**Optuna (Bayesian):**
- Large parameter space
- Expensive model training
- Want best results with fewer trials
- Multi-objective optimization

**Pruning:**
- Model is too large for deployment
- Need faster inference
- Can afford fine-tuning time
- Target: Edge devices, mobile

**Quantization:**
- Model size is critical
- Inference speed is critical
- Integer operations preferred (INT8)
- Target: Mobile, embedded systems

**Knowledge Distillation:**
- Large accuracy gap acceptable
- Need very fast inference
- Can train small model from scratch
- Have pre-trained large model

**AutoML:**
- Don't know which algorithm works best
- Want to try many models quickly
- Need ensemble for best performance
- Exploratory phase

### Recommended Workflow

1. **Start with AutoML** to find best model family
2. **Hyperparameter Tune** with Optuna for best configuration
3. **Validate** with stratified k-fold cross-validation
4. **Compress** if needed:
   - Light compression: Pruning (30-50%)
   - Medium compression: Pruning + Quantization
   - Heavy compression: Knowledge Distillation
5. **Deploy** optimized model

## 🛠️ Common Patterns

### Pattern 1: Complete Optimization Pipeline
```python
# 1. AutoML for model selection
automl = AutoMLPipeline(task='classification')
automl.fit(X_train, y_train)
best_model = automl.get_best_model()

# 2. Hyperparameter tuning with Optuna
def objective(trial, X, y):
    # ... define search space
    return score

tuner = OptunaTuner(objective_fn=objective, n_trials=100)
tuner.optimize(X=X_train, y=y_train)

# 3. Train final model
final_model = train_with_best_params(tuner.best_params_)

# 4. Compress for deployment
pruner = ModelPruner(final_model, amount=0.5)
pruner.prune_global()

quantizer = ModelQuantizer(final_model)
optimized_model = quantizer.quantize_dynamic()
```

### Pattern 2: Neural Network Compression
```python
# Train large teacher model
teacher = train_large_model()

# Create small student
student = SmallModel()

# Knowledge distillation
distiller = KnowledgeDistiller(teacher, student, temperature=4.0)
distiller.train(train_loader, val_loader, optimizer, epochs=100)

# Further compress with pruning
pruner = ModelPruner(student, amount=0.3)
pruner.prune_global()

# Quantize
quantizer = ModelQuantizer(student)
final_model = quantizer.quantize_dynamic()

# Result: 95%+ compression, 10x+ speedup
```

## 📚 Additional Resources

- Run `examples.py` for complete working examples
- See `README.md` for detailed documentation
- Check individual module docstrings for API details

## 🚀 Quick Test

```python
# Test all optimization techniques
python examples.py

# Test individual components
python hyperparameter_tuning.py
python optimization.py
python automl.py
```
