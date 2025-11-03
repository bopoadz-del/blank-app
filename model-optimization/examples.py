"""
Complete Model Optimization Examples
Practical demonstrations of all optimization techniques
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Hyperparameter Tuning with GridSearch, RandomSearch, and Optuna
# ============================================================================

def example_hyperparameter_tuning():
    """Demonstrate all hyperparameter tuning methods"""
    print("\n" + "=" * 70)
    print("Example 1: Hyperparameter Tuning")
    print("=" * 70)

    # Load dataset
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

    # 1. Grid Search
    print("\n1.1 Grid Search")
    print("-" * 70)

    from hyperparameter_tuning import GridSearchTuner

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }

    grid_tuner = GridSearchTuner(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        verbose=0
    )

    grid_tuner.search(X_train, y_train)

    print(f"Best parameters: {grid_tuner.best_params_}")
    print(f"Best CV score: {grid_tuner.best_score_:.4f}")

    # Test on holdout set
    y_pred = grid_tuner.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_acc:.4f}")

    # 2. Random Search
    print("\n1.2 Random Search")
    print("-" * 70)

    from hyperparameter_tuning import RandomSearchTuner
    from scipy.stats import randint, uniform

    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 20),
        'max_features': uniform(0.5, 0.5)  # 0.5 to 1.0
    }

    random_tuner = RandomSearchTuner(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=50,
        scoring='accuracy',
        cv=5,
        verbose=0
    )

    random_tuner.search(X_train, y_train)

    print(f"Best parameters: {random_tuner.best_params_}")
    print(f"Best CV score: {random_tuner.best_score_:.4f}")

    # 3. Optuna (Bayesian Optimization)
    print("\n1.3 Optuna Bayesian Optimization")
    print("-" * 70)

    try:
        from hyperparameter_tuning import OptunaTuner
        from sklearn.model_selection import cross_val_score
        import optuna

        def objective(trial, X=None, y=None):
            # Define search space
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            max_features = trial.suggest_float('max_features', 0.5, 1.0)

            # Create and evaluate model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features=max_features,
                random_state=42,
                n_jobs=-1
            )

            # Cross-validation
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
            return scores.mean()

        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        optuna_tuner = OptunaTuner(
            objective_fn=objective,
            n_trials=50,
            direction='maximize',
            sampler='tpe'
        )

        optuna_tuner.optimize(X=X_train, y=y_train)

        print(f"Best parameters: {optuna_tuner.best_params_}")
        print(f"Best score: {optuna_tuner.best_score_:.4f}")

        # Train final model with best params
        best_model = RandomForestClassifier(**optuna_tuner.best_params_, random_state=42)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {test_acc:.4f}")

    except ImportError:
        print("Optuna not available. Install with: pip install optuna")


# ============================================================================
# Example 2: Cross-Validation
# ============================================================================

def example_cross_validation():
    """Demonstrate different cross-validation strategies"""
    print("\n" + "=" * 70)
    print("Example 2: Cross-Validation")
    print("=" * 70)

    from hyperparameter_tuning import CrossValidator

    # Load dataset
    X, y = load_breast_cancer(return_X_y=True)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 1. K-Fold CV
    print("\n2.1 K-Fold Cross-Validation")
    print("-" * 70)

    cv_kfold = CrossValidator(cv_type='kfold', n_splits=5, shuffle=True)
    scores = cv_kfold.evaluate(
        estimator=model,
        X=X,
        y=y,
        scoring=['accuracy', 'f1', 'precision', 'recall'],
        verbose=0
    )

    # 2. Stratified K-Fold CV
    print("\n2.2 Stratified K-Fold Cross-Validation")
    print("-" * 70)

    cv_stratified = CrossValidator(cv_type='stratified', n_splits=5, shuffle=True)
    scores = cv_stratified.evaluate(
        estimator=model,
        X=X,
        y=y,
        scoring=['accuracy', 'roc_auc'],
        verbose=0
    )


# ============================================================================
# Example 3: Model Pruning
# ============================================================================

def example_model_pruning():
    """Demonstrate neural network pruning"""
    print("\n" + "=" * 70)
    print("Example 3: Model Pruning")
    print("=" * 70)

    from optimization import ModelPruner

    # Create a simple neural network
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(30, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 2)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = SimpleNet()

    print("Original Model:")
    size_info = ModelPruner(model, amount=0.0).get_model_size()
    print(f"  Total parameters: {size_info['total_params']:,}")
    print(f"  Size: {size_info['size_mb']:.2f} MB")

    # 1. Magnitude Pruning
    print("\n3.1 Magnitude Pruning (30% sparsity)")
    print("-" * 70)

    pruner = ModelPruner(model, amount=0.3, structured=False)
    pruner.prune_magnitude()

    size_info = pruner.get_model_size()
    print(f"  Sparsity: {size_info['sparsity'] * 100:.2f}%")
    print(f"  Size: {size_info['size_mb']:.2f} MB")

    # 2. Global Pruning
    print("\n3.2 Global Pruning (50% sparsity)")
    print("-" * 70)

    model2 = SimpleNet()
    pruner2 = ModelPruner(model2, amount=0.5)
    pruner2.prune_global()

    size_info = pruner2.get_model_size()
    print(f"  Sparsity: {size_info['sparsity'] * 100:.2f}%")
    print(f"  Size: {size_info['size_mb']:.2f} MB")

    # Make pruning permanent
    print("\n3.3 Making Pruning Permanent")
    print("-" * 70)
    pruner2.make_permanent()
    print("  Pruning masks removed, sparsity is now permanent")


# ============================================================================
# Example 4: Model Quantization
# ============================================================================

def example_model_quantization():
    """Demonstrate model quantization"""
    print("\n" + "=" * 70)
    print("Example 4: Model Quantization")
    print("=" * 70)

    from optimization import ModelQuantizer

    # Create a simple model
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x.view(-1, 784)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = SimpleNet()

    # 1. Dynamic Quantization
    print("\n4.1 Dynamic Quantization (INT8)")
    print("-" * 70)

    quantizer = ModelQuantizer(model)
    quantized_model = quantizer.quantize_dynamic(dtype=torch.qint8)

    print("Quantization complete!")

    # 2. Compare model sizes
    print("\n4.2 Model Size Comparison")
    print("-" * 70)

    def get_model_size(model):
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    orig_size = get_model_size(model)
    quant_size = get_model_size(quantized_model)

    print(f"  Original model: {orig_size:.2f} MB")
    print(f"  Quantized model: {quant_size:.2f} MB")
    print(f"  Compression ratio: {orig_size / quant_size:.2f}x")
    print(f"  Size reduction: {(1 - quant_size / orig_size) * 100:.1f}%")


# ============================================================================
# Example 5: Knowledge Distillation
# ============================================================================

def example_knowledge_distillation():
    """Demonstrate knowledge distillation"""
    print("\n" + "=" * 70)
    print("Example 5: Knowledge Distillation")
    print("=" * 70)

    from optimization import KnowledgeDistiller

    # Create synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                               n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Define teacher (large) and student (small) models
    class TeacherNet(nn.Module):
        def __init__(self):
            super(TeacherNet, self).__init__()
            self.fc1 = nn.Linear(20, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, 3)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    class StudentNet(nn.Module):
        def __init__(self):
            super(StudentNet, self).__init__()
            self.fc1 = nn.Linear(20, 32)
            self.fc2 = nn.Linear(32, 3)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    print("\n5.1 Model Comparison")
    print("-" * 70)

    teacher = TeacherNet()
    student = StudentNet()

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())

    print(f"  Teacher parameters: {teacher_params:,}")
    print(f"  Student parameters: {student_params:,}")
    print(f"  Compression ratio: {teacher_params / student_params:.2f}x")

    # Train teacher first (simplified for demonstration)
    print("\n5.2 Training Teacher Model")
    print("-" * 70)

    criterion = nn.CrossEntropyLoss()
    teacher_optimizer = optim.Adam(teacher.parameters(), lr=1e-3)

    teacher.train()
    for epoch in range(5):
        epoch_loss = 0
        for inputs, labels in train_loader:
            teacher_optimizer.zero_grad()
            outputs = teacher(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            teacher_optimizer.step()
            epoch_loss += loss.item()

        print(f"  Epoch {epoch + 1}/5 - Loss: {epoch_loss / len(train_loader):.4f}")

    # Evaluate teacher
    teacher.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = teacher(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    teacher_accuracy = 100 * correct / total
    print(f"  Teacher test accuracy: {teacher_accuracy:.2f}%")

    # Knowledge distillation
    print("\n5.3 Knowledge Distillation (Training Student)")
    print("-" * 70)

    distiller = KnowledgeDistiller(
        teacher=teacher,
        student=student,
        temperature=4.0,
        alpha=0.7,
        device='cpu'
    )

    student_optimizer = optim.Adam(student.parameters(), lr=1e-3)

    distiller.train(
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=student_optimizer,
        epochs=10,
        scheduler=None
    )

    print("\nKnowledge distillation complete!")


# ============================================================================
# Example 6: AutoML Pipeline
# ============================================================================

def example_automl():
    """Demonstrate AutoML pipeline"""
    print("\n" + "=" * 70)
    print("Example 6: AutoML Pipeline")
    print("=" * 70)

    from automl import AutoMLPipeline

    # Load dataset
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

    # 1. Classification AutoML
    print("\n6.1 Classification AutoML")
    print("-" * 70)

    automl = AutoMLPipeline(
        task='classification',
        models=['logistic_regression', 'random_forest', 'gradient_boosting', 'svm'],
        tune_hyperparameters=False,  # Disable for speed
        create_ensemble=True,
        n_jobs=-1
    )

    # Fit AutoML
    automl.fit(X_train, y_train)

    # Get leaderboard
    print("\nLeaderboard:")
    leaderboard = automl.get_leaderboard()
    print(leaderboard[['Model', 'Val Score', 'Train Score', 'Time (s)']])

    # Make predictions
    y_pred = automl.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Get best model
    best_model = automl.get_best_model()
    print(f"Best model: {type(best_model).__name__}")

    # Save pipeline
    print("\n6.2 Saving AutoML Pipeline")
    print("-" * 70)
    automl.save('automl_pipeline.pkl')
    print("Pipeline saved to: automl_pipeline.pkl")

    # Load pipeline
    loaded_automl = AutoMLPipeline.load('automl_pipeline.pkl')
    y_pred_loaded = loaded_automl.predict(X_test)
    print(f"Loaded pipeline test accuracy: {accuracy_score(y_test, y_pred_loaded):.4f}")


# ============================================================================
# Main: Run All Examples
# ============================================================================

def main():
    """Run all optimization examples"""
    print("\n" + "=" * 80)
    print("COMPLETE MODEL OPTIMIZATION EXAMPLES")
    print("=" * 80)

    # Run examples
    example_hyperparameter_tuning()
    example_cross_validation()
    example_model_pruning()
    example_model_quantization()
    example_knowledge_distillation()
    example_automl()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    print("\n" + "Summary of Optimization Techniques:")
    print("-" * 80)
    print("✓ Hyperparameter Tuning: GridSearch, RandomSearch, Optuna")
    print("✓ Cross-Validation: K-Fold, Stratified K-Fold, Time Series")
    print("✓ Model Pruning: Magnitude, Structured, Global (30-90% compression)")
    print("✓ Model Quantization: Dynamic, Static, QAT (4x compression)")
    print("✓ Knowledge Distillation: Teacher-Student training (10x+ speedup)")
    print("✓ AutoML: Automated model selection and ensemble")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
