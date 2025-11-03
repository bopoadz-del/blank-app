# Traditional Machine Learning Framework

Comprehensive implementation of traditional machine learning algorithms using Scikit-learn, XGBoost, and LightGBM.

## 📋 Features

### Classification Algorithms
- **Random Forest**: Ensemble of decision trees with bagging
- **XGBoost**: Gradient boosting with advanced regularization
- **LightGBM**: Fast gradient boosting with histogram-based learning
- **SVM**: Support Vector Machines with various kernels
- **KNN**: K-Nearest Neighbors with distance metrics
- **Decision Trees**: CART algorithm with pruning
- **Naive Bayes**: Gaussian, Multinomial, and Bernoulli variants
- **Logistic Regression**: Linear classification with regularization

### Regression Algorithms
- **Linear Regression**: Ordinary Least Squares with regularization
- **Ridge Regression**: L2 regularization
- **Lasso Regression**: L1 regularization
- **ElasticNet**: Combined L1 and L2 regularization
- **Random Forest Regressor**: Ensemble regression
- **XGBoost Regressor**: Gradient boosting for regression
- **LightGBM Regressor**: Fast gradient boosting
- **SVR**: Support Vector Regression

### Clustering Algorithms
- **K-Means**: Centroid-based clustering
- **DBSCAN**: Density-based clustering
- **Hierarchical Clustering**: Agglomerative clustering
- **Gaussian Mixture Models**: Probabilistic clustering
- **MeanShift**: Mode-seeking clustering
- **Spectral Clustering**: Graph-based clustering

### Dimensionality Reduction
- **PCA**: Principal Component Analysis
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding
- **UMAP**: Uniform Manifold Approximation and Projection
- **LDA**: Linear Discriminant Analysis
- **Truncated SVD**: Singular Value Decomposition
- **Kernel PCA**: Non-linear PCA

### Feature Selection
- **Univariate Selection**: Statistical tests (chi2, f_classif, mutual_info)
- **Recursive Feature Elimination**: RFE with cross-validation
- **Feature Importance**: Tree-based importance
- **L1-based Selection**: Lasso feature selection
- **Variance Threshold**: Remove low-variance features
- **Correlation Analysis**: Remove highly correlated features

### Model Utilities
- **Cross-Validation**: K-fold, stratified, time-series
- **Hyperparameter Tuning**: Grid search, random search, Bayesian optimization
- **Model Evaluation**: Comprehensive metrics for classification and regression
- **Pipeline Creation**: End-to-end ML pipelines
- **Model Persistence**: Save and load models

## 🚀 Quick Start

### Classification Example

```python
from classifiers.ensemble import RandomForestClassifier
from utils.evaluation import ModelEvaluator
from sklearn.datasets import make_classification

# Create dataset
X, y = make_classification(n_samples=1000, n_features=20)

# Train model
clf = RandomForestClassifier(n_estimators=100, max_depth=10)
clf.fit(X_train, y_train)

# Evaluate
evaluator = ModelEvaluator(clf, X_test, y_test)
metrics = evaluator.evaluate_classification()
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Regression Example

```python
from regressors.ensemble import XGBoostRegressor
from utils.evaluation import ModelEvaluator

# Train model
reg = XGBoostRegressor(n_estimators=100, learning_rate=0.1)
reg.fit(X_train, y_train)

# Evaluate
evaluator = ModelEvaluator(reg, X_test, y_test)
metrics = evaluator.evaluate_regression()
print(f"R² Score: {metrics['r2_score']:.4f}")
```

### Clustering Example

```python
from clustering.density import DBSCANClustering
from utils.visualization import plot_clusters

# Cluster data
clusterer = DBSCANClustering(eps=0.5, min_samples=5)
labels = clusterer.fit_predict(X)

# Visualize
plot_clusters(X, labels)
```

### Dimensionality Reduction Example

```python
from dimensionality_reduction.linear import PCAReducer
from dimensionality_reduction.nonlinear import TSNEReducer

# PCA
pca = PCAReducer(n_components=2)
X_pca = pca.fit_transform(X)

# t-SNE
tsne = TSNEReducer(n_components=2, perplexity=30)
X_tsne = tsne.fit_transform(X)
```

### Feature Selection Example

```python
from feature_selection.statistical import UnivariateSelector
from feature_selection.model_based import TreeBasedSelector

# Univariate selection
selector = UnivariateSelector(score_func='f_classif', k=10)
X_selected = selector.fit_transform(X, y)

# Tree-based selection
selector = TreeBasedSelector(estimator='random_forest', threshold='median')
X_selected = selector.fit_transform(X, y)
```

### Hyperparameter Tuning Example

```python
from utils.tuning import HyperparameterTuner
from classifiers.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

# Tune hyperparameters
tuner = HyperparameterTuner(
    RandomForestClassifier(),
    param_grid,
    method='grid_search',
    cv=5
)

best_model = tuner.fit(X_train, y_train)
print(f"Best parameters: {tuner.best_params}")
```

## 📁 Project Structure

```
traditional-ml/
├── classifiers/
│   ├── ensemble.py          # Random Forest, XGBoost, LightGBM
│   ├── linear.py            # Logistic Regression
│   ├── svm.py               # Support Vector Machines
│   ├── neighbors.py         # K-Nearest Neighbors
│   ├── tree.py              # Decision Trees
│   └── naive_bayes.py       # Naive Bayes variants
├── regressors/
│   ├── linear.py            # Linear, Ridge, Lasso, ElasticNet
│   ├── ensemble.py          # RF, XGBoost, LightGBM regressors
│   └── svm.py               # Support Vector Regression
├── clustering/
│   ├── centroid.py          # K-Means
│   ├── density.py           # DBSCAN
│   ├── hierarchical.py      # Agglomerative clustering
│   └── probabilistic.py     # Gaussian Mixture Models
├── dimensionality_reduction/
│   ├── linear.py            # PCA, LDA
│   ├── nonlinear.py         # t-SNE, UMAP
│   └── manifold.py          # Other manifold methods
├── feature_selection/
│   ├── statistical.py       # Univariate selection
│   ├── model_based.py       # RFE, feature importance
│   └── variance.py          # Variance threshold
├── utils/
│   ├── evaluation.py        # Model evaluation metrics
│   ├── tuning.py            # Hyperparameter tuning
│   ├── pipeline.py          # ML pipelines
│   ├── preprocessing.py     # Data preprocessing
│   └── visualization.py     # Plotting utilities
├── examples/
│   ├── classification_example.py
│   ├── regression_example.py
│   ├── clustering_example.py
│   └── feature_selection_example.py
└── README.md
```

## 📊 Model Comparison

### Classification Performance
| Model | Accuracy | Training Time | Interpretability |
|-------|----------|---------------|------------------|
| Random Forest | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| XGBoost | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| LightGBM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| SVM | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| KNN | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Decision Tree | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Naive Bayes | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Logistic Reg | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🔧 Installation

```bash
pip install scikit-learn xgboost lightgbm umap-learn
pip install numpy pandas matplotlib seaborn
```

## 📚 Documentation

Each module contains detailed documentation with:
- Algorithm description
- Parameter explanations
- Usage examples
- Best practices
- Performance considerations

## 🎯 Best Practices

1. **Always split your data**: Use train-test split or cross-validation
2. **Scale your features**: Most algorithms benefit from feature scaling
3. **Handle missing values**: Use imputation or remove missing data
4. **Tune hyperparameters**: Use grid search or random search
5. **Evaluate properly**: Use appropriate metrics for your problem
6. **Check for overfitting**: Monitor train vs. validation performance
7. **Use pipelines**: Combine preprocessing and modeling steps

## 🚀 Performance Tips

### Random Forest
- Increase `n_estimators` for better performance
- Use `max_features='sqrt'` for classification
- Set `n_jobs=-1` for parallel processing

### XGBoost
- Start with `learning_rate=0.1` and `n_estimators=100`
- Use early stopping to prevent overfitting
- Tune `max_depth` and `min_child_weight`

### LightGBM
- Use `num_leaves` instead of `max_depth`
- Enable `categorical_feature` for categorical data
- Set `bagging_fraction < 1` to prevent overfitting

### SVM
- Scale features before training
- Use RBF kernel for non-linear problems
- Tune `C` and `gamma` with grid search

## 📈 Metrics Reference

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity, true positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: True/false positives/negatives

### Regression Metrics
- **R² Score**: Coefficient of determination
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

### Clustering Metrics
- **Silhouette Score**: Cluster cohesion
- **Calinski-Harabasz Index**: Cluster separation
- **Davies-Bouldin Index**: Cluster compactness

## 📝 License

This project is provided as-is for educational and commercial use.
