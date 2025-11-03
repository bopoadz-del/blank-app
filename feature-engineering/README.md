# Feature Engineering

Comprehensive feature engineering utilities for machine learning.

## Features

### Preprocessing Module (`preprocessing.py`)

#### Feature Scaling
- **FeatureScaler**: Multiple scaling methods
  - StandardScaler: Z-score normalization
  - MinMaxScaler: Scale to range
  - RobustScaler: Robust to outliers
  - MaxAbsScaler: Scale by max absolute value
  - Normalizer: L1/L2 normalization

#### Feature Transformation
- **FeatureTransformer**: Advanced transformations
  - PowerTransformer: Box-Cox, Yeo-Johnson
  - QuantileTransformer: Uniform/normal distribution
  - Log transformation
  - Square root transformation

#### Categorical Encoding
- **CategoricalEncoder**: Encode categorical variables
  - Label encoding
  - One-hot encoding
  - Ordinal encoding
  - Target encoding

#### Binning
- **BinningTransformer**: Discretize continuous features
  - Uniform binning
  - Quantile binning
  - K-means binning

#### Feature Interactions
- **FeatureInteractionGenerator**: Create interactions
  - Polynomial features
  - Interaction terms
  - Ratio features
  - Difference features

### Selection Module (`selection.py`)

#### Feature Selection
- **FeatureSelector**: Multiple selection methods
  - Statistical: Chi2, ANOVA F-test, Mutual Information
  - Model-based: Lasso, Random Forest
  - Wrapper: Recursive Feature Elimination (RFE)
  - Sequential: Forward/backward selection
  - Filter: Variance threshold

#### Dimensionality Reduction
- **DimensionalityReducer**: Various reduction techniques
  - Linear: PCA, LDA, SVD, Factor Analysis
  - Non-linear: t-SNE, Isomap, LLE, MDS
  - Other: ICA, NMF

## Usage

### Feature Scaling

```python
from preprocessing import FeatureScaler

# Standard scaling
scaler = FeatureScaler(method='standard')
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MinMax scaling
scaler = FeatureScaler(method='minmax', feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_train)
```

### Categorical Encoding

```python
from preprocessing import CategoricalEncoder

# One-hot encoding
encoder = CategoricalEncoder(method='onehot', drop='first')
X_encoded = encoder.fit_transform(X_cat)

# Target encoding
encoder = CategoricalEncoder(method='target')
X_encoded = encoder.fit_transform(X_cat, y)
```

### Feature Interactions

```python
from preprocessing import FeatureInteractionGenerator

# Generate polynomial features
generator = FeatureInteractionGenerator(
    degree=2,
    include_ratios=True,
    include_differences=True
)
X_interactions = generator.fit_transform(X)

# Get feature names
feature_names = generator.get_feature_names(['x1', 'x2', 'x3'])
```

### Feature Selection

```python
from selection import FeatureSelector

# Mutual information selection
selector = FeatureSelector(method='mutual_info', k=10)
X_selected = selector.fit_transform(X, y)

# Get selected features
selected_features = selector.get_selected_features()

# Plot feature importance
selector.plot_feature_importance(feature_names, top_k=20)
```

### Dimensionality Reduction

```python
from selection import DimensionalityReducer

# PCA
reducer = DimensionalityReducer(method='pca', n_components=10)
X_reduced = reducer.fit_transform(X)

# Get explained variance
variance = reducer.get_explained_variance()
reducer.plot_explained_variance(cumulative=True)

# t-SNE for visualization
reducer = DimensionalityReducer(method='tsne', n_components=2)
X_2d = reducer.fit_transform(X)
reducer.plot_2d(X_2d, y=labels)
```

## Requirements

- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0 (for plotting)
