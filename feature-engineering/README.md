# Feature Engineering

Comprehensive feature engineering framework for machine learning with extraction, preprocessing, selection, and complete pipelines.

## Features

### Extraction Module (`extraction.py`)

#### Text Feature Extraction
- **TextFeatureExtractor**: Extract features from text
  - Text statistics (length, word count, sentence count)
  - Character-level features (uppercase, lowercase, digits, punctuation)
  - Special characters (hashtags, mentions, exclamations)
  - N-gram features
  - Readability metrics

#### Time Series Feature Extraction
- **TimeSeriesFeatureExtractor**: Extract features from time series
  - Statistical features (mean, std, min, max, percentiles)
  - Trend features (slope, intercept)
  - Variation features (coefficient of variation, changes)
  - Autocorrelation features
  - Peak/valley detection
  - Energy features

#### DateTime Feature Extraction
- **DateTimeFeatureExtractor**: Extract features from datetime
  - Temporal features (year, month, day, hour, minute)
  - Day of week, week of year, quarter
  - Boolean features (is_weekend, is_month_start/end)
  - Cyclical encoding (sin/cos for month, day_of_week, hour)

#### Structured Data Feature Extraction
- **StructuredFeatureExtractor**: Extract features from tabular data
  - Aggregation features (groupby operations)
  - Ratio features
  - Difference features
  - Count features for categorical variables
  - Null/missing value features

#### Feature Importance Analysis
- **FeatureImportanceAnalyzer**: Analyze feature importance
  - Random Forest importance
  - Permutation importance
  - Correlation with target
  - Mutual information
  - Visualization tools

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

### Pipeline Module (`pipeline.py`)

#### Complete Feature Engineering Pipeline
- **FeatureEngineeringPipeline**: End-to-end feature engineering
  - Automatic feature extraction (text, datetime, structured)
  - Preprocessing (scaling, encoding, missing values)
  - Feature generation (interactions, polynomials)
  - Feature selection
  - Dimensionality reduction
  - Save/load pipeline
  - Reproducible transformations

## Usage

### Feature Extraction

```python
from extraction import TextFeatureExtractor, TimeSeriesFeatureExtractor

# Text feature extraction
text_extractor = TextFeatureExtractor()
text_features = text_extractor.extract([
    "This is a sample text!",
    "Another example with #hashtags"
])

# Time series feature extraction
ts_extractor = TimeSeriesFeatureExtractor()
time_series = np.sin(np.linspace(0, 4*np.pi, 100))
ts_features = ts_extractor.extract(time_series)

# DateTime feature extraction
from extraction import DateTimeFeatureExtractor

datetime_extractor = DateTimeFeatureExtractor(cyclical_encoding=True)
dates = pd.date_range('2023-01-01', periods=100)
datetime_features = datetime_extractor.extract(pd.Series(dates))
```

### Feature Importance Analysis

```python
from extraction import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer()

# Random Forest importance
rf_importance = analyzer.analyze_random_forest(X, y)

# Permutation importance
perm_importance = analyzer.analyze_permutation(trained_model, X, y)

# Correlation analysis
corr_importance = analyzer.analyze_correlation(X, y)

# Mutual information
mi_importance = analyzer.analyze_mutual_info(X, y, task='classification')

# Visualize
analyzer.plot_importance(method='random_forest', top_k=20)

# Get top features
top_features = analyzer.get_top_features(method='random_forest', top_k=10)
```

### Complete Pipeline

```python
from pipeline import FeatureEngineeringPipeline

# Create pipeline with all steps
pipeline = FeatureEngineeringPipeline(
    # Extraction
    extract_datetime_features=True,
    datetime_columns=['timestamp', 'created_at'],
    extract_text_features=True,
    text_columns=['description', 'comments'],

    # Preprocessing
    scaling_method='standard',
    encoding_method='onehot',
    handle_missing='mean',

    # Feature generation
    create_interactions=True,
    interaction_degree=2,

    # Feature selection
    select_features=True,
    selection_method='mutual_info',
    n_features_to_select=50,

    # Dimensionality reduction
    reduce_dimensions=True,
    reduction_method='pca',
    n_components=20,

    verbose=True
)

# Fit on training data
X_train_transformed = pipeline.fit_transform(X_train, y_train)

# Transform test data
X_test_transformed = pipeline.transform(X_test)

# Get feature names
feature_names = pipeline.get_feature_names()

# Save pipeline for later use
pipeline.save('feature_pipeline.pkl')

# Load pipeline
loaded_pipeline = FeatureEngineeringPipeline.load('feature_pipeline.pkl')
```

## Requirements

- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0 (for time series features)
- matplotlib >= 3.7.0 (for plotting)
