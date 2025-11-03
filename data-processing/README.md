# Data Processing Framework

Comprehensive data processing utilities for machine learning workflows including augmentation, imbalanced data handling, imputation, outlier detection, validation, and splitting.

## Overview

This framework provides production-ready tools for:

### Data Augmentation
- **Image**: rotation, flipping, cropping, color jittering, noise, cutout, mixup
- **Text**: synonym replacement, random insertion/swap/deletion, back-translation
- **Tabular**: noise injection, mixup, SMOTE-like interpolation, feature perturbation

### Imbalanced Data Handling
- **Oversampling**: Random oversampling, SMOTE, ADASYN
- **Undersampling**: Random undersampling, Tomek links, NearMiss
- **Combined**: SMOTE+Tomek
- **Weighting**: Class and sample weight computation

### Missing Value Imputation
- **Simple**: Mean, median, mode, constant
- **KNN**: Distance-based imputation
- **Iterative**: MICE (Multiple Imputation by Chained Equations)
- **Time Series**: Forward fill, backward fill, linear/spline interpolation
- **Indicators**: Missing value indicator features

### Outlier Detection
- **Statistical**: IQR, Z-score, Modified Z-score (MAD)
- **Distance-based**: KNN, Local Outlier Factor (LOF)
- **Isolation-based**: Isolation Forest
- **Multivariate**: Mahalanobis distance

### Data Validation
- **Schema validation** with type and range checking
- **Consistency checks**: duplicates, constant features, correlations
- **Data quality metrics**: completeness, uniqueness, validity
- **Validation reports**

### Data Splitting
- **Train/test/validation splits** with stratification
- **Cross-validation**: K-Fold, Stratified K-Fold, Group K-Fold
- **Time series splits** with gap support

## Quick Start

```python
import numpy as np
from data_processing import *

# Sample data
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, 1000)

# 1. Augmentation
augmenter = TabularAugmenter()
X_aug, y_aug = augmenter.augment_batch(X, y, augmentation_factor=2)

# 2. Handle imbalanced data
smote = SMOTE(k_neighbors=5)
X_balanced, y_balanced = smote.fit_resample(X, y)

# 3. Imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 4. Outlier detection
detector = IsolationForest(contamination=0.1)
outliers = detector.fit_predict(X)

# 5. Validation
validator = DataValidator(schema={...})
is_valid, errors = validator.validate(X)

# 6. Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
```

## Modules

### augmentation.py
```python
from data_processing.augmentation import ImageAugmenter, TextAugmenter, TabularAugmenter

# Image augmentation
img_aug = ImageAugmenter()
rotated = img_aug.rotate(image, angle=30)
augmented = img_aug.random_augment(image, n_ops=3)

# Text augmentation
text_aug = TextAugmenter()
augmented_text = text_aug.synonym_replacement(text, n_replacements=2)

# Tabular augmentation
tab_aug = TabularAugmenter()
X_aug, y_aug = tab_aug.mixup(X, y, alpha=0.2)
```

### imbalanced.py
```python
from data_processing.imbalanced import SMOTE, RandomUnderSampler, SMOTETomek

# SMOTE oversampling
smote = SMOTE(k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Random undersampling
rus = RandomUnderSampler()
X_under, y_under = rus.fit_resample(X, y)

# Combined approach
smote_tomek = SMOTETomek()
X_clean, y_clean = smote_tomek.fit_resample(X, y)

# Class weights
weights = compute_class_weights(y, mode='balanced')
```

### imputation.py
```python
from data_processing.imputation import SimpleImputer, KNNImputer, IterativeImputer

# Simple imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_knn = knn_imputer.fit_transform(X)

# Iterative imputation (MICE)
iter_imputer = IterativeImputer(max_iter=10)
X_iter = iter_imputer.fit_transform(X)

# Time series
ts_imputer = TimeSeriesImputer(method='linear')
X_ts = ts_imputer.fit_transform(X_timeseries)
```

### outliers.py
```python
from data_processing.outliers import IQRDetector, IsolationForest, LocalOutlierFactor

# IQR method
iqr = IQRDetector(k=1.5)
labels = iqr.fit_predict(X)  # -1 for outliers, 1 for inliers

# Isolation Forest
iforest = IsolationForest(n_estimators=100, contamination=0.1)
labels = iforest.fit_predict(X)

# LOF
lof = LocalOutlierFactor(n_neighbors=20)
labels = lof.fit_predict(X)
```

### validation.py
```python
from data_processing.validation import DataValidator, ValidationReport

# Schema validation
schema = {
    'age': {'type': 'numeric', 'min': 0, 'max': 120, 'nullable': False},
    'score': {'type': 'numeric', 'min': 0, 'max': 100, 'nullable': True}
}

validator = DataValidator(schema)
is_valid, errors = validator.validate(X, feature_names)

# Validation report
report = ValidationReport.generate(X, feature_names, validator)
ValidationReport.print_report(report)
```

### splitting.py
```python
from data_processing.splitting import train_test_split, StratifiedKFold, TimeSeriesSplit

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# Train-val-test split
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    X, y, val_size=0.15, test_size=0.15, stratify=y
)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5)
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

## Complete Workflow Example

```python
import numpy as np
from data_processing import *

# Load data
X = load_data()  # Shape: (n_samples, n_features)
y = load_labels()  # Shape: (n_samples,)

# 1. Validation
print("Step 1: Validating data...")
schema = {
    'feature1': {'type': 'numeric', 'min': 0, 'max': 100},
    'feature2': {'type': 'numeric', 'min': 0, 'nullable': False}
}
validator = DataValidator(schema)
is_valid, errors = validator.validate(X, feature_names)

if not is_valid:
    print(f"Validation errors: {errors}")

# 2. Missing value imputation
print("Step 2: Imputing missing values...")
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

# 3. Outlier detection and removal
print("Step 3: Detecting outliers...")
detector = IsolationForest(contamination=0.05)
labels = detector.fit_predict(X_imputed)
X_clean = X_imputed[labels == 1]
y_clean = y[labels == 1]

print(f"Removed {np.sum(labels == -1)} outliers")

# 4. Handle imbalanced classes
print("Step 4: Handling imbalanced data...")
smote = SMOTE(k_neighbors=5)
X_balanced, y_balanced = smote.fit_resample(X_clean, y_clean)

print(f"Original distribution: {Counter(y_clean)}")
print(f"Balanced distribution: {Counter(y_balanced)}")

# 5. Data augmentation
print("Step 5: Augmenting data...")
augmenter = TabularAugmenter()
X_aug, y_aug = augmenter.augment_batch(
    X_balanced, y_balanced,
    augmentation_factor=2,
    methods=['noise', 'mixup']
)

# 6. Train/val/test split
print("Step 6: Splitting data...")
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    X_aug, y_aug,
    val_size=0.15,
    test_size=0.15,
    stratify=y_aug
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 7. Validation report
print("Step 7: Generating validation report...")
report = ValidationReport.generate(X_train, feature_names)
ValidationReport.print_report(report)

print("Data processing complete!")
```

## Best Practices

### 1. Always Validate First
```python
# Validate before processing
validator = DataValidator(schema)
is_valid, errors = validator.validate(X)
if not is_valid:
    print(f"Fix these issues first: {errors}")
```

### 2. Handle Missing Values Before Outlier Detection
```python
# Correct order
X_imputed = imputer.fit_transform(X)
outliers = detector.fit_predict(X_imputed)

# Wrong: detecting outliers with missing values will cause issues
```

### 3. Use Stratified Splits for Imbalanced Data
```python
# Always use stratify parameter for classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y  # ← Important!
)
```

### 4. Apply Augmentation After Splitting
```python
# Correct: augment training data only
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_aug, y_train_aug = augmenter.augment_batch(X_train, y_train)

# Wrong: augmenting before split causes data leakage
```

### 5. Choose Appropriate Resampling Strategy
```python
# For moderate imbalance (2:1 to 5:1)
smote = SMOTE()
X_balanced, y_balanced = smote.fit_resample(X, y)

# For severe imbalance (>10:1)
# Combine oversampling and undersampling
smote_tomek = SMOTETomek()
X_balanced, y_balanced = smote_tomek.fit_resample(X, y)

# Or use class weights
weights = compute_sample_weights(y)
# Pass weights to model.fit(X, y, sample_weight=weights)
```

### 6. Pipeline Multiple Operations
```python
from collections import Counter

def preprocess_pipeline(X, y):
    """Complete preprocessing pipeline."""
    # 1. Imputation
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # 2. Outlier removal
    detector = IsolationForest(contamination=0.05)
    mask = detector.fit_predict(X) == 1
    X, y = X[mask], y[mask]

    # 3. Handle imbalance
    smote = SMOTE(k_neighbors=5)
    X, y = smote.fit_resample(X, y)

    return X, y

X_processed, y_processed = preprocess_pipeline(X, y)
```

## Performance Considerations

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| SMOTE | O(n * k * d) | n=samples, k=neighbors, d=features |
| KNN Imputation | O(n² * d) | Slow for large datasets |
| Isolation Forest | O(t * ψ * log ψ) | t=trees, ψ=subsample size |
| LOF | O(n² * d) | Slow for large datasets |
| IQR Detection | O(n * d) | Very fast |

**Tips:**
- Use IQR/Z-score for quick outlier detection on large datasets
- Use Isolation Forest for accurate outlier detection
- KNN Imputation is slow; use IterativeImputer or SimpleImputer for large data
- SMOTE can generate many synthetic samples; consider contamination parameter

## Requirements

```bash
pip install numpy
```

Optional:
```bash
pip install scipy  # For advanced statistics
```

## File Structure

```
data-processing/
├── __init__.py              # Package initialization
├── augmentation.py          # Data augmentation
├── imbalanced.py            # Imbalanced data handling
├── imputation.py            # Missing value imputation
├── outliers.py              # Outlier detection
├── validation.py            # Data validation
├── splitting.py             # Train/test splits
└── README.md                # This file
```

## License

This framework is provided as-is for educational and commercial use.

---

**Happy Processing! 🔧**
