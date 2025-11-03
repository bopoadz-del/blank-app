"""
Data Processing Framework

Comprehensive data processing utilities for machine learning workflows.

Modules:
- augmentation: Data augmentation for images, text, and tabular data
- imbalanced: Handling imbalanced datasets (SMOTE, undersampling, etc.)
- imputation: Missing value imputation strategies
- outliers: Outlier detection methods
- validation: Data validation and quality checks
- splitting: Train/test/validation splits with stratification

Author: ML Framework Team
Version: 1.0.0
"""

# Augmentation
from .augmentation import (
    ImageAugmenter,
    TextAugmenter,
    TabularAugmenter
)

# Imbalanced data handling
from .imbalanced import (
    RandomOverSampler,
    SMOTE,
    ADASYN,
    RandomUnderSampler,
    TomekLinks,
    NearMiss,
    SMOTETomek,
    compute_class_weights,
    compute_sample_weights
)

# Imputation
from .imputation import (
    SimpleImputer,
    KNNImputer,
    IterativeImputer,
    TimeSeriesImputer,
    MissingIndicator
)

# Outlier detection
from .outliers import (
    IQRDetector,
    ZScoreDetector,
    ModifiedZScoreDetector,
    KNNOutlierDetector,
    LocalOutlierFactor,
    IsolationForest,
    MahalanobisDetector
)

# Validation
from .validation import (
    DataValidator,
    TypeValidator,
    RangeValidator,
    ConsistencyValidator,
    DataQualityMetrics,
    ValidationReport
)

# Splitting
from .splitting import (
    train_test_split,
    train_val_test_split,
    KFold,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit
)

__all__ = [
    # Augmentation
    'ImageAugmenter',
    'TextAugmenter',
    'TabularAugmenter',

    # Imbalanced data
    'RandomOverSampler',
    'SMOTE',
    'ADASYN',
    'RandomUnderSampler',
    'TomekLinks',
    'NearMiss',
    'SMOTETomek',
    'compute_class_weights',
    'compute_sample_weights',

    # Imputation
    'SimpleImputer',
    'KNNImputer',
    'IterativeImputer',
    'TimeSeriesImputer',
    'MissingIndicator',

    # Outlier detection
    'IQRDetector',
    'ZScoreDetector',
    'ModifiedZScoreDetector',
    'KNNOutlierDetector',
    'LocalOutlierFactor',
    'IsolationForest',
    'MahalanobisDetector',

    # Validation
    'DataValidator',
    'TypeValidator',
    'RangeValidator',
    'ConsistencyValidator',
    'DataQualityMetrics',
    'ValidationReport',

    # Splitting
    'train_test_split',
    'train_val_test_split',
    'KFold',
    'StratifiedKFold',
    'GroupKFold',
    'TimeSeriesSplit',
]

__version__ = '1.0.0'
__author__ = 'ML Framework Team'
