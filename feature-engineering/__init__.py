"""
Feature Engineering Framework
Comprehensive tools for feature extraction, preprocessing, selection, and pipeline creation
"""

__version__ = "1.0.0"

# Import main components
from .extraction import (
    TextFeatureExtractor,
    TimeSeriesFeatureExtractor,
    DateTimeFeatureExtractor,
    StructuredFeatureExtractor,
    FeatureImportanceAnalyzer
)

from .preprocessing import (
    FeatureScaler,
    FeatureTransformer,
    CategoricalEncoder,
    BinningTransformer,
    FeatureInteractionGenerator
)

from .selection import (
    FeatureSelector,
    DimensionalityReducer
)

from .pipeline import (
    FeatureEngineeringPipeline
)

__all__ = [
    # Extraction
    'TextFeatureExtractor',
    'TimeSeriesFeatureExtractor',
    'DateTimeFeatureExtractor',
    'StructuredFeatureExtractor',
    'FeatureImportanceAnalyzer',

    # Preprocessing
    'FeatureScaler',
    'FeatureTransformer',
    'CategoricalEncoder',
    'BinningTransformer',
    'FeatureInteractionGenerator',

    # Selection
    'FeatureSelector',
    'DimensionalityReducer',

    # Pipeline
    'FeatureEngineeringPipeline',
]
