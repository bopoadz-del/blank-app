"""
Model Optimization Framework
Comprehensive tools for hyperparameter tuning, model compression, and AutoML
"""

__version__ = "1.0.0"

# Import main components
from .hyperparameter_tuning import (
    GridSearchTuner,
    RandomSearchTuner,
    OptunaTuner,
    CrossValidator
)

from .optimization import (
    ModelPruner,
    ModelQuantizer,
    KnowledgeDistiller
)

from .automl import (
    AutoMLPipeline
)

__all__ = [
    # Hyperparameter Tuning
    'GridSearchTuner',
    'RandomSearchTuner',
    'OptunaTuner',
    'CrossValidator',

    # Model Compression
    'ModelPruner',
    'ModelQuantizer',
    'KnowledgeDistiller',

    # AutoML
    'AutoMLPipeline',
]
