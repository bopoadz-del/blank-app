"""
Model Serving Framework

Production-ready infrastructure for model deployment.

Modules:
- export: Model export to ONNX, TorchScript, SavedModel, etc.
- optimization: Quantization, pruning, graph optimization
- inference: Batch and real-time inference engines
- ab_testing: A/B testing and gradual rollout infrastructure

Author: ML Framework Team
Version: 1.0.0
"""

# Export
from .export import (
    ONNXExporter,
    TorchScriptExporter,
    TensorFlowExporter,
    SklearnExporter,
    ModelMetadata,
    ModelExporter
)

# Optimization
from .optimization import (
    PyTorchQuantizer,
    PyTorchPruner,
    TensorFlowQuantizer,
    ONNXOptimizer,
    ModelCompressor
)

# Inference
from .inference import (
    ModelLoader,
    BatchInferenceEngine,
    RealtimeInferenceServer,
    InferencePipeline
)

# A/B Testing
from .ab_testing import (
    ModelVariant,
    TrafficRouter,
    ModelComparator,
    ChampionChallenger,
    GradualRollout
)

__all__ = [
    # Export
    'ONNXExporter',
    'TorchScriptExporter',
    'TensorFlowExporter',
    'SklearnExporter',
    'ModelMetadata',
    'ModelExporter',

    # Optimization
    'PyTorchQuantizer',
    'PyTorchPruner',
    'TensorFlowQuantizer',
    'ONNXOptimizer',
    'ModelCompressor',

    # Inference
    'ModelLoader',
    'BatchInferenceEngine',
    'RealtimeInferenceServer',
    'InferencePipeline',

    # A/B Testing
    'ModelVariant',
    'TrafficRouter',
    'ModelComparator',
    'ChampionChallenger',
    'GradualRollout',
]

__version__ = '1.0.0'
__author__ = 'ML Framework Team'
