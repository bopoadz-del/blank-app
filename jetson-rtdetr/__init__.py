"""
RT-DETR TensorRT Inference for NVIDIA Jetson
High-performance object detection with TensorRT optimization
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

# Import main classes for convenience
try:
    from .inference.inference_engine import InferenceEngine, MultiStreamInference, BatchProcessor
    from .models.rtdetr_tensorrt import RTDETRTensorRTBuilder, TensorRTInferenceContext
    from .preprocessing.image_preprocessor import ImagePreprocessor, VideoPreprocessor
    from .postprocessing.nms_filter import (
        RTDETRPostprocessor,
        Detection,
        DetectionVisualizer
    )
    from .benchmarks.fps_benchmark import (
        FPSMeter,
        LatencyTracker,
        InferenceBenchmark,
        BenchmarkResult
    )

    __all__ = [
        # Inference
        'InferenceEngine',
        'MultiStreamInference',
        'BatchProcessor',
        # TensorRT
        'RTDETRTensorRTBuilder',
        'TensorRTInferenceContext',
        # Preprocessing
        'ImagePreprocessor',
        'VideoPreprocessor',
        # Postprocessing
        'RTDETRPostprocessor',
        'Detection',
        'DetectionVisualizer',
        # Benchmarking
        'FPSMeter',
        'LatencyTracker',
        'InferenceBenchmark',
        'BenchmarkResult',
    ]

except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Some dependencies are missing: {e}")
    __all__ = []
