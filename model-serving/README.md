# Model Serving Framework

Production-ready model serving infrastructure with export, optimization, inference, and A/B testing capabilities.

## Overview

This framework provides complete tools for deploying ML models to production:

### Model Export
- **ONNX**: Cross-platform model format
- **TorchScript**: PyTorch production format (tracing & scripting)
- **SavedModel**: TensorFlow serving format
- **TFLite**: Mobile/edge deployment
- **Pickle/Joblib**: scikit-learn models
- **Model metadata**: Versioning and tracking

### Model Optimization
- **Quantization**: INT8, FP16 for faster inference
- **Pruning**: Structured and unstructured weight pruning
- **Graph optimization**: ONNX optimization passes
- **Compression**: Model size analysis and comparison

### Inference
- **Batch inference**: Efficient processing of large datasets
- **Real-time inference**: Low-latency serving with request batching
- **Model loading**: Unified interface for all frameworks
- **Performance monitoring**: Latency and throughput tracking

### A/B Testing
- **Traffic routing**: Weighted, round-robin, epsilon-greedy
- **Champion/Challenger**: Safe model comparison
- **Gradual rollout**: Staged deployment with automatic rollback
- **Performance comparison**: Statistical significance testing

## Quick Start

```python
from model_serving import *

# 1. Export model
exporter = ModelExporter()
exporter.export(model, 'onnx', 'model.onnx', dummy_input=x)

# 2. Optimize
quantizer = PyTorchQuantizer()
quantized_model = quantizer.dynamic_quantization(model)

# 3. Serve
engine = BatchInferenceEngine(model, framework='pytorch')
predictions = engine.predict(inputs)

# 4. A/B test
router = TrafficRouter(variants, routing_strategy='weighted')
prediction, variant = router.predict(inputs)
```

## Modules

### export.py

**ONNX Export:**
```python
exporter = ONNXExporter()
exporter.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)
```

**TorchScript Export:**
```python
exporter = TorchScriptExporter()
# Tracing (for static models)
exporter.export_trace(model, example_inputs, 'model_traced.pt')
# Scripting (for models with control flow)
exporter.export_script(model, 'model_scripted.pt')
```

**Model Metadata:**
```python
metadata = ModelMetadata.create_metadata(
    model_name='my_classifier',
    version='1.0.0',
    framework='pytorch',
    model_type='classifier',
    input_shape=[1, 10],
    output_shape=[1, 2],
    metrics={'accuracy': 0.95}
)
ModelMetadata.save_metadata(metadata, 'metadata.json')
```

### optimization.py

**Dynamic Quantization (fastest, minimal accuracy loss):**
```python
quantizer = PyTorchQuantizer()
quantized = quantizer.dynamic_quantization(model, dtype='qint8')

# Check compression
compressor = ModelCompressor()
comparison = compressor.compare_models(model, quantized)
print(f"Size reduction: {comparison['size_reduction_percent']:.1f}%")
```

**Static Quantization (requires calibration):**
```python
quantized = quantizer.static_quantization(
    model,
    calibration_data_loader,
    backend='fbgemm'
)
```

**Pruning:**
```python
pruner = PyTorchPruner()
# Remove 30% of weights
pruned = pruner.unstructured_pruning(model, amount=0.3, method='l1')
# Structured pruning (entire channels)
pruned = pruner.structured_pruning(model, amount=0.2, dim=0)
```

**ONNX Optimization:**
```python
optimizer = ONNXOptimizer()
optimizer.optimize(
    'model.onnx',
    'model_optimized.onnx',
    optimization_level='extended'
)
```

### inference.py

**Batch Inference:**
```python
# Load model
loader = ModelLoader()
model = loader.load_onnx('model.onnx')

# Create engine
engine = BatchInferenceEngine(
    model,
    framework='onnx',
    batch_size=32
)

# Process large dataset
predictions = engine.predict_batches(test_data, show_progress=True)
```

**Real-time Inference Server:**
```python
server = RealtimeInferenceServer(
    model,
    framework='pytorch',
    max_batch_size=8,
    max_wait_time=0.01  # 10ms batching window
)

server.start()

# Submit requests
for i in range(1000):
    prediction = server.predict(input_data)

# Get metrics
metrics = server.get_metrics()
print(f"Avg latency: {metrics['avg_latency']:.4f}s")
print(f"Throughput: {metrics['requests_per_second']:.2f} QPS")

server.stop()
```

**Custom Inference Pipeline:**
```python
class MyPipeline(InferencePipeline):
    def preprocess(self, inputs):
        # Custom preprocessing
        return inputs / 255.0

    def postprocess(self, predictions):
        # Custom postprocessing
        return np.argmax(predictions, axis=1)

pipeline = MyPipeline('model.onnx', framework='onnx')
results = pipeline.predict(data)

stats = pipeline.get_performance_stats()
print(f"P95 latency: {stats['p95_latency_ms']:.2f}ms")
```

### ab_testing.py

**Traffic Routing:**
```python
variants = {
    'v1': ModelVariant('v1', model_v1, traffic_weight=0.7),
    'v2': ModelVariant('v2', model_v2, traffic_weight=0.3)
}

router = TrafficRouter(variants, routing_strategy='weighted')

# Route traffic
prediction, variant = router.predict(inputs)

# Compare performance
comparison = ModelComparator.compare_metrics(variants)
```

**Champion/Challenger:**
```python
cc = ChampionChallenger(
    champion_model=current_model,
    challenger_model=new_model,
    challenger_traffic=0.1,  # 10% to new model
    promotion_threshold=0.05  # 5% improvement needed
)

# Serve traffic
for request in requests:
    prediction, variant = cc.predict(request)

# Evaluate promotion
evaluation = cc.evaluate_promotion()
if evaluation['recommendation'] == 'promote':
    cc.promote_challenger()
```

**Gradual Rollout:**
```python
rollout = GradualRollout(
    old_model=v1,
    new_model=v2,
    stages=[0.05, 0.10, 0.25, 0.50, 1.0],
    stage_duration_minutes=60,
    rollback_error_threshold=0.05
)

# Monitor rollout
while True:
    status = rollout.advance_stage()

    if status['status'] == 'complete':
        print("Rollout complete!")
        break
    elif status['status'] == 'rollback':
        print(f"Rollback triggered: {status['reason']}")
        break
```

## Complete Deployment Example

```python
import numpy as np
from model_serving import *

# Step 1: Export model
print("Exporting model...")
exporter = ModelExporter()
exporter.export(pytorch_model, 'onnx', 'model.onnx', dummy_input=sample_input)

# Step 2: Optimize
print("Optimizing ONNX model...")
optimizer = ONNXOptimizer()
optimizer.optimize('model.onnx', 'model_opt.onnx', optimization_level='extended')

# Step 3: Load optimized model
print("Loading optimized model...")
loader = ModelLoader()
model = loader.load_onnx('model_opt.onnx')

# Step 4: Set up inference
print("Starting inference server...")
server = RealtimeInferenceServer(
    model,
    framework='onnx',
    max_batch_size=16,
    max_wait_time=0.01
)
server.start()

# Step 5: Serve traffic
print("Serving predictions...")
for i in range(10000):
    input_data = generate_input()
    prediction = server.predict(input_data)
    process_prediction(prediction)

# Step 6: Monitor performance
metrics = server.get_metrics()
print(f"Average latency: {metrics['avg_latency']:.4f}s")
print(f"Average batch size: {metrics['avg_batch_size']:.2f}")
print(f"Throughput: {metrics['requests_per_second']:.2f} QPS")

server.stop()
```

## A/B Testing Workflow

```python
# Load models
champion = ModelLoader.load_onnx('model_v1.onnx')
challenger = ModelLoader.load_onnx('model_v2.onnx')

# Set up Champion/Challenger test
cc = ChampionChallenger(
    champion_model=champion,
    challenger_model=challenger,
    challenger_traffic=0.1  # Start with 10%
)

# Serve traffic for evaluation period
for i in range(10000):
    input_data = get_next_request()
    prediction, variant = cc.predict(input_data)

    if i % 1000 == 0:
        # Check progress
        comparison = ModelComparator.compare_metrics(cc.variants)
        print(f"Progress: {i}/10000")
        print(comparison)

# Evaluate after sufficient traffic
evaluation = cc.evaluate_promotion()
print("\nEvaluation Results:")
print(f"Recommendation: {evaluation['recommendation']}")
print(f"Reason: {evaluation['reason']}")
print(f"Latency improvement: {evaluation['latency_improvement_percent']:.2f}%")

if evaluation['recommendation'] == 'promote':
    print("Promoting challenger to champion!")
    cc.promote_challenger()
else:
    print("Keeping current champion")
```

## Performance Tips

### 1. Choose the Right Optimization

| Model Type | Recommended Optimization | Speed Gain | Accuracy Loss |
|------------|-------------------------|------------|---------------|
| Transformer | Dynamic Quantization | 2-3x | Minimal |
| CNN | Static Quantization + Pruning | 3-4x | Low |
| MLP/Linear | Dynamic Quantization | 2-3x | Minimal |

### 2. Batch Size Selection

```python
# Small batches: lower latency, lower throughput
engine = BatchInferenceEngine(model, batch_size=8)

# Large batches: higher latency, higher throughput
engine = BatchInferenceEngine(model, batch_size=128)
```

### 3. Real-time Serving Parameters

```python
# Low latency (single requests)
server = RealtimeInferenceServer(
    model,
    max_batch_size=1,
    max_wait_time=0.001  # 1ms
)

# High throughput (dynamic batching)
server = RealtimeInferenceServer(
    model,
    max_batch_size=32,
    max_wait_time=0.050  # 50ms
)
```

### 4. Model Format Selection

- **ONNX**: Best for cross-platform, supports TensorRT
- **TorchScript**: Best for PyTorch deployment, mobile
- **TFLite**: Best for mobile/edge devices
- **SavedModel**: Best for TensorFlow Serving

## Requirements

Core:
```bash
pip install numpy
```

Optional (for specific features):
```bash
# PyTorch support
pip install torch

# ONNX support
pip install onnx onnxruntime

# TensorFlow support
pip install tensorflow

# Quantization (TensorFlow)
pip install tensorflow-model-optimization

# Scikit-learn support
pip install joblib
```

## File Structure

```
model-serving/
├── __init__.py              # Package initialization
├── export.py                # Model export utilities
├── optimization.py          # Model optimization
├── inference.py             # Batch and real-time inference
├── ab_testing.py            # A/B testing infrastructure
└── README.md                # This file
```

## License

This framework is provided as-is for educational and commercial use.

---

**Happy Serving! 🚀**
