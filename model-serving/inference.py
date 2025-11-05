"""
Model Inference Module

Comprehensive inference utilities:
- Batch inference
- Real-time inference
- Model loading and caching
- Request batching
- Inference optimization
- Performance monitoring

Author: ML Framework Team
"""

import numpy as np
import time
from typing import Any, List, Dict, Optional, Union, Callable
from collections import deque
import threading
import queue
import warnings


# ============================================================================
# MODEL LOADER
# ============================================================================

class ModelLoader:
    """
    Unified model loader for different frameworks.
    """

    @staticmethod
    def load_pytorch(model_path: str, device: str = 'cpu') -> Any:
        """Load PyTorch model."""
        try:
            import torch
            model = torch.jit.load(model_path, map_location=device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading PyTorch model: {str(e)}")
            return None

    @staticmethod
    def load_onnx(model_path: str) -> Any:
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(model_path)
            return session
        except ImportError:
            print("onnxruntime not available. Install with: pip install onnxruntime")
            return None
        except Exception as e:
            print(f"Error loading ONNX model: {str(e)}")
            return None

    @staticmethod
    def load_tensorflow(model_path: str) -> Any:
        """Load TensorFlow SavedModel."""
        try:
            import tensorflow as tf
            model = tf.saved_model.load(model_path)
            return model
        except Exception as e:
            print(f"Error loading TensorFlow model: {str(e)}")
            return None

    @staticmethod
    def load_sklearn(model_path: str, format: str = 'joblib') -> Any:
        """Load scikit-learn model."""
        try:
            if format == 'joblib':
                import joblib
                return joblib.load(model_path)
            else:  # pickle
                import pickle
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading sklearn model: {str(e)}")
            return None


# ============================================================================
# BATCH INFERENCE
# ============================================================================

class BatchInferenceEngine:
    """
    Efficient batch inference with preprocessing and postprocessing.
    """

    def __init__(
        self,
        model: Any,
        framework: str = 'pytorch',
        device: str = 'cpu',
        batch_size: int = 32,
        preprocessing_fn: Optional[Callable] = None,
        postprocessing_fn: Optional[Callable] = None
    ):
        """
        Initialize batch inference engine.

        Parameters:
        -----------
        model : Model object
            Loaded model.
        framework : str
            Model framework.
        device : str
            Device for inference.
        batch_size : int
            Batch size for inference.
        preprocessing_fn : callable, optional
            Preprocessing function.
        postprocessing_fn : callable, optional
            Postprocessing function.
        """
        self.model = model
        self.framework = framework
        self.device = device
        self.batch_size = batch_size
        self.preprocessing_fn = preprocessing_fn or (lambda x: x)
        self.postprocessing_fn = postprocessing_fn or (lambda x: x)

    def predict(self, inputs: Union[np.ndarray, List]) -> np.ndarray:
        """
        Run inference on batch of inputs.

        Parameters:
        -----------
        inputs : np.ndarray or list
            Input data.

        Returns:
        --------
        predictions : np.ndarray
            Model predictions.
        """
        # Preprocess
        inputs = self.preprocessing_fn(inputs)

        # Convert to numpy if needed
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)

        # Framework-specific inference
        if self.framework == 'pytorch':
            predictions = self._predict_pytorch(inputs)
        elif self.framework == 'onnx':
            predictions = self._predict_onnx(inputs)
        elif self.framework == 'tensorflow':
            predictions = self._predict_tensorflow(inputs)
        elif self.framework == 'sklearn':
            predictions = self._predict_sklearn(inputs)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

        # Postprocess
        predictions = self.postprocessing_fn(predictions)

        return predictions

    def _predict_pytorch(self, inputs: np.ndarray) -> np.ndarray:
        """PyTorch inference."""
        import torch

        with torch.no_grad():
            # Convert to tensor
            inputs_tensor = torch.from_numpy(inputs).to(self.device)

            # Inference
            outputs = self.model(inputs_tensor)

            # Convert back to numpy
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            return outputs.cpu().numpy()

    def _predict_onnx(self, inputs: np.ndarray) -> np.ndarray:
        """ONNX inference."""
        # Get input name
        input_name = self.model.get_inputs()[0].name

        # Run inference
        outputs = self.model.run(None, {input_name: inputs.astype(np.float32)})

        return outputs[0]

    def _predict_tensorflow(self, inputs: np.ndarray) -> np.ndarray:
        """TensorFlow inference."""
        import tensorflow as tf

        # Convert to tensor
        inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)

        # Inference
        outputs = self.model(inputs_tensor)

        return outputs.numpy()

    def _predict_sklearn(self, inputs: np.ndarray) -> np.ndarray:
        """Scikit-learn inference."""
        return self.model.predict(inputs)

    def predict_batches(
        self,
        inputs: np.ndarray,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Run inference on large dataset in batches.

        Parameters:
        -----------
        inputs : np.ndarray
            Large input dataset.
        show_progress : bool
            Show progress bar.

        Returns:
        --------
        predictions : np.ndarray
            All predictions.
        """
        n_samples = len(inputs)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        predictions_list = []

        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)

            batch = inputs[start_idx:end_idx]
            batch_predictions = self.predict(batch)
            predictions_list.append(batch_predictions)

            if show_progress:
                progress = (i + 1) / n_batches * 100
                print(f"\rProgress: {progress:.1f}%", end='')

        if show_progress:
            print()  # New line

        return np.concatenate(predictions_list, axis=0)


# ============================================================================
# REAL-TIME INFERENCE
# ============================================================================

class RealtimeInferenceServer:
    """
    Real-time inference server with request batching.
    """

    def __init__(
        self,
        model: Any,
        framework: str = 'pytorch',
        device: str = 'cpu',
        max_batch_size: int = 8,
        max_wait_time: float = 0.01,  # 10ms
        preprocessing_fn: Optional[Callable] = None,
        postprocessing_fn: Optional[Callable] = None
    ):
        """
        Initialize real-time inference server.

        Parameters:
        -----------
        model : Model object
            Loaded model.
        framework : str
            Model framework.
        device : str
            Device for inference.
        max_batch_size : int
            Maximum batch size for dynamic batching.
        max_wait_time : float
            Maximum time to wait for batching (seconds).
        preprocessing_fn : callable, optional
            Preprocessing function.
        postprocessing_fn : callable, optional
            Postprocessing function.
        """
        self.batch_engine = BatchInferenceEngine(
            model, framework, device, max_batch_size,
            preprocessing_fn, postprocessing_fn
        )

        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time

        # Request queue
        self.request_queue = queue.Queue()
        self.running = False
        self.worker_thread = None

        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'total_batches': 0,
            'total_latency': 0.0,
            'avg_batch_size': 0.0
        }

    def start(self):
        """Start inference server."""
        if self.running:
            print("Server already running")
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._process_requests)
        self.worker_thread.start()
        print("Real-time inference server started")

    def stop(self):
        """Stop inference server."""
        if not self.running:
            return

        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        print("Real-time inference server stopped")

    def predict(self, input_data: np.ndarray, timeout: float = 5.0) -> np.ndarray:
        """
        Submit prediction request.

        Parameters:
        -----------
        input_data : np.ndarray
            Input data (single sample).
        timeout : float
            Request timeout in seconds.

        Returns:
        --------
        prediction : np.ndarray
            Model prediction.
        """
        if not self.running:
            raise RuntimeError("Server not running. Call start() first.")

        # Create result queue for this request
        result_queue = queue.Queue()

        # Submit request
        request = {
            'input': input_data,
            'result_queue': result_queue,
            'submit_time': time.time()
        }

        self.request_queue.put(request)
        self.metrics['total_requests'] += 1

        # Wait for result
        try:
            result = result_queue.get(timeout=timeout)
            return result
        except queue.Empty:
            raise TimeoutError(f"Request timed out after {timeout}s")

    def _process_requests(self):
        """Process requests with dynamic batching."""
        while self.running:
            # Collect requests for batch
            batch_requests = []
            batch_start_time = time.time()

            # Get first request (blocking with timeout)
            try:
                first_request = self.request_queue.get(timeout=0.1)
                batch_requests.append(first_request)
            except queue.Empty:
                continue

            # Collect more requests until batch is full or timeout
            while len(batch_requests) < self.max_batch_size:
                time_elapsed = time.time() - batch_start_time

                if time_elapsed >= self.max_wait_time:
                    break

                try:
                    remaining_time = self.max_wait_time - time_elapsed
                    request = self.request_queue.get(timeout=remaining_time)
                    batch_requests.append(request)
                except queue.Empty:
                    break

            # Process batch
            if batch_requests:
                self._process_batch(batch_requests)

    def _process_batch(self, batch_requests: List[Dict]):
        """Process a batch of requests."""
        # Collect inputs
        inputs = np.array([req['input'] for req in batch_requests])

        # Run inference
        start_time = time.time()
        predictions = self.batch_engine.predict(inputs)
        latency = time.time() - start_time

        # Update metrics
        self.metrics['total_batches'] += 1
        self.metrics['total_latency'] += latency
        self.metrics['avg_batch_size'] = (
            (self.metrics['avg_batch_size'] * (self.metrics['total_batches'] - 1) + len(batch_requests))
            / self.metrics['total_batches']
        )

        # Return results
        for i, request in enumerate(batch_requests):
            request['result_queue'].put(predictions[i])

    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        metrics = self.metrics.copy()

        if metrics['total_requests'] > 0:
            metrics['avg_latency'] = metrics['total_latency'] / metrics['total_batches']
            metrics['requests_per_second'] = metrics['total_requests'] / metrics['total_latency'] if metrics['total_latency'] > 0 else 0

        return metrics


# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

class InferencePipeline:
    """
    Complete inference pipeline with preprocessing, inference, and postprocessing.
    """

    def __init__(
        self,
        model_path: str,
        framework: str = 'pytorch',
        device: str = 'cpu'
    ):
        """
        Initialize inference pipeline.

        Parameters:
        -----------
        model_path : str
            Path to model file.
        framework : str
            Model framework.
        device : str
            Device for inference.
        """
        self.framework = framework
        self.device = device

        # Load model
        print(f"Loading {framework} model from {model_path}...")
        self.model = self._load_model(model_path)

        # Performance tracking
        self.inference_times = deque(maxlen=1000)

    def _load_model(self, model_path: str) -> Any:
        """Load model based on framework."""
        loader = ModelLoader()

        if self.framework == 'pytorch':
            return loader.load_pytorch(model_path, self.device)
        elif self.framework == 'onnx':
            return loader.load_onnx(model_path)
        elif self.framework == 'tensorflow':
            return loader.load_tensorflow(model_path)
        elif self.framework == 'sklearn':
            return loader.load_sklearn(model_path)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

    def preprocess(self, inputs: Any) -> np.ndarray:
        """
        Preprocess inputs (override in subclass).

        Parameters:
        -----------
        inputs : Any
            Raw inputs.

        Returns:
        --------
        processed : np.ndarray
            Processed inputs.
        """
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
        return inputs

    def postprocess(self, predictions: np.ndarray) -> Any:
        """
        Postprocess predictions (override in subclass).

        Parameters:
        -----------
        predictions : np.ndarray
            Raw predictions.

        Returns:
        --------
        processed : Any
            Processed predictions.
        """
        return predictions

    def predict(self, inputs: Any) -> Any:
        """
        Run complete inference pipeline.

        Parameters:
        -----------
        inputs : Any
            Raw inputs.

        Returns:
        --------
        predictions : Any
            Final predictions.
        """
        start_time = time.time()

        # Preprocess
        processed_inputs = self.preprocess(inputs)

        # Create batch engine
        engine = BatchInferenceEngine(
            self.model,
            self.framework,
            self.device
        )

        # Inference
        predictions = engine.predict(processed_inputs)

        # Postprocess
        final_predictions = self.postprocess(predictions)

        # Track performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        return final_predictions

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.inference_times:
            return {}

        times = list(self.inference_times)
        return {
            'avg_latency_ms': np.mean(times) * 1000,
            'p50_latency_ms': np.percentile(times, 50) * 1000,
            'p95_latency_ms': np.percentile(times, 95) * 1000,
            'p99_latency_ms': np.percentile(times, 99) * 1000,
            'throughput_qps': 1 / np.mean(times) if np.mean(times) > 0 else 0
        }


# ============================================================================
# EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MODEL INFERENCE EXAMPLES")
    print("=" * 70)

    print("\n1. Batch Inference")
    print("-" * 70)
    print("""
# Load model
loader = ModelLoader()
model = loader.load_pytorch('model.pt', device='cpu')

# Create batch engine
engine = BatchInferenceEngine(
    model,
    framework='pytorch',
    device='cpu',
    batch_size=32
)

# Generate test data
test_data = np.random.randn(1000, 10).astype(np.float32)

# Batch inference
predictions = engine.predict_batches(test_data, show_progress=True)
print(f"Predictions shape: {predictions.shape}")
""")

    print("\n2. Real-time Inference")
    print("-" * 70)
    print("""
# Create real-time server
server = RealtimeInferenceServer(
    model,
    framework='pytorch',
    device='cpu',
    max_batch_size=8,
    max_wait_time=0.01  # 10ms
)

# Start server
server.start()

# Submit requests
for i in range(100):
    input_data = np.random.randn(10).astype(np.float32)
    prediction = server.predict(input_data)
    print(f"Request {i}: {prediction}")

# Get metrics
metrics = server.get_metrics()
print(f"Average latency: {metrics['avg_latency']:.4f}s")
print(f"Average batch size: {metrics['avg_batch_size']:.2f}")
print(f"Requests per second: {metrics['requests_per_second']:.2f}")

# Stop server
server.stop()
""")

    print("\n3. Inference Pipeline")
    print("-" * 70)
    print("""
class ImageClassificationPipeline(InferencePipeline):
    def preprocess(self, inputs):
        # Resize, normalize, etc.
        processed = inputs / 255.0
        return processed.astype(np.float32)

    def postprocess(self, predictions):
        # Get class labels
        class_ids = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        return list(zip(class_ids, confidences))

# Create pipeline
pipeline = ImageClassificationPipeline(
    'model.onnx',
    framework='onnx',
    device='cpu'
)

# Run inference
results = pipeline.predict(images)

# Get performance stats
stats = pipeline.get_performance_stats()
print(f"Average latency: {stats['avg_latency_ms']:.2f}ms")
print(f"P95 latency: {stats['p95_latency_ms']:.2f}ms")
print(f"Throughput: {stats['throughput_qps']:.2f} QPS")
""")

    print("\n" + "=" * 70)
    print("Model inference examples completed!")
    print("=" * 70)
