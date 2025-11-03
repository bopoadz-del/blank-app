"""
Model Optimization Module

Techniques for optimizing models for inference:
- Quantization (INT8, FP16)
- Pruning (structured, unstructured)
- Knowledge distillation
- Graph optimization
- Operator fusion
- Memory optimization

Author: ML Framework Team
"""

import numpy as np
from typing import Any, Optional, Dict, List, Tuple, Callable
import warnings


# ============================================================================
# PYTORCH QUANTIZATION
# ============================================================================

class PyTorchQuantizer:
    """
    Quantization for PyTorch models.
    """

    def __init__(self):
        """Initialize quantizer."""
        self.torch_available = self._check_torch()

    def _check_torch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            warnings.warn("PyTorch not available")
            return False

    def dynamic_quantization(
        self,
        model: Any,
        dtype: str = 'qint8',
        qconfig_spec: Optional[Dict] = None
    ) -> Any:
        """
        Dynamic quantization (post-training, no calibration data needed).

        Best for: LSTMs, Transformers, Linear layers
        Speed: Fast (no calibration needed)
        Accuracy: High (minimal accuracy loss)

        Parameters:
        -----------
        model : torch.nn.Module
            Model to quantize.
        dtype : str
            Quantization dtype ('qint8' or 'float16').
        qconfig_spec : dict, optional
            Layer-specific quantization config.

        Returns:
        --------
        quantized_model : torch.nn.Module
            Quantized model.
        """
        if not self.torch_available:
            return model

        try:
            import torch

            # Set model to eval
            model.eval()

            # Default quantization: Linear and LSTM layers
            if qconfig_spec is None:
                layers_to_quantize = {torch.nn.Linear, torch.nn.LSTM}
            else:
                layers_to_quantize = qconfig_spec

            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                layers_to_quantize,
                dtype=getattr(torch, dtype)
            )

            print(f"Model dynamically quantized to {dtype}")
            return quantized_model

        except Exception as e:
            print(f"Error in dynamic quantization: {str(e)}")
            return model

    def static_quantization(
        self,
        model: Any,
        calibration_data_loader: Any,
        backend: str = 'fbgemm'
    ) -> Any:
        """
        Static quantization (requires calibration data).

        Best for: CNNs
        Speed: Slower (requires calibration)
        Accuracy: Lower than dynamic (but faster inference)

        Parameters:
        -----------
        model : torch.nn.Module
            Model to quantize.
        calibration_data_loader : DataLoader
            Calibration data for quantization.
        backend : str
            Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM).

        Returns:
        --------
        quantized_model : torch.nn.Module
            Quantized model.
        """
        if not self.torch_available:
            return model

        try:
            import torch

            # Set backend
            torch.backends.quantized.engine = backend

            # Prepare model for quantization
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig(backend)
            model_prepared = torch.quantization.prepare(model)

            # Calibrate
            print("Calibrating model...")
            with torch.no_grad():
                for batch in calibration_data_loader:
                    if isinstance(batch, (tuple, list)):
                        model_prepared(batch[0])
                    else:
                        model_prepared(batch)

            # Convert to quantized model
            quantized_model = torch.quantization.convert(model_prepared)

            print("Model statically quantized")
            return quantized_model

        except Exception as e:
            print(f"Error in static quantization: {str(e)}")
            return model

    def quantization_aware_training(
        self,
        model: Any,
        train_loader: Any,
        num_epochs: int,
        optimizer: Any,
        criterion: Any,
        backend: str = 'fbgemm'
    ) -> Any:
        """
        Quantization-aware training (QAT).

        Best for: Maximum accuracy with quantization
        Speed: Slowest (requires retraining)
        Accuracy: Highest

        Parameters:
        -----------
        model : torch.nn.Module
            Model to train with quantization.
        train_loader : DataLoader
            Training data.
        num_epochs : int
            Number of training epochs.
        optimizer : torch.optim.Optimizer
            Optimizer.
        criterion : torch.nn.Module
            Loss function.
        backend : str
            Quantization backend.

        Returns:
        --------
        quantized_model : torch.nn.Module
            Quantized model.
        """
        if not self.torch_available:
            return model

        try:
            import torch

            # Set backend
            torch.backends.quantized.engine = backend

            # Prepare for QAT
            model.train()
            model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
            model_prepared = torch.quantization.prepare_qat(model)

            # Training loop
            print("Starting quantization-aware training...")
            for epoch in range(num_epochs):
                total_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model_prepared(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Convert to quantized model
            model_prepared.eval()
            quantized_model = torch.quantization.convert(model_prepared)

            print("Quantization-aware training completed")
            return quantized_model

        except Exception as e:
            print(f"Error in QAT: {str(e)}")
            return model


# ============================================================================
# PYTORCH PRUNING
# ============================================================================

class PyTorchPruner:
    """
    Pruning for PyTorch models.
    """

    def __init__(self):
        """Initialize pruner."""
        self.torch_available = self._check_torch()

    def _check_torch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            warnings.warn("PyTorch not available")
            return False

    def unstructured_pruning(
        self,
        model: Any,
        amount: float = 0.3,
        method: str = 'l1'
    ) -> Any:
        """
        Unstructured pruning (prune individual weights).

        Parameters:
        -----------
        model : torch.nn.Module
            Model to prune.
        amount : float
            Fraction of parameters to prune (0.0 to 1.0).
        method : str
            Pruning method ('l1' or 'random').

        Returns:
        --------
        pruned_model : torch.nn.Module
            Pruned model.
        """
        if not self.torch_available:
            return model

        try:
            import torch
            import torch.nn.utils.prune as prune

            # Prune all linear and conv layers
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    if method == 'l1':
                        prune.l1_unstructured(module, name='weight', amount=amount)
                    elif method == 'random':
                        prune.random_unstructured(module, name='weight', amount=amount)

                    # Remove pruning reparameterization (make pruning permanent)
                    prune.remove(module, 'weight')

            print(f"Model pruned ({amount*100:.1f}% of weights removed)")
            return model

        except Exception as e:
            print(f"Error in unstructured pruning: {str(e)}")
            return model

    def structured_pruning(
        self,
        model: Any,
        amount: float = 0.3,
        dim: int = 0
    ) -> Any:
        """
        Structured pruning (prune entire channels/filters).

        Parameters:
        -----------
        model : torch.nn.Module
            Model to prune.
        amount : float
            Fraction of structures to prune.
        dim : int
            Dimension to prune along (0 for output channels, 1 for input channels).

        Returns:
        --------
        pruned_model : torch.nn.Module
            Pruned model.
        """
        if not self.torch_available:
            return model

        try:
            import torch
            import torch.nn.utils.prune as prune

            # Prune conv layers
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.ln_structured(
                        module,
                        name='weight',
                        amount=amount,
                        n=2,
                        dim=dim
                    )
                    prune.remove(module, 'weight')

            print(f"Model structurally pruned ({amount*100:.1f}% of structures removed)")
            return model

        except Exception as e:
            print(f"Error in structured pruning: {str(e)}")
            return model

    def iterative_pruning(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Any,
        num_iterations: int = 5,
        prune_amount_per_iter: float = 0.2,
        fine_tune_epochs: int = 2
    ) -> Any:
        """
        Iterative magnitude pruning with fine-tuning.

        Parameters:
        -----------
        model : torch.nn.Module
            Model to prune.
        train_loader : DataLoader
            Training data.
        val_loader : DataLoader
            Validation data.
        num_iterations : int
            Number of pruning iterations.
        prune_amount_per_iter : float
            Pruning amount per iteration.
        fine_tune_epochs : int
            Fine-tuning epochs after each pruning.

        Returns:
        --------
        pruned_model : torch.nn.Module
            Pruned model.
        """
        print("Iterative pruning not fully implemented in this example")
        print("See PyTorch pruning tutorial for complete implementation")
        return model


# ============================================================================
# TENSORFLOW QUANTIZATION
# ============================================================================

class TensorFlowQuantizer:
    """
    Quantization for TensorFlow models.
    """

    def __init__(self):
        """Initialize quantizer."""
        self.tf_available = self._check_tf()

    def _check_tf(self) -> bool:
        """Check if TensorFlow is available."""
        try:
            import tensorflow as tf
            return True
        except ImportError:
            warnings.warn("TensorFlow not available")
            return False

    def post_training_quantization(
        self,
        model: Any,
        representative_dataset: Optional[Callable] = None
    ) -> bytes:
        """
        Post-training quantization for TensorFlow models.

        Parameters:
        -----------
        model : tf.keras.Model
            Model to quantize.
        representative_dataset : callable, optional
            Generator function yielding representative data.

        Returns:
        --------
        tflite_model : bytes
            Quantized TFLite model.
        """
        if not self.tf_available:
            return None

        try:
            import tensorflow as tf

            # Convert to TFLite with quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # Enable quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # If representative dataset provided, use full integer quantization
            if representative_dataset is not None:
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8

            tflite_model = converter.convert()

            print("Model quantized (TFLite)")
            return tflite_model

        except Exception as e:
            print(f"Error in post-training quantization: {str(e)}")
            return None

    def quantization_aware_training(
        self,
        model: Any
    ) -> Any:
        """
        Prepare model for quantization-aware training.

        Parameters:
        -----------
        model : tf.keras.Model
            Model to prepare for QAT.

        Returns:
        --------
        qat_model : tf.keras.Model
            Model with fake quantization nodes.
        """
        if not self.tf_available:
            return model

        try:
            import tensorflow as tf
            import tensorflow_model_optimization as tfmot

            # Apply quantization-aware training
            quantize_model = tfmot.quantization.keras.quantize_model

            qat_model = quantize_model(model)

            print("Model prepared for quantization-aware training")
            return qat_model

        except ImportError:
            print("tensorflow-model-optimization not available")
            print("Install with: pip install tensorflow-model-optimization")
            return model
        except Exception as e:
            print(f"Error in QAT preparation: {str(e)}")
            return model


# ============================================================================
# ONNX OPTIMIZATION
# ============================================================================

class ONNXOptimizer:
    """
    Optimize ONNX models for inference.
    """

    def __init__(self):
        """Initialize optimizer."""
        self.onnx_available = self._check_onnx()

    def _check_onnx(self) -> bool:
        """Check if ONNX is available."""
        try:
            import onnx
            return True
        except ImportError:
            warnings.warn("ONNX not available")
            return False

    def optimize(
        self,
        model_path: str,
        output_path: str,
        optimization_level: str = 'basic'
    ) -> bool:
        """
        Optimize ONNX model.

        Parameters:
        -----------
        model_path : str
            Path to ONNX model.
        output_path : str
            Path to save optimized model.
        optimization_level : str
            'basic', 'extended', or 'all'.

        Returns:
        --------
        success : bool
        """
        if not self.onnx_available:
            return False

        try:
            from onnx import optimizer

            # Load model
            import onnx
            model = onnx.load(model_path)

            # Optimization passes
            if optimization_level == 'basic':
                passes = ['eliminate_nop_pad', 'eliminate_unused_initializer']
            elif optimization_level == 'extended':
                passes = [
                    'eliminate_nop_pad',
                    'eliminate_unused_initializer',
                    'fuse_bn_into_conv',
                    'fuse_consecutive_squeezes',
                    'fuse_consecutive_transposes',
                ]
            else:  # 'all'
                passes = optimizer.get_available_passes()

            # Optimize
            optimized_model = optimizer.optimize(model, passes)

            # Save
            onnx.save(optimized_model, output_path)

            print(f"ONNX model optimized: {output_path}")
            return True

        except Exception as e:
            print(f"Error optimizing ONNX model: {str(e)}")
            return False


# ============================================================================
# MODEL COMPRESSION UTILITIES
# ============================================================================

class ModelCompressor:
    """
    General model compression utilities.
    """

    @staticmethod
    def calculate_model_size(model: Any, framework: str = 'pytorch') -> Dict[str, float]:
        """
        Calculate model size.

        Parameters:
        -----------
        model : Model object
            Model to analyze.
        framework : str
            Framework ('pytorch', 'tensorflow', 'sklearn').

        Returns:
        --------
        size_info : dict
            Model size information.
        """
        if framework == 'pytorch':
            try:
                import torch
                param_size = 0
                buffer_size = 0

                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()

                for buffer in model.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()

                total_size = param_size + buffer_size

                return {
                    'param_size_mb': param_size / (1024**2),
                    'buffer_size_mb': buffer_size / (1024**2),
                    'total_size_mb': total_size / (1024**2),
                    'n_parameters': sum(p.numel() for p in model.parameters())
                }
            except Exception as e:
                print(f"Error calculating size: {str(e)}")
                return {}

        elif framework == 'tensorflow':
            try:
                import tensorflow as tf
                total_params = model.count_params()

                # Estimate size (assuming float32)
                size_bytes = total_params * 4

                return {
                    'total_size_mb': size_bytes / (1024**2),
                    'n_parameters': total_params
                }
            except Exception as e:
                print(f"Error calculating size: {str(e)}")
                return {}

        else:
            return {}

    @staticmethod
    def compare_models(
        original_model: Any,
        optimized_model: Any,
        framework: str = 'pytorch'
    ) -> Dict[str, Any]:
        """
        Compare original and optimized models.

        Parameters:
        -----------
        original_model : Model
            Original model.
        optimized_model : Model
            Optimized model.
        framework : str
            Framework.

        Returns:
        --------
        comparison : dict
            Comparison metrics.
        """
        compressor = ModelCompressor()

        original_size = compressor.calculate_model_size(original_model, framework)
        optimized_size = compressor.calculate_model_size(optimized_model, framework)

        size_reduction = (
            (original_size['total_size_mb'] - optimized_size['total_size_mb'])
            / original_size['total_size_mb'] * 100
        )

        return {
            'original_size_mb': original_size['total_size_mb'],
            'optimized_size_mb': optimized_size['total_size_mb'],
            'size_reduction_percent': size_reduction,
            'compression_ratio': original_size['total_size_mb'] / optimized_size['total_size_mb']
        }


# ============================================================================
# EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MODEL OPTIMIZATION EXAMPLES")
    print("=" * 70)

    print("\n1. PyTorch Dynamic Quantization")
    print("-" * 70)
    print("""
import torch
import torch.nn as nn

# Define model
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

# Quantize
quantizer = PyTorchQuantizer()
quantized_model = quantizer.dynamic_quantization(model, dtype='qint8')

# Compare sizes
compressor = ModelCompressor()
comparison = compressor.compare_models(model, quantized_model, framework='pytorch')
print(f"Size reduction: {comparison['size_reduction_percent']:.1f}%")
""")

    print("\n2. PyTorch Pruning")
    print("-" * 70)
    print("""
# Unstructured pruning
pruner = PyTorchPruner()
pruned_model = pruner.unstructured_pruning(model, amount=0.3, method='l1')

# Structured pruning (channel pruning)
pruned_model = pruner.structured_pruning(model, amount=0.2, dim=0)
""")

    print("\n3. TensorFlow Quantization")
    print("-" * 70)
    print("""
# Post-training quantization
tf_quantizer = TensorFlowQuantizer()

def representative_dataset():
    for _ in range(100):
        yield [np.random.randn(1, 224, 224, 3).astype(np.float32)]

tflite_model = tf_quantizer.post_training_quantization(
    keras_model,
    representative_dataset=representative_dataset
)

# Save quantized model
with open('quantized_model.tflite', 'wb') as f:
    f.write(tflite_model)
""")

    print("\n4. ONNX Optimization")
    print("-" * 70)
    print("""
optimizer = ONNXOptimizer()
optimizer.optimize(
    'model.onnx',
    'model_optimized.onnx',
    optimization_level='extended'
)
""")

    print("\n" + "=" * 70)
    print("Model optimization examples completed!")
    print("=" * 70)
