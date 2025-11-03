"""
Model Export Module

Utilities for exporting models to various formats for deployment:
- ONNX (Open Neural Network Exchange)
- TorchScript (PyTorch)
- SavedModel (TensorFlow)
- Pickle/Joblib (scikit-learn)
- Model metadata and versioning

Author: ML Framework Team
"""

import numpy as np
import os
import json
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import pickle
import warnings


# ============================================================================
# ONNX EXPORT
# ============================================================================

class ONNXExporter:
    """
    Export PyTorch models to ONNX format for cross-platform inference.
    """

    def __init__(self):
        """Initialize ONNX exporter."""
        self.onnx_available = self._check_onnx()

    def _check_onnx(self) -> bool:
        """Check if ONNX is available."""
        try:
            import torch
            import onnx
            return True
        except ImportError:
            warnings.warn("ONNX or PyTorch not available. Install with: pip install torch onnx")
            return False

    def export(
        self,
        model: Any,
        dummy_input: Any,
        export_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict] = None,
        opset_version: int = 13,
        do_constant_folding: bool = True
    ) -> bool:
        """
        Export PyTorch model to ONNX format.

        Parameters:
        -----------
        model : torch.nn.Module
            PyTorch model to export.
        dummy_input : torch.Tensor or tuple
            Example input for tracing.
        export_path : str
            Path to save ONNX model.
        input_names : list of str, optional
            Names for input tensors.
        output_names : list of str, optional
            Names for output tensors.
        dynamic_axes : dict, optional
            Dynamic axes specification (e.g., for batch size).
        opset_version : int
            ONNX opset version.
        do_constant_folding : bool
            Optimize constant folding.

        Returns:
        --------
        success : bool
            True if export successful.
        """
        if not self.onnx_available:
            print("ONNX not available")
            return False

        try:
            import torch

            # Set model to eval mode
            model.eval()

            # Default names
            if input_names is None:
                input_names = ['input']
            if output_names is None:
                output_names = ['output']

            # Default dynamic axes (batch size)
            if dynamic_axes is None:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }

            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=do_constant_folding,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )

            print(f"Model exported to ONNX: {export_path}")

            # Verify the model
            self.verify(export_path)

            return True

        except Exception as e:
            print(f"Error exporting to ONNX: {str(e)}")
            return False

    def verify(self, onnx_path: str) -> bool:
        """
        Verify exported ONNX model.

        Parameters:
        -----------
        onnx_path : str
            Path to ONNX model.

        Returns:
        --------
        is_valid : bool
            True if model is valid.
        """
        try:
            import onnx

            # Load model
            onnx_model = onnx.load(onnx_path)

            # Check model
            onnx.checker.check_model(onnx_model)

            print(f"ONNX model is valid: {onnx_path}")
            return True

        except Exception as e:
            print(f"ONNX verification failed: {str(e)}")
            return False

    def get_model_info(self, onnx_path: str) -> Dict[str, Any]:
        """
        Get information about ONNX model.

        Parameters:
        -----------
        onnx_path : str
            Path to ONNX model.

        Returns:
        --------
        info : dict
            Model information.
        """
        try:
            import onnx

            model = onnx.load(onnx_path)

            # Input/output info
            inputs = []
            for input_tensor in model.graph.input:
                shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
                inputs.append({
                    'name': input_tensor.name,
                    'shape': shape,
                    'dtype': input_tensor.type.tensor_type.elem_type
                })

            outputs = []
            for output_tensor in model.graph.output:
                shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
                outputs.append({
                    'name': output_tensor.name,
                    'shape': shape,
                    'dtype': output_tensor.type.tensor_type.elem_type
                })

            return {
                'producer': model.producer_name,
                'opset_version': model.opset_import[0].version,
                'inputs': inputs,
                'outputs': outputs,
                'n_nodes': len(model.graph.node)
            }

        except Exception as e:
            print(f"Error getting model info: {str(e)}")
            return {}


# ============================================================================
# TORCHSCRIPT EXPORT
# ============================================================================

class TorchScriptExporter:
    """
    Export PyTorch models to TorchScript for production deployment.
    """

    def __init__(self):
        """Initialize TorchScript exporter."""
        self.torch_available = self._check_torch()

    def _check_torch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            warnings.warn("PyTorch not available. Install with: pip install torch")
            return False

    def export_trace(
        self,
        model: Any,
        example_inputs: Any,
        export_path: str,
        strict: bool = True
    ) -> bool:
        """
        Export model using tracing (records operations during forward pass).

        Parameters:
        -----------
        model : torch.nn.Module
            Model to export.
        example_inputs : torch.Tensor or tuple
            Example inputs for tracing.
        export_path : str
            Path to save TorchScript model.
        strict : bool
            Strict tracing mode.

        Returns:
        --------
        success : bool
        """
        if not self.torch_available:
            return False

        try:
            import torch

            # Set model to eval mode
            model.eval()

            # Trace model
            traced_model = torch.jit.trace(model, example_inputs, strict=strict)

            # Save
            traced_model.save(export_path)

            print(f"Model traced and saved to: {export_path}")
            return True

        except Exception as e:
            print(f"Error tracing model: {str(e)}")
            return False

    def export_script(
        self,
        model: Any,
        export_path: str
    ) -> bool:
        """
        Export model using scripting (compiles model code directly).

        Better for models with control flow (if/else, loops).

        Parameters:
        -----------
        model : torch.nn.Module
            Model to export.
        export_path : str
            Path to save TorchScript model.

        Returns:
        --------
        success : bool
        """
        if not self.torch_available:
            return False

        try:
            import torch

            # Set model to eval mode
            model.eval()

            # Script model
            scripted_model = torch.jit.script(model)

            # Save
            scripted_model.save(export_path)

            print(f"Model scripted and saved to: {export_path}")
            return True

        except Exception as e:
            print(f"Error scripting model: {str(e)}")
            return False

    def optimize_for_mobile(
        self,
        torchscript_path: str,
        output_path: str
    ) -> bool:
        """
        Optimize TorchScript model for mobile deployment.

        Parameters:
        -----------
        torchscript_path : str
            Path to TorchScript model.
        output_path : str
            Path to save optimized model.

        Returns:
        --------
        success : bool
        """
        try:
            import torch
            from torch.utils.mobile_optimizer import optimize_for_mobile

            # Load model
            model = torch.jit.load(torchscript_path)

            # Optimize
            optimized_model = optimize_for_mobile(model)

            # Save
            optimized_model._save_for_lite_interpreter(output_path)

            print(f"Model optimized for mobile: {output_path}")
            return True

        except Exception as e:
            print(f"Error optimizing for mobile: {str(e)}")
            return False


# ============================================================================
# TENSORFLOW EXPORT
# ============================================================================

class TensorFlowExporter:
    """
    Export TensorFlow/Keras models to SavedModel format.
    """

    def __init__(self):
        """Initialize TensorFlow exporter."""
        self.tf_available = self._check_tf()

    def _check_tf(self) -> bool:
        """Check if TensorFlow is available."""
        try:
            import tensorflow as tf
            return True
        except ImportError:
            warnings.warn("TensorFlow not available. Install with: pip install tensorflow")
            return False

    def export_savedmodel(
        self,
        model: Any,
        export_path: str,
        signatures: Optional[Any] = None
    ) -> bool:
        """
        Export TensorFlow/Keras model to SavedModel format.

        Parameters:
        -----------
        model : tf.keras.Model or tf.Module
            Model to export.
        export_path : str
            Path to save model.
        signatures : ConcreteFunction, optional
            Serving signatures.

        Returns:
        --------
        success : bool
        """
        if not self.tf_available:
            return False

        try:
            import tensorflow as tf

            # Save model
            tf.saved_model.save(model, export_path, signatures=signatures)

            print(f"Model saved to SavedModel format: {export_path}")
            return True

        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def export_tflite(
        self,
        model: Any,
        export_path: str,
        quantize: bool = False
    ) -> bool:
        """
        Export model to TensorFlow Lite format for mobile/edge devices.

        Parameters:
        -----------
        model : tf.keras.Model
            Model to export.
        export_path : str
            Path to save .tflite model.
        quantize : bool
            Apply quantization for smaller model size.

        Returns:
        --------
        success : bool
        """
        if not self.tf_available:
            return False

        try:
            import tensorflow as tf

            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            tflite_model = converter.convert()

            # Save
            with open(export_path, 'wb') as f:
                f.write(tflite_model)

            print(f"Model saved to TFLite format: {export_path}")
            return True

        except Exception as e:
            print(f"Error converting to TFLite: {str(e)}")
            return False


# ============================================================================
# SKLEARN / GENERIC MODEL EXPORT
# ============================================================================

class SklearnExporter:
    """
    Export scikit-learn and generic Python models.
    """

    @staticmethod
    def export_pickle(
        model: Any,
        export_path: str,
        protocol: int = pickle.HIGHEST_PROTOCOL
    ) -> bool:
        """
        Export model using pickle.

        Parameters:
        -----------
        model : Any
            Model to export.
        export_path : str
            Path to save model.
        protocol : int
            Pickle protocol version.

        Returns:
        --------
        success : bool
        """
        try:
            with open(export_path, 'wb') as f:
                pickle.dump(model, f, protocol=protocol)

            print(f"Model saved with pickle: {export_path}")
            return True

        except Exception as e:
            print(f"Error saving model with pickle: {str(e)}")
            return False

    @staticmethod
    def export_joblib(
        model: Any,
        export_path: str,
        compress: Union[int, bool, str] = 3
    ) -> bool:
        """
        Export model using joblib (more efficient for large numpy arrays).

        Parameters:
        -----------
        model : Any
            Model to export.
        export_path : str
            Path to save model.
        compress : int, bool, or str
            Compression level (0-9) or algorithm.

        Returns:
        --------
        success : bool
        """
        try:
            import joblib

            joblib.dump(model, export_path, compress=compress)

            print(f"Model saved with joblib: {export_path}")
            return True

        except ImportError:
            print("joblib not available. Install with: pip install joblib")
            return False
        except Exception as e:
            print(f"Error saving model with joblib: {str(e)}")
            return False

    @staticmethod
    def load_pickle(model_path: str) -> Any:
        """Load model from pickle file."""
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_joblib(model_path: str) -> Any:
        """Load model from joblib file."""
        import joblib
        return joblib.load(model_path)


# ============================================================================
# MODEL METADATA AND VERSIONING
# ============================================================================

class ModelMetadata:
    """
    Manage model metadata and versioning.
    """

    def __init__(self):
        """Initialize metadata manager."""
        pass

    @staticmethod
    def create_metadata(
        model_name: str,
        version: str,
        framework: str,
        model_type: str,
        input_shape: List[int],
        output_shape: List[int],
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_data: Optional[Dict[str, Any]] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create model metadata.

        Parameters:
        -----------
        model_name : str
            Model name.
        version : str
            Model version.
        framework : str
            Framework (pytorch, tensorflow, sklearn, etc.).
        model_type : str
            Model type (classifier, regressor, etc.).
        input_shape : list
            Input shape.
        output_shape : list
            Output shape.
        metrics : dict, optional
            Performance metrics.
        hyperparameters : dict, optional
            Model hyperparameters.
        training_data : dict, optional
            Training data information.
        additional_info : dict, optional
            Additional metadata.

        Returns:
        --------
        metadata : dict
            Model metadata.
        """
        metadata = {
            'model_name': model_name,
            'version': version,
            'framework': framework,
            'model_type': model_type,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'created_at': str(np.datetime64('now')),
        }

        if metrics:
            metadata['metrics'] = metrics

        if hyperparameters:
            metadata['hyperparameters'] = hyperparameters

        if training_data:
            metadata['training_data'] = training_data

        if additional_info:
            metadata.update(additional_info)

        return metadata

    @staticmethod
    def save_metadata(
        metadata: Dict[str, Any],
        save_path: str
    ) -> bool:
        """
        Save metadata to JSON file.

        Parameters:
        -----------
        metadata : dict
            Metadata dictionary.
        save_path : str
            Path to save metadata.

        Returns:
        --------
        success : bool
        """
        try:
            with open(save_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Metadata saved to: {save_path}")
            return True

        except Exception as e:
            print(f"Error saving metadata: {str(e)}")
            return False

    @staticmethod
    def load_metadata(metadata_path: str) -> Dict[str, Any]:
        """
        Load metadata from JSON file.

        Parameters:
        -----------
        metadata_path : str
            Path to metadata file.

        Returns:
        --------
        metadata : dict
        """
        with open(metadata_path, 'r') as f:
            return json.load(f)


# ============================================================================
# UNIFIED MODEL EXPORTER
# ============================================================================

class ModelExporter:
    """
    Unified interface for exporting models to various formats.
    """

    def __init__(self):
        """Initialize model exporter."""
        self.onnx_exporter = ONNXExporter()
        self.torchscript_exporter = TorchScriptExporter()
        self.tf_exporter = TensorFlowExporter()
        self.sklearn_exporter = SklearnExporter()
        self.metadata_manager = ModelMetadata()

    def export(
        self,
        model: Any,
        export_format: str,
        export_path: str,
        **kwargs
    ) -> bool:
        """
        Export model to specified format.

        Parameters:
        -----------
        model : Any
            Model to export.
        export_format : str
            Export format: 'onnx', 'torchscript', 'savedmodel', 'tflite', 'pickle', 'joblib'.
        export_path : str
            Path to save model.
        **kwargs : dict
            Additional arguments for specific exporters.

        Returns:
        --------
        success : bool
        """
        export_format = export_format.lower()

        if export_format == 'onnx':
            return self.onnx_exporter.export(model, export_path=export_path, **kwargs)

        elif export_format == 'torchscript':
            method = kwargs.pop('method', 'trace')
            if method == 'trace':
                return self.torchscript_exporter.export_trace(model, export_path=export_path, **kwargs)
            else:
                return self.torchscript_exporter.export_script(model, export_path)

        elif export_format == 'savedmodel':
            return self.tf_exporter.export_savedmodel(model, export_path)

        elif export_format == 'tflite':
            return self.tf_exporter.export_tflite(model, export_path, **kwargs)

        elif export_format == 'pickle':
            return self.sklearn_exporter.export_pickle(model, export_path, **kwargs)

        elif export_format == 'joblib':
            return self.sklearn_exporter.export_joblib(model, export_path, **kwargs)

        else:
            print(f"Unknown export format: {export_format}")
            return False


# ============================================================================
# EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MODEL EXPORT EXAMPLES")
    print("=" * 70)

    # Note: These examples require the respective libraries to be installed

    print("\n1. ONNX Export (requires torch and onnx)")
    print("-" * 70)
    print("Example code:")
    print("""
import torch
import torch.nn as nn

# Define model
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)

# Dummy input
dummy_input = torch.randn(1, 10)

# Export to ONNX
exporter = ONNXExporter()
exporter.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
""")

    print("\n2. TorchScript Export")
    print("-" * 70)
    print("Example code:")
    print("""
# Export using tracing
exporter = TorchScriptExporter()
exporter.export_trace(model, dummy_input, 'model_traced.pt')

# Export using scripting (for control flow)
exporter.export_script(model, 'model_scripted.pt')
""")

    print("\n3. Scikit-learn Export")
    print("-" * 70)
    print("Example code:")
    print("""
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Export
exporter = SklearnExporter()
exporter.export_joblib(model, 'model.joblib', compress=3)

# Load
loaded_model = exporter.load_joblib('model.joblib')
""")

    print("\n4. Model Metadata")
    print("-" * 70)
    print("Example code:")
    print("""
metadata_manager = ModelMetadata()

metadata = metadata_manager.create_metadata(
    model_name='my_classifier',
    version='1.0.0',
    framework='pytorch',
    model_type='classifier',
    input_shape=[1, 10],
    output_shape=[1, 2],
    metrics={'accuracy': 0.95, 'f1': 0.93},
    hyperparameters={'learning_rate': 0.001, 'epochs': 100}
)

metadata_manager.save_metadata(metadata, 'model_metadata.json')
""")

    print("\n5. Unified Exporter")
    print("-" * 70)
    print("Example code:")
    print("""
exporter = ModelExporter()

# Export to different formats
exporter.export(model, 'onnx', 'model.onnx', dummy_input=dummy_input)
exporter.export(model, 'torchscript', 'model.pt', example_inputs=dummy_input, method='trace')
exporter.export(sklearn_model, 'joblib', 'model.joblib')
""")

    print("\n" + "=" * 70)
    print("Model export examples completed!")
    print("=" * 70)
