"""
Model Manager for Jetson AGX Orin.
Handles model loading, execution, and OTA updates.
"""
import os
import json
import logging
import pickle
import time
from typing import Dict, Optional, Any
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages ML models on Jetson device.
    Supports PyTorch, TensorFlow, ONNX, and pickle models.
    """

    def __init__(self, models_dir: str = "/opt/ml-platform/models"):
        """
        Initialize Model Manager.

        Args:
            models_dir: Directory to store models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.active_model = None
        self.active_model_version = None
        self.active_model_id = None
        self.model_metadata = {}

        logger.info(f"Model Manager initialized. Models directory: {self.models_dir}")

    def load_model(
        self,
        model_path: str,
        model_id: int,
        model_version: str,
        model_type: str = "auto"
    ) -> bool:
        """
        Load a model from file.

        Args:
            model_path: Path to model file
            model_id: Model/formula ID
            model_version: Model version
            model_type: Model type (pytorch, tensorflow, onnx, pickle, auto)

        Returns:
            True if loaded successfully
        """
        try:
            logger.info(f"Loading model: {model_path} (type: {model_type})")

            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            # Auto-detect model type
            if model_type == "auto":
                model_type = self._detect_model_type(model_path)
                logger.info(f"Auto-detected model type: {model_type}")

            # Load based on type
            if model_type == "pytorch":
                self.active_model = self._load_pytorch_model(model_path)
            elif model_type == "tensorflow":
                self.active_model = self._load_tensorflow_model(model_path)
            elif model_type == "onnx":
                self.active_model = self._load_onnx_model(model_path)
            elif model_type == "pickle":
                self.active_model = self._load_pickle_model(model_path)
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return False

            if self.active_model is None:
                logger.error("Model loading failed")
                return False

            # Update state
            self.active_model_id = model_id
            self.active_model_version = model_version
            self.model_metadata = {
                'model_id': model_id,
                'model_version': model_version,
                'model_type': model_type,
                'model_path': str(model_path),
                'loaded_at': time.time()
            }

            logger.info(f"Model loaded successfully: {model_version}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return False

    def _detect_model_type(self, model_path: Path) -> str:
        """Detect model type from file extension."""
        suffix = model_path.suffix.lower()

        if suffix in ['.pt', '.pth']:
            return 'pytorch'
        elif suffix in ['.h5', '.pb', '.keras']:
            return 'tensorflow'
        elif suffix == '.onnx':
            return 'onnx'
        elif suffix in ['.pkl', '.pickle']:
            return 'pickle'
        else:
            return 'unknown'

    def _load_pytorch_model(self, model_path: Path):
        """Load PyTorch model."""
        try:
            import torch
            model = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            if hasattr(model, 'eval'):
                model.eval()
            logger.info("PyTorch model loaded")
            return model
        except ImportError:
            logger.error("PyTorch not installed")
            return None
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            return None

    def _load_tensorflow_model(self, model_path: Path):
        """Load TensorFlow model."""
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(str(model_path))
            logger.info("TensorFlow model loaded")
            return model
        except ImportError:
            logger.error("TensorFlow not installed")
            return None
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {e}")
            return None

    def _load_onnx_model(self, model_path: Path):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(
                str(model_path),
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            logger.info("ONNX model loaded")
            return session
        except ImportError:
            logger.error("ONNX Runtime not installed")
            return None
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            return None

    def _load_pickle_model(self, model_path: Path):
        """Load pickled model (scikit-learn, etc.)."""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Pickle model loaded")
            return model
        except Exception as e:
            logger.error(f"Error loading pickle model: {e}")
            return None

    def execute(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute the active model with input data.

        Args:
            input_data: Input dictionary

        Returns:
            Output dictionary or None on error
        """
        if self.active_model is None:
            logger.error("No active model loaded")
            return None

        try:
            start_time = time.time()

            # Execute based on model type
            model_type = self.model_metadata.get('model_type', 'unknown')

            if model_type == 'pytorch':
                output = self._execute_pytorch(input_data)
            elif model_type == 'tensorflow':
                output = self._execute_tensorflow(input_data)
            elif model_type == 'onnx':
                output = self._execute_onnx(input_data)
            elif model_type == 'pickle':
                output = self._execute_pickle(input_data)
            else:
                logger.error(f"Cannot execute unknown model type: {model_type}")
                return None

            execution_time_ms = (time.time() - start_time) * 1000

            logger.info(f"Model executed in {execution_time_ms:.2f}ms")

            return {
                'output': output,
                'execution_time_ms': execution_time_ms,
                'model_version': self.active_model_version
            }

        except Exception as e:
            logger.error(f"Model execution error: {e}", exc_info=True)
            return None

    def _execute_pytorch(self, input_data: Dict) -> Any:
        """Execute PyTorch model."""
        import torch

        # Convert input to tensor
        if isinstance(input_data, dict):
            # Assume single input key or 'data' key
            data = input_data.get('data', list(input_data.values())[0])
        else:
            data = input_data

        input_tensor = torch.tensor(data, dtype=torch.float32)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        with torch.no_grad():
            output = self.active_model(input_tensor)

        # Convert back to numpy/list
        if hasattr(output, 'cpu'):
            output = output.cpu().numpy().tolist()

        return output

    def _execute_tensorflow(self, input_data: Dict) -> Any:
        """Execute TensorFlow model."""
        import numpy as np

        if isinstance(input_data, dict):
            data = input_data.get('data', list(input_data.values())[0])
        else:
            data = input_data

        input_array = np.array(data, dtype=np.float32)
        output = self.active_model.predict(input_array)

        return output.tolist()

    def _execute_onnx(self, input_data: Dict) -> Any:
        """Execute ONNX model."""
        import numpy as np

        if isinstance(input_data, dict):
            data = input_data.get('data', list(input_data.values())[0])
        else:
            data = input_data

        input_array = np.array(data, dtype=np.float32)

        # Get input name
        input_name = self.active_model.get_inputs()[0].name

        output = self.active_model.run(None, {input_name: input_array})

        return output[0].tolist() if output else None

    def _execute_pickle(self, input_data: Dict) -> Any:
        """Execute pickled model (scikit-learn, etc.)."""
        import numpy as np

        if isinstance(input_data, dict):
            data = input_data.get('data', list(input_data.values())[0])
        else:
            data = input_data

        input_array = np.array(data)

        # Assume model has predict method
        if hasattr(self.active_model, 'predict'):
            output = self.active_model.predict(input_array)
        elif callable(self.active_model):
            output = self.active_model(input_array)
        else:
            raise ValueError("Model does not have predict method or callable")

        return output.tolist() if hasattr(output, 'tolist') else output

    def save_model_metadata(self, metadata: Dict, filename: str = "model_metadata.json"):
        """Save model metadata to file."""
        metadata_path = self.models_dir / filename
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Model metadata saved: {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving model metadata: {e}")

    def load_model_metadata(self, filename: str = "model_metadata.json") -> Optional[Dict]:
        """Load model metadata from file."""
        metadata_path = self.models_dir / filename
        try:
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading model metadata: {e}")
        return None

    def get_model_checksum(self, model_path: str) -> str:
        """Calculate MD5 checksum of model file."""
        md5_hash = hashlib.md5()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def cleanup_old_models(self, keep_latest: int = 3):
        """Delete old model files, keeping only the latest N."""
        try:
            model_files = sorted(
                self.models_dir.glob("*.pt") |
                self.models_dir.glob("*.pth") |
                self.models_dir.glob("*.h5") |
                self.models_dir.glob("*.onnx") |
                self.models_dir.glob("*.pkl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            # Delete old files
            for old_file in model_files[keep_latest:]:
                logger.info(f"Deleting old model: {old_file}")
                old_file.unlink()

        except Exception as e:
            logger.error(f"Error cleaning up old models: {e}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    manager = ModelManager()

    print("Model Manager initialized")
    print(f"Models directory: {manager.models_dir}")
