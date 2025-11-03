"""
RT-DETR TensorRT Engine Builder
Builds optimized TensorRT engines from ONNX models for Jetson inference
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logging.warning("TensorRT not available. Install with: pip install tensorrt pycuda")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RTDETRTensorRTBuilder:
    """
    Builds TensorRT engines from ONNX models with optimization for Jetson
    """

    def __init__(
        self,
        onnx_path: str,
        engine_path: Optional[str] = None,
        precision: str = "fp16",  # fp32, fp16, int8
        max_batch_size: int = 1,
        workspace_size: int = 1 << 30,  # 1GB
        verbose: bool = False
    ):
        """
        Initialize TensorRT builder

        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save TensorRT engine (auto-generated if None)
            precision: Precision mode (fp32, fp16, int8)
            max_batch_size: Maximum batch size
            workspace_size: Maximum workspace size in bytes
            verbose: Enable verbose logging
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT is not available")

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Auto-generate engine path
        if engine_path is None:
            engine_path = self.onnx_path.with_suffix(f".{precision}.engine")
        self.engine_path = Path(engine_path)

        self.precision = precision.lower()
        self.max_batch_size = max_batch_size
        self.workspace_size = workspace_size
        self.verbose = verbose

        # TensorRT logger
        self.trt_logger = trt.Logger(
            trt.Logger.VERBOSE if verbose else trt.Logger.INFO
        )

        logger.info(f"Initialized TensorRT builder")
        logger.info(f"  ONNX model: {self.onnx_path}")
        logger.info(f"  Engine path: {self.engine_path}")
        logger.info(f"  Precision: {self.precision}")
        logger.info(f"  Max batch size: {self.max_batch_size}")

    def build_engine(self) -> bool:
        """
        Build TensorRT engine from ONNX model

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Building TensorRT engine...")

            # Create builder and network
            builder = trt.Builder(self.trt_logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, self.trt_logger)

            # Parse ONNX model
            logger.info(f"Parsing ONNX model: {self.onnx_path}")
            with open(self.onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return False

            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = self.workspace_size

            # Set precision
            if self.precision == "fp16":
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    logger.info("FP16 mode enabled")
                else:
                    logger.warning("FP16 not supported, using FP32")
            elif self.precision == "int8":
                if builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    logger.info("INT8 mode enabled")
                    # Note: INT8 calibration required for production
                else:
                    logger.warning("INT8 not supported, using FP32")

            # Set optimization profile for dynamic shapes
            profile = builder.create_optimization_profile()

            # Get input tensor name and shape
            input_tensor = network.get_input(0)
            input_name = input_tensor.name
            input_shape = input_tensor.shape

            logger.info(f"Input tensor: {input_name}, shape: {input_shape}")

            # Set dynamic shape ranges
            # min_shape, opt_shape, max_shape
            if input_shape[0] == -1:  # Dynamic batch size
                min_shape = (1, *input_shape[1:])
                opt_shape = (self.max_batch_size // 2, *input_shape[1:])
                max_shape = (self.max_batch_size, *input_shape[1:])
            else:
                min_shape = tuple(input_shape)
                opt_shape = tuple(input_shape)
                max_shape = tuple(input_shape)

            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

            # Build engine
            logger.info("Building TensorRT engine (this may take several minutes)...")
            engine = builder.build_engine(network, config)

            if engine is None:
                logger.error("Failed to build engine")
                return False

            # Serialize and save engine
            logger.info(f"Saving engine to: {self.engine_path}")
            with open(self.engine_path, 'wb') as f:
                f.write(engine.serialize())

            logger.info("✓ TensorRT engine built successfully!")
            return True

        except Exception as e:
            logger.error(f"Error building engine: {e}", exc_info=True)
            return False

    def load_engine(self) -> Optional[trt.ICudaEngine]:
        """
        Load pre-built TensorRT engine

        Returns:
            TensorRT engine or None if failed
        """
        try:
            if not self.engine_path.exists():
                logger.error(f"Engine file not found: {self.engine_path}")
                return None

            logger.info(f"Loading TensorRT engine: {self.engine_path}")

            runtime = trt.Runtime(self.trt_logger)
            with open(self.engine_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())

            if engine is None:
                logger.error("Failed to load engine")
                return None

            logger.info("✓ Engine loaded successfully")
            self._print_engine_info(engine)
            return engine

        except Exception as e:
            logger.error(f"Error loading engine: {e}", exc_info=True)
            return None

    def _print_engine_info(self, engine: trt.ICudaEngine):
        """Print engine information"""
        logger.info("Engine Information:")
        logger.info(f"  Max batch size: {engine.max_batch_size}")
        logger.info(f"  Number of bindings: {engine.num_bindings}")

        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            is_input = engine.binding_is_input(i)
            logger.info(f"  Binding {i}: {name}")
            logger.info(f"    Shape: {shape}")
            logger.info(f"    Type: {dtype}")
            logger.info(f"    Is input: {is_input}")

    @staticmethod
    def convert_onnx_to_tensorrt(
        onnx_path: str,
        engine_path: Optional[str] = None,
        precision: str = "fp16",
        max_batch_size: int = 1,
        force_rebuild: bool = False
    ) -> Optional[str]:
        """
        Convenience method to convert ONNX to TensorRT engine

        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save engine
            precision: Precision mode
            max_batch_size: Maximum batch size
            force_rebuild: Force rebuild even if engine exists

        Returns:
            Path to engine file or None if failed
        """
        builder = RTDETRTensorRTBuilder(
            onnx_path=onnx_path,
            engine_path=engine_path,
            precision=precision,
            max_batch_size=max_batch_size
        )

        # Check if engine already exists
        if builder.engine_path.exists() and not force_rebuild:
            logger.info(f"Engine already exists: {builder.engine_path}")
            return str(builder.engine_path)

        # Build engine
        if builder.build_engine():
            return str(builder.engine_path)
        return None


class TensorRTInferenceContext:
    """
    TensorRT inference execution context with memory management
    """

    def __init__(self, engine: trt.ICudaEngine):
        """
        Initialize inference context

        Args:
            engine: TensorRT engine
        """
        self.engine = engine
        self.context = engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        # Allocate device memory for all bindings
        for i in range(engine.num_bindings):
            binding_name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            size = trt.volume(shape)

            # Allocate device memory
            device_mem = cuda.mem_alloc(size * dtype().itemsize)

            self.bindings.append(int(device_mem))

            if engine.binding_is_input(i):
                self.inputs.append({
                    'name': binding_name,
                    'shape': shape,
                    'dtype': dtype,
                    'device_mem': device_mem,
                    'size': size
                })
            else:
                self.outputs.append({
                    'name': binding_name,
                    'shape': shape,
                    'dtype': dtype,
                    'device_mem': device_mem,
                    'size': size
                })

        logger.info(f"Allocated {len(self.inputs)} input buffers")
        logger.info(f"Allocated {len(self.outputs)} output buffers")

    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """
        Run inference on input data

        Args:
            input_data: Input numpy array

        Returns:
            List of output numpy arrays
        """
        # Copy input to device
        cuda.memcpy_htod_async(
            self.inputs[0]['device_mem'],
            input_data.astype(self.inputs[0]['dtype']).ravel(),
            self.stream
        )

        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Copy outputs from device
        outputs = []
        for output in self.outputs:
            host_mem = np.empty(output['size'], dtype=output['dtype'])
            cuda.memcpy_dtoh_async(host_mem, output['device_mem'], self.stream)
            outputs.append(host_mem.reshape(output['shape']))

        # Synchronize stream
        self.stream.synchronize()

        return outputs

    def __del__(self):
        """Cleanup resources"""
        # CUDA memory is automatically freed
        pass


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Build RT-DETR TensorRT engine")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--output", help="Output engine path")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    builder = RTDETRTensorRTBuilder(
        onnx_path=args.onnx,
        engine_path=args.output,
        precision=args.precision,
        max_batch_size=args.batch_size,
        verbose=args.verbose
    )

    if builder.build_engine():
        print(f"✓ Engine saved to: {builder.engine_path}")
    else:
        print("✗ Failed to build engine")
        exit(1)
