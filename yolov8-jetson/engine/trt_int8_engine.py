"""
TensorRT INT8 Quantization Engine Builder
Builds optimized INT8 engines with calibration for maximum FPS
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logging.warning("TensorRT not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class INT8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 Entropy Calibrator for TensorRT quantization
    Uses calibration dataset to determine optimal quantization scales
    """

    def __init__(
        self,
        calibration_images: List[np.ndarray],
        cache_file: str = "calibration.cache",
        batch_size: int = 1
    ):
        """
        Initialize INT8 calibrator

        Args:
            calibration_images: List of calibration images (preprocessed)
            cache_file: Path to cache file
            batch_size: Calibration batch size
        """
        super().__init__()

        self.calibration_images = calibration_images
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0

        # Get input shape from first image
        self.input_shape = calibration_images[0].shape

        # Allocate device memory for one batch
        self.device_input = cuda.mem_alloc(
            self.batch_size * np.prod(self.input_shape) * np.dtype(np.float32).itemsize
        )

        logger.info(f"INT8 Calibrator initialized")
        logger.info(f"  Calibration images: {len(calibration_images)}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Cache file: {cache_file}")

    def get_batch_size(self) -> int:
        """Return batch size"""
        return self.batch_size

    def get_batch(self, names: List[str]) -> List[int]:
        """
        Get next calibration batch

        Args:
            names: List of input tensor names

        Returns:
            List of device memory pointers
        """
        if self.current_index >= len(self.calibration_images):
            return None

        # Get batch
        batch = []
        for i in range(self.batch_size):
            if self.current_index < len(self.calibration_images):
                batch.append(self.calibration_images[self.current_index])
                self.current_index += 1
            else:
                # Pad with last image if needed
                batch.append(self.calibration_images[-1])

        # Stack batch
        batch = np.stack(batch, axis=0).astype(np.float32).ravel()

        # Copy to device
        cuda.memcpy_htod(self.device_input, batch)

        return [int(self.device_input)]

    def read_calibration_cache(self) -> bytes:
        """Read calibration cache if exists"""
        if os.path.exists(self.cache_file):
            logger.info(f"Reading calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes):
        """Write calibration cache"""
        logger.info(f"Writing calibration cache: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class TensorRTINT8Builder:
    """
    TensorRT Engine Builder with INT8 quantization support
    Optimized for maximum FPS on Jetson
    """

    def __init__(
        self,
        onnx_path: str,
        calibration_data: Optional[List[np.ndarray]] = None,
        cache_file: str = "calibration.cache",
        workspace_size: int = 2 << 30,  # 2GB
        max_batch_size: int = 1,
        fp16_mode: bool = False,
        int8_mode: bool = True,
        strict_types: bool = False,
        verbose: bool = False
    ):
        """
        Initialize TensorRT INT8 builder

        Args:
            onnx_path: Path to ONNX model
            calibration_data: List of calibration images for INT8
            cache_file: Calibration cache file
            workspace_size: Maximum workspace size in bytes
            max_batch_size: Maximum batch size
            fp16_mode: Enable FP16 (if INT8 not available)
            int8_mode: Enable INT8 quantization
            strict_types: Use strict type constraints
            verbose: Verbose logging
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT not available")

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self.calibration_data = calibration_data
        self.cache_file = cache_file
        self.workspace_size = workspace_size
        self.max_batch_size = max_batch_size
        self.fp16_mode = fp16_mode
        self.int8_mode = int8_mode
        self.strict_types = strict_types

        # TensorRT logger
        self.trt_logger = trt.Logger(
            trt.Logger.VERBOSE if verbose else trt.Logger.INFO
        )

        logger.info(f"Initialized TensorRT INT8 builder")
        logger.info(f"  ONNX: {self.onnx_path}")
        logger.info(f"  INT8 mode: {int8_mode}")
        logger.info(f"  FP16 mode: {fp16_mode}")
        logger.info(f"  Max batch: {max_batch_size}")
        logger.info(f"  Workspace: {workspace_size / (1024**3):.2f} GB")

    def build_engine(self, engine_path: str) -> bool:
        """
        Build TensorRT engine with INT8 quantization

        Args:
            engine_path: Path to save engine

        Returns:
            True if successful
        """
        try:
            logger.info("Building TensorRT INT8 engine...")

            # Create builder and network
            builder = trt.Builder(self.trt_logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, self.trt_logger)

            # Parse ONNX
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

            # Set precision modes
            if self.int8_mode:
                if not builder.platform_has_fast_int8:
                    logger.warning("INT8 not supported, falling back to FP16")
                    self.int8_mode = False
                    self.fp16_mode = True

            if self.int8_mode:
                logger.info("Enabling INT8 precision")
                config.set_flag(trt.BuilderFlag.INT8)

                # Set calibrator if provided
                if self.calibration_data:
                    logger.info("Setting up INT8 calibrator...")
                    calibrator = INT8Calibrator(
                        calibration_images=self.calibration_data,
                        cache_file=self.cache_file,
                        batch_size=self.max_batch_size
                    )
                    config.int8_calibrator = calibrator
                elif not os.path.exists(self.cache_file):
                    logger.error("INT8 requires calibration data or cache file")
                    return False
                else:
                    logger.info(f"Using existing calibration cache: {self.cache_file}")

            if self.fp16_mode:
                if builder.platform_has_fast_fp16:
                    logger.info("Enabling FP16 precision")
                    config.set_flag(trt.BuilderFlag.FP16)
                else:
                    logger.warning("FP16 not supported")

            # Strict type constraints (for better performance)
            if self.strict_types:
                config.set_flag(trt.BuilderFlag.STRICT_TYPES)

            # Optimization profile for dynamic shapes
            profile = builder.create_optimization_profile()

            # Get input tensor
            input_tensor = network.get_input(0)
            input_name = input_tensor.name
            input_shape = input_tensor.shape

            logger.info(f"Input tensor: {input_name}")
            logger.info(f"  Shape: {input_shape}")

            # Set shape ranges
            if input_shape[0] == -1:  # Dynamic batch
                min_shape = (1, *input_shape[1:])
                opt_shape = (self.max_batch_size, *input_shape[1:])
                max_shape = (self.max_batch_size, *input_shape[1:])
            else:
                min_shape = tuple(input_shape)
                opt_shape = tuple(input_shape)
                max_shape = tuple(input_shape)

            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

            # Build engine
            logger.info("Building engine (this may take 10-30 minutes for INT8)...")
            engine = builder.build_engine(network, config)

            if engine is None:
                logger.error("Failed to build engine")
                return False

            # Serialize and save
            logger.info(f"Saving engine to: {engine_path}")
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())

            # Print engine info
            self._print_engine_info(engine)

            logger.info("✓ INT8 engine built successfully!")
            return True

        except Exception as e:
            logger.error(f"Error building engine: {e}", exc_info=True)
            return False

    def _print_engine_info(self, engine: trt.ICudaEngine):
        """Print engine information"""
        logger.info("\nEngine Information:")
        logger.info(f"  Max batch size: {engine.max_batch_size}")
        logger.info(f"  Device memory: {engine.device_memory_size / (1024**2):.2f} MB")
        logger.info(f"  Number of bindings: {engine.num_bindings}")

        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = engine.get_binding_dtype(i)
            is_input = engine.binding_is_input(i)

            logger.info(f"\n  Binding {i}: {name}")
            logger.info(f"    Shape: {shape}")
            logger.info(f"    Type: {dtype}")
            logger.info(f"    Is input: {is_input}")

    @staticmethod
    def load_calibration_images(
        image_dir: str,
        preprocessor,
        num_images: int = 500,
        batch_size: int = 1
    ) -> List[np.ndarray]:
        """
        Load and preprocess calibration images

        Args:
            image_dir: Directory containing calibration images
            preprocessor: Preprocessing function
            num_images: Number of images to load
            batch_size: Batch size for calibration

        Returns:
            List of preprocessed image batches
        """
        import cv2
        from pathlib import Path

        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

        if len(image_files) == 0:
            raise ValueError(f"No images found in: {image_dir}")

        # Limit number of images
        image_files = image_files[:num_images]

        logger.info(f"Loading {len(image_files)} calibration images...")

        calibration_data = []

        for i, image_file in enumerate(image_files):
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                continue

            # Preprocess
            tensor = preprocessor.preprocess(image, return_metadata=False)

            # Add to batch or accumulate
            if batch_size == 1:
                calibration_data.append(tensor)
            else:
                # Batch processing (not implemented in this example)
                pass

            if (i + 1) % 50 == 0:
                logger.info(f"  Loaded: {i + 1}/{len(image_files)}")

        logger.info(f"✓ Loaded {len(calibration_data)} calibration images")
        return calibration_data


class TensorRTInferenceEngine:
    """
    TensorRT inference engine with zero-copy memory
    Optimized for maximum throughput
    """

    def __init__(self, engine_path: str):
        """
        Initialize inference engine

        Args:
            engine_path: Path to TensorRT engine
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT not available")

        logger.info(f"Loading TensorRT engine: {engine_path}")

        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load engine")

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = trt.volume(shape)

            # Allocate device memory
            device_mem = cuda.mem_alloc(size * dtype().itemsize)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
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

        logger.info(f"✓ Engine loaded successfully")
        logger.info(f"  Inputs: {len(self.inputs)}")
        logger.info(f"  Outputs: {len(self.outputs)}")

    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """
        Run inference with zero-copy

        Args:
            input_data: Input tensor

        Returns:
            List of output tensors
        """
        # Copy input to device (async)
        cuda.memcpy_htod_async(
            self.inputs[0]['device_mem'],
            input_data.astype(self.inputs[0]['dtype']).ravel(),
            self.stream
        )

        # Execute (async)
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Copy outputs from device (async)
        outputs = []
        for output in self.outputs:
            host_mem = np.empty(output['size'], dtype=output['dtype'])
            cuda.memcpy_dtoh_async(host_mem, output['device_mem'], self.stream)
            outputs.append(host_mem.reshape(output['shape']))

        # Synchronize
        self.stream.synchronize()

        return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build TensorRT INT8 engine")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--output", required=True, help="Output engine path")
    parser.add_argument("--calib-dir", help="Calibration images directory")
    parser.add_argument("--calib-cache", default="calibration.cache", help="Calibration cache")
    parser.add_argument("--num-calib", type=int, default=500, help="Number of calibration images")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 instead of INT8")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load calibration data if provided
    calibration_data = None
    if args.calib_dir and not args.fp16:
        # Note: Requires preprocessor implementation
        logger.error("Calibration data loading requires preprocessor")
        logger.error("Use --calib-cache with existing cache or implement preprocessor")
        exit(1)

    # Build engine
    builder = TensorRTINT8Builder(
        onnx_path=args.onnx,
        calibration_data=calibration_data,
        cache_file=args.calib_cache,
        fp16_mode=args.fp16,
        int8_mode=not args.fp16,
        verbose=args.verbose
    )

    if builder.build_engine(args.output):
        logger.info(f"\n✓ Engine saved to: {args.output}")
    else:
        logger.error("✗ Engine build failed")
        exit(1)
