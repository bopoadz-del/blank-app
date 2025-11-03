"""
Optimized Inference Engine for RT-DETR on Jetson
Handles TensorRT inference with batch processing and multi-stream support
"""

import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
from collections import deque
from threading import Thread, Lock
import queue
import numpy as np

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.rtdetr_tensorrt import TensorRTInferenceContext, RTDETRTensorRTBuilder
from preprocessing.image_preprocessor import ImagePreprocessor
from postprocessing.nms_filter import RTDETRPostprocessor, Detection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCache:
    """
    Model caching system for fast loading
    """

    def __init__(self, cache_dir: str = "./model_cache"):
        """
        Initialize model cache

        Args:
            cache_dir: Directory to store cached models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        logger.info(f"Model cache directory: {self.cache_dir}")

    def get_cache_path(self, model_path: str, precision: str) -> Path:
        """Get cache path for model"""
        model_name = Path(model_path).stem
        return self.cache_dir / f"{model_name}_{precision}.engine"

    def has_cached(self, model_path: str, precision: str) -> bool:
        """Check if model is cached"""
        cache_path = self.get_cache_path(model_path, precision)
        return cache_path.exists()

    def load(self, model_path: str, precision: str):
        """Load model from cache"""
        cache_key = (model_path, precision)
        if cache_key in self.cache:
            logger.info("Model loaded from memory cache")
            return self.cache[cache_key]

        cache_path = self.get_cache_path(model_path, precision)
        if cache_path.exists():
            logger.info(f"Loading cached engine: {cache_path}")
            builder = RTDETRTensorRTBuilder(
                onnx_path=model_path,
                engine_path=str(cache_path),
                precision=precision
            )
            engine = builder.load_engine()
            if engine:
                self.cache[cache_key] = engine
            return engine
        return None

    def save(self, model_path: str, precision: str, engine):
        """Save model to cache"""
        cache_key = (model_path, precision)
        self.cache[cache_key] = engine


class InferenceEngine:
    """
    Optimized RT-DETR inference engine
    """

    def __init__(
        self,
        model_path: str,
        precision: str = "fp16",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        class_names: Optional[List[str]] = None,
        enable_cache: bool = True,
        warmup_runs: int = 10
    ):
        """
        Initialize inference engine

        Args:
            model_path: Path to ONNX model
            precision: Precision mode (fp32, fp16, int8)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            input_size: Model input size (H, W)
            class_names: List of class names
            enable_cache: Enable model caching
            warmup_runs: Number of warmup iterations
        """
        self.model_path = model_path
        self.precision = precision
        self.input_size = input_size

        # Initialize cache
        self.cache = ModelCache() if enable_cache else None

        # Build/load TensorRT engine
        logger.info("Loading TensorRT engine...")
        self.engine = self._load_or_build_engine()
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        # Create inference context
        self.context = TensorRTInferenceContext(self.engine)

        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor(
            input_size=input_size,
            keep_ratio=True,
            backend="cv2"
        )

        # Initialize postprocessor
        self.postprocessor = RTDETRPostprocessor(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            class_names=class_names
        )

        # Warmup
        logger.info(f"Warming up engine ({warmup_runs} iterations)...")
        self._warmup(warmup_runs)

        logger.info("✓ Inference engine ready!")

    def _load_or_build_engine(self):
        """Load cached engine or build new one"""
        # Try to load from cache
        if self.cache:
            engine = self.cache.load(self.model_path, self.precision)
            if engine:
                return engine

        # Build new engine
        logger.info("Building TensorRT engine...")
        builder = RTDETRTensorRTBuilder(
            onnx_path=self.model_path,
            precision=self.precision,
            max_batch_size=1
        )

        if not builder.build_engine():
            return None

        engine = builder.load_engine()

        # Save to cache
        if self.cache and engine:
            self.cache.save(self.model_path, self.precision, engine)

        return engine

    def _warmup(self, num_runs: int):
        """Warmup engine with dummy data"""
        dummy_input = np.random.rand(1, 3, *self.input_size).astype(np.float32)

        for i in range(num_runs):
            _ = self.context.infer(dummy_input)

        logger.info("✓ Warmup complete")

    def infer(
        self,
        image: Union[str, np.ndarray],
        return_original: bool = False
    ) -> Union[List[Detection], Tuple[List[Detection], np.ndarray]]:
        """
        Run inference on single image

        Args:
            image: Image path or numpy array
            return_original: Return original image with results

        Returns:
            List of Detection objects, optionally with original image
        """
        # Preprocess
        tensor, metadata = self.preprocessor.preprocess(image, return_metadata=True)

        # Add batch dimension
        batch = np.expand_dims(tensor, axis=0)

        # Inference
        outputs = self.context.infer(batch)

        # Postprocess
        detections = self.postprocessor.process(outputs, metadata)

        if return_original:
            if isinstance(image, str):
                original = self.preprocessor.load_image(image)
            else:
                original = image
            return detections, original

        return detections

    def infer_batch(
        self,
        images: List[Union[str, np.ndarray]]
    ) -> List[List[Detection]]:
        """
        Run inference on batch of images

        Args:
            images: List of image paths or numpy arrays

        Returns:
            List of detection lists for each image
        """
        # Preprocess batch
        batch, metadata_list = self.preprocessor.preprocess_batch(
            images, return_metadata=True
        )

        # Inference
        outputs = self.context.infer(batch)

        # Postprocess each image
        all_detections = []
        for i, metadata in enumerate(metadata_list):
            # Extract outputs for this image
            image_outputs = [output[i:i+1] for output in outputs]
            detections = self.postprocessor.process(image_outputs, metadata)
            all_detections.append(detections)

        return all_detections


class MultiStreamInference:
    """
    Multi-stream inference with thread pool
    Handles concurrent inference on multiple video streams
    """

    def __init__(
        self,
        engine: InferenceEngine,
        num_workers: int = 2,
        queue_size: int = 10
    ):
        """
        Initialize multi-stream inference

        Args:
            engine: Inference engine
            num_workers: Number of worker threads
            queue_size: Maximum queue size per stream
        """
        self.engine = engine
        self.num_workers = num_workers
        self.queue_size = queue_size

        self.streams = {}
        self.workers = []
        self.running = False
        self.lock = Lock()

        logger.info(f"Initialized MultiStreamInference")
        logger.info(f"  Workers: {num_workers}")
        logger.info(f"  Queue size: {queue_size}")

    def add_stream(self, stream_id: str) -> bool:
        """
        Add new stream

        Args:
            stream_id: Unique stream identifier

        Returns:
            True if added, False if already exists
        """
        with self.lock:
            if stream_id in self.streams:
                logger.warning(f"Stream {stream_id} already exists")
                return False

            self.streams[stream_id] = {
                'input_queue': queue.Queue(maxsize=self.queue_size),
                'output_queue': queue.Queue(maxsize=self.queue_size),
                'active': True
            }

            logger.info(f"Added stream: {stream_id}")
            return True

    def remove_stream(self, stream_id: str):
        """Remove stream"""
        with self.lock:
            if stream_id in self.streams:
                self.streams[stream_id]['active'] = False
                del self.streams[stream_id]
                logger.info(f"Removed stream: {stream_id}")

    def put_frame(
        self,
        stream_id: str,
        frame: np.ndarray,
        frame_id: Optional[int] = None,
        timeout: float = 1.0
    ) -> bool:
        """
        Add frame to stream for processing

        Args:
            stream_id: Stream identifier
            frame: Input frame
            frame_id: Optional frame identifier
            timeout: Queue put timeout

        Returns:
            True if added, False if queue full
        """
        if stream_id not in self.streams:
            logger.error(f"Stream {stream_id} does not exist")
            return False

        try:
            self.streams[stream_id]['input_queue'].put(
                (frame, frame_id),
                timeout=timeout
            )
            return True
        except queue.Full:
            logger.warning(f"Stream {stream_id} input queue full")
            return False

    def get_result(
        self,
        stream_id: str,
        timeout: float = 1.0
    ) -> Optional[Tuple[List[Detection], Optional[int]]]:
        """
        Get inference result from stream

        Args:
            stream_id: Stream identifier
            timeout: Queue get timeout

        Returns:
            Tuple of (detections, frame_id) or None if timeout
        """
        if stream_id not in self.streams:
            logger.error(f"Stream {stream_id} does not exist")
            return None

        try:
            return self.streams[stream_id]['output_queue'].get(timeout=timeout)
        except queue.Empty:
            return None

    def _worker(self):
        """Worker thread for processing frames"""
        while self.running:
            # Round-robin through streams
            for stream_id, stream_data in list(self.streams.items()):
                if not stream_data['active']:
                    continue

                try:
                    # Get frame from input queue
                    frame, frame_id = stream_data['input_queue'].get(timeout=0.01)

                    # Run inference
                    detections = self.engine.infer(frame)

                    # Put result to output queue
                    stream_data['output_queue'].put((detections, frame_id), timeout=0.01)

                except queue.Empty:
                    continue
                except queue.Full:
                    logger.warning(f"Stream {stream_id} output queue full")
                    continue
                except Exception as e:
                    logger.error(f"Error processing frame: {e}", exc_info=True)

    def start(self):
        """Start worker threads"""
        if self.running:
            logger.warning("Workers already running")
            return

        self.running = True
        self.workers = []

        for i in range(self.num_workers):
            worker = Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)

        logger.info(f"Started {self.num_workers} workers")

    def stop(self):
        """Stop worker threads"""
        self.running = False

        for worker in self.workers:
            worker.join(timeout=2.0)

        self.workers = []
        logger.info("Stopped workers")

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


class BatchProcessor:
    """
    Batch processing for efficient inference
    Accumulates frames and processes in batches
    """

    def __init__(
        self,
        engine: InferenceEngine,
        batch_size: int = 4,
        max_wait_time: float = 0.1
    ):
        """
        Initialize batch processor

        Args:
            engine: Inference engine
            batch_size: Target batch size
            max_wait_time: Maximum time to wait for batch (seconds)
        """
        self.engine = engine
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time

        self.batch_queue = deque()
        self.last_process_time = time.time()

        logger.info(f"Initialized BatchProcessor")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Max wait time: {max_wait_time}s")

    def add_frame(self, frame: np.ndarray, callback=None):
        """
        Add frame to batch queue

        Args:
            frame: Input frame
            callback: Callback function(detections) called when processed
        """
        self.batch_queue.append((frame, callback))

        # Process if batch is full or wait time exceeded
        if (len(self.batch_queue) >= self.batch_size or
            time.time() - self.last_process_time > self.max_wait_time):
            self.process_batch()

    def process_batch(self):
        """Process accumulated batch"""
        if len(self.batch_queue) == 0:
            return

        # Extract frames and callbacks
        items = list(self.batch_queue)
        self.batch_queue.clear()

        frames = [item[0] for item in items]
        callbacks = [item[1] for item in items]

        # Run batch inference
        try:
            results = self.engine.infer_batch(frames)

            # Call callbacks
            for callback, detections in zip(callbacks, results):
                if callback:
                    callback(detections)

        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)

        self.last_process_time = time.time()


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="RT-DETR Inference Engine")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")

    args = parser.parse_args()

    # Create engine
    engine = InferenceEngine(
        model_path=args.model,
        precision=args.precision,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    # Run inference
    detections, original = engine.infer(args.image, return_original=True)

    print(f"Found {len(detections)} detections:")
    for det in detections:
        print(f"  {det}")
