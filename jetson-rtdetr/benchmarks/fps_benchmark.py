"""
FPS Benchmarking and Performance Analysis
Measures inference speed, latency, and throughput
"""

import time
import logging
from typing import List, Optional, Dict
from collections import deque
from dataclasses import dataclass, field
import statistics
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result statistics"""
    total_frames: int = 0
    total_time: float = 0.0
    latencies: List[float] = field(default_factory=list)

    @property
    def fps(self) -> float:
        """Average FPS"""
        if self.total_time == 0:
            return 0.0
        return self.total_frames / self.total_time

    @property
    def avg_latency(self) -> float:
        """Average latency in ms"""
        if not self.latencies:
            return 0.0
        return statistics.mean(self.latencies) * 1000

    @property
    def min_latency(self) -> float:
        """Minimum latency in ms"""
        if not self.latencies:
            return 0.0
        return min(self.latencies) * 1000

    @property
    def max_latency(self) -> float:
        """Maximum latency in ms"""
        if not self.latencies:
            return 0.0
        return max(self.latencies) * 1000

    @property
    def p50_latency(self) -> float:
        """50th percentile latency in ms"""
        if not self.latencies:
            return 0.0
        return np.percentile(self.latencies, 50) * 1000

    @property
    def p95_latency(self) -> float:
        """95th percentile latency in ms"""
        if not self.latencies:
            return 0.0
        return np.percentile(self.latencies, 95) * 1000

    @property
    def p99_latency(self) -> float:
        """99th percentile latency in ms"""
        if not self.latencies:
            return 0.0
        return np.percentile(self.latencies, 99) * 1000

    @property
    def std_latency(self) -> float:
        """Standard deviation of latency in ms"""
        if len(self.latencies) < 2:
            return 0.0
        return statistics.stdev(self.latencies) * 1000

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'total_frames': self.total_frames,
            'total_time': self.total_time,
            'fps': self.fps,
            'avg_latency_ms': self.avg_latency,
            'min_latency_ms': self.min_latency,
            'max_latency_ms': self.max_latency,
            'p50_latency_ms': self.p50_latency,
            'p95_latency_ms': self.p95_latency,
            'p99_latency_ms': self.p99_latency,
            'std_latency_ms': self.std_latency
        }

    def __str__(self) -> str:
        """String representation"""
        return (
            f"Benchmark Results:\n"
            f"  Frames: {self.total_frames}\n"
            f"  Time: {self.total_time:.2f}s\n"
            f"  FPS: {self.fps:.2f}\n"
            f"  Latency (ms):\n"
            f"    Avg: {self.avg_latency:.2f}\n"
            f"    Min: {self.min_latency:.2f}\n"
            f"    Max: {self.max_latency:.2f}\n"
            f"    P50: {self.p50_latency:.2f}\n"
            f"    P95: {self.p95_latency:.2f}\n"
            f"    P99: {self.p99_latency:.2f}\n"
            f"    Std: {self.std_latency:.2f}"
        )


class FPSMeter:
    """
    Real-time FPS meter with moving average
    """

    def __init__(self, window_size: int = 30):
        """
        Initialize FPS meter

        Args:
            window_size: Number of frames for moving average
        """
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)

    def update(self):
        """Update with new frame"""
        self.timestamps.append(time.time())

    def get_fps(self) -> float:
        """
        Get current FPS

        Returns:
            FPS value
        """
        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed == 0:
            return 0.0

        return (len(self.timestamps) - 1) / elapsed

    def reset(self):
        """Reset meter"""
        self.timestamps.clear()


class LatencyTracker:
    """
    Track inference latency with breakdown
    """

    def __init__(self):
        """Initialize latency tracker"""
        self.start_time = None
        self.stage_times = {}
        self.results = []

    def start(self):
        """Start timing"""
        self.start_time = time.time()
        self.stage_times = {}

    def mark(self, stage_name: str):
        """Mark stage completion"""
        if self.start_time is None:
            raise ValueError("Call start() first")
        self.stage_times[stage_name] = time.time() - self.start_time

    def stop(self) -> Dict[str, float]:
        """
        Stop timing and return latency breakdown

        Returns:
            Dictionary of stage latencies in milliseconds
        """
        if self.start_time is None:
            raise ValueError("Call start() first")

        total_time = time.time() - self.start_time
        latencies = {
            'total': total_time * 1000
        }

        # Add stage latencies
        prev_time = 0
        for stage, stage_time in self.stage_times.items():
            latencies[stage] = (stage_time - prev_time) * 1000
            prev_time = stage_time

        self.results.append(latencies)
        self.start_time = None

        return latencies

    def get_average_latencies(self) -> Dict[str, float]:
        """Get average latencies across all measurements"""
        if not self.results:
            return {}

        all_stages = set()
        for result in self.results:
            all_stages.update(result.keys())

        avg_latencies = {}
        for stage in all_stages:
            values = [r[stage] for r in self.results if stage in r]
            avg_latencies[stage] = statistics.mean(values)

        return avg_latencies


class InferenceBenchmark:
    """
    Comprehensive inference benchmark
    """

    def __init__(
        self,
        engine,
        warmup_iterations: int = 10,
        test_iterations: int = 100
    ):
        """
        Initialize benchmark

        Args:
            engine: Inference engine to benchmark
            warmup_iterations: Number of warmup iterations
            test_iterations: Number of test iterations
        """
        self.engine = engine
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations

        logger.info(f"Initialized InferenceBenchmark")
        logger.info(f"  Warmup: {warmup_iterations} iterations")
        logger.info(f"  Test: {test_iterations} iterations")

    def benchmark_single_image(
        self,
        image: np.ndarray,
        verbose: bool = True
    ) -> BenchmarkResult:
        """
        Benchmark single image inference

        Args:
            image: Input image
            verbose: Print progress

        Returns:
            BenchmarkResult object
        """
        result = BenchmarkResult()

        # Warmup
        if verbose:
            logger.info("Warming up...")
        for i in range(self.warmup_iterations):
            _ = self.engine.infer(image)
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"  Warmup: {i + 1}/{self.warmup_iterations}")

        # Benchmark
        if verbose:
            logger.info(f"Running benchmark ({self.test_iterations} iterations)...")

        start_time = time.time()

        for i in range(self.test_iterations):
            iter_start = time.time()
            _ = self.engine.infer(image)
            iter_time = time.time() - iter_start

            result.latencies.append(iter_time)

            if verbose and (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i + 1}/{self.test_iterations}")

        result.total_time = time.time() - start_time
        result.total_frames = self.test_iterations

        return result

    def benchmark_batch_inference(
        self,
        images: List[np.ndarray],
        batch_sizes: List[int] = [1, 2, 4, 8],
        verbose: bool = True
    ) -> Dict[int, BenchmarkResult]:
        """
        Benchmark batch inference with different batch sizes

        Args:
            images: List of input images
            batch_sizes: List of batch sizes to test
            verbose: Print progress

        Returns:
            Dictionary mapping batch size to BenchmarkResult
        """
        results = {}

        for batch_size in batch_sizes:
            if batch_size > len(images):
                logger.warning(f"Batch size {batch_size} > available images, skipping")
                continue

            if verbose:
                logger.info(f"\nBenchmarking batch size: {batch_size}")

            result = BenchmarkResult()

            # Create batches
            num_batches = self.test_iterations // batch_size
            batch = images[:batch_size]

            # Warmup
            for _ in range(self.warmup_iterations // batch_size):
                _ = self.engine.infer_batch(batch)

            # Benchmark
            start_time = time.time()

            for i in range(num_batches):
                iter_start = time.time()
                _ = self.engine.infer_batch(batch)
                iter_time = time.time() - iter_start

                result.latencies.append(iter_time / batch_size)  # Per-image latency

            result.total_time = time.time() - start_time
            result.total_frames = num_batches * batch_size

            results[batch_size] = result

            if verbose:
                logger.info(f"  Batch {batch_size}: {result.fps:.2f} FPS")

        return results

    def benchmark_end_to_end(
        self,
        image_paths: List[str],
        verbose: bool = True
    ) -> BenchmarkResult:
        """
        Benchmark end-to-end pipeline (load + preprocess + infer + postprocess)

        Args:
            image_paths: List of image paths
            verbose: Print progress

        Returns:
            BenchmarkResult object
        """
        result = BenchmarkResult()
        latency_tracker = LatencyTracker()

        if verbose:
            logger.info("Benchmarking end-to-end pipeline...")

        start_time = time.time()

        for i, image_path in enumerate(image_paths):
            latency_tracker.start()

            # Load image
            image = self.engine.preprocessor.load_image(image_path)
            latency_tracker.mark('load')

            # Preprocess
            tensor, metadata = self.engine.preprocessor.preprocess(
                image, return_metadata=True
            )
            latency_tracker.mark('preprocess')

            # Inference
            batch = np.expand_dims(tensor, axis=0)
            outputs = self.engine.context.infer(batch)
            latency_tracker.mark('inference')

            # Postprocess
            _ = self.engine.postprocessor.process(outputs, metadata)
            latency_tracker.mark('postprocess')

            latencies = latency_tracker.stop()
            result.latencies.append(latencies['total'] / 1000)  # Convert to seconds

            if verbose and (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i + 1}/{len(image_paths)}")

        result.total_time = time.time() - start_time
        result.total_frames = len(image_paths)

        # Print stage breakdown
        if verbose:
            avg_latencies = latency_tracker.get_average_latencies()
            logger.info("\nStage Latency Breakdown (ms):")
            for stage, latency in sorted(avg_latencies.items()):
                if stage != 'total':
                    logger.info(f"  {stage:15s}: {latency:7.2f}")
            logger.info(f"  {'total':15s}: {avg_latencies['total']:7.2f}")

        return result

    def benchmark_throughput(
        self,
        image: np.ndarray,
        duration: float = 10.0,
        verbose: bool = True
    ) -> BenchmarkResult:
        """
        Benchmark maximum throughput for specified duration

        Args:
            image: Input image
            duration: Test duration in seconds
            verbose: Print progress

        Returns:
            BenchmarkResult object
        """
        if verbose:
            logger.info(f"Benchmarking throughput for {duration}s...")

        result = BenchmarkResult()
        end_time = time.time() + duration

        start_time = time.time()
        frame_count = 0

        while time.time() < end_time:
            iter_start = time.time()
            _ = self.engine.infer(image)
            iter_time = time.time() - iter_start

            result.latencies.append(iter_time)
            frame_count += 1

            if verbose and frame_count % 100 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                logger.info(f"  Frames: {frame_count}, FPS: {current_fps:.2f}")

        result.total_time = time.time() - start_time
        result.total_frames = frame_count

        return result


def compare_results(results: Dict[str, BenchmarkResult], title: str = "Comparison"):
    """
    Compare multiple benchmark results

    Args:
        results: Dictionary mapping name to BenchmarkResult
        title: Comparison title
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"{title:^70}")
    logger.info(f"{'='*70}")

    # Print header
    logger.info(f"{'Config':<20} {'FPS':>10} {'Avg(ms)':>10} {'P95(ms)':>10} {'P99(ms)':>10}")
    logger.info(f"{'-'*70}")

    # Print results
    for name, result in results.items():
        logger.info(
            f"{name:<20} "
            f"{result.fps:>10.2f} "
            f"{result.avg_latency:>10.2f} "
            f"{result.p95_latency:>10.2f} "
            f"{result.p99_latency:>10.2f}"
        )

    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="FPS Benchmark")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--iterations", type=int, default=100, help="Test iterations")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])

    args = parser.parse_args()

    # Import after argument parsing to avoid unnecessary imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    from inference.inference_engine import InferenceEngine
    import cv2

    # Create engine
    engine = InferenceEngine(
        model_path=args.model,
        precision=args.precision
    )

    # Load test image
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Failed to load image: {args.image}")

    # Run benchmark
    benchmark = InferenceBenchmark(
        engine=engine,
        test_iterations=args.iterations
    )

    result = benchmark.benchmark_single_image(image)

    print("\n" + "="*70)
    print(result)
    print("="*70)
