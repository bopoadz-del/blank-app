"""
RT-DETR Benchmark Example
Demonstrates comprehensive performance benchmarking
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import argparse
import numpy as np
from inference.inference_engine import InferenceEngine
from benchmarks.fps_benchmark import InferenceBenchmark, compare_results


def main():
    parser = argparse.ArgumentParser(description="RT-DETR Benchmark Example")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--iterations", type=int, default=100, help="Test iterations")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--size", type=int, default=640, help="Input size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")

    args = parser.parse_args()

    print(f"="*70)
    print(f"RT-DETR Performance Benchmark".center(70))
    print(f"="*70)

    # Load test image
    print(f"\nLoading test image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Failed to load image")
        return

    print(f"  Image size: {image.shape[1]}x{image.shape[0]}")

    # Create inference engine
    print(f"\nInitializing inference engine...")
    print(f"  Model: {args.model}")
    print(f"  Precision: {args.precision}")
    print(f"  Input size: {args.size}x{args.size}")

    engine = InferenceEngine(
        model_path=args.model,
        precision=args.precision,
        input_size=(args.size, args.size)
    )

    # Create benchmark
    benchmark = InferenceBenchmark(
        engine=engine,
        warmup_iterations=args.warmup,
        test_iterations=args.iterations
    )

    # Test 1: Single image inference
    print(f"\n{'='*70}")
    print(f"Test 1: Single Image Inference".center(70))
    print(f"{'='*70}")

    result_single = benchmark.benchmark_single_image(image, verbose=True)

    print(f"\n{result_single}")

    # Test 2: Batch inference (if supported)
    print(f"\n{'='*70}")
    print(f"Test 2: Batch Inference".center(70))
    print(f"{'='*70}")

    # Create multiple test images
    test_images = [image.copy() for _ in range(8)]

    batch_results = benchmark.benchmark_batch_inference(
        test_images,
        batch_sizes=[1, 2, 4],
        verbose=True
    )

    # Compare batch results
    compare_dict = {
        f"Batch {bs}": result
        for bs, result in batch_results.items()
    }
    compare_results(compare_dict, "Batch Size Comparison")

    # Test 3: Throughput test
    print(f"\n{'='*70}")
    print(f"Test 3: Maximum Throughput".center(70))
    print(f"{'='*70}")

    result_throughput = benchmark.benchmark_throughput(
        image,
        duration=10.0,
        verbose=True
    )

    print(f"\n{result_throughput}")

    # Summary comparison
    print(f"\n{'='*70}")
    print(f"Summary Comparison".center(70))
    print(f"{'='*70}")

    summary = {
        "Single Image": result_single,
        "Batch 1": batch_results.get(1, result_single),
        "Batch 4": batch_results.get(4, result_single),
        "Max Throughput": result_throughput
    }

    compare_results(summary, "All Tests")

    # Performance analysis
    print(f"\n{'='*70}")
    print(f"Performance Analysis".center(70))
    print(f"{'='*70}")

    print(f"\nThroughput Efficiency:")
    print(f"  Single image: {result_single.fps:.2f} FPS")
    print(f"  Max throughput: {result_throughput.fps:.2f} FPS")
    print(f"  Efficiency: {(result_throughput.fps / result_single.fps * 100):.1f}%")

    if 4 in batch_results:
        print(f"\nBatch Processing:")
        print(f"  Batch 4 FPS: {batch_results[4].fps:.2f}")
        print(f"  Speedup vs single: {(batch_results[4].fps / result_single.fps):.2f}x")

    print(f"\nLatency Analysis:")
    print(f"  Average: {result_single.avg_latency:.2f} ms")
    print(f"  P95: {result_single.p95_latency:.2f} ms")
    print(f"  P99: {result_single.p99_latency:.2f} ms")

    if result_single.p99_latency < 50:
        print(f"  ✓ Excellent latency (< 50ms)")
    elif result_single.p99_latency < 100:
        print(f"  ✓ Good latency (< 100ms)")
    else:
        print(f"  ⚠ High latency (> 100ms)")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
