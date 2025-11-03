"""
Basic RT-DETR Inference Example
Demonstrates single image inference with visualization
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import argparse
from inference.inference_engine import InferenceEngine
from postprocessing.nms_filter import DetectionVisualizer


def main():
    parser = argparse.ArgumentParser(description="RT-DETR Basic Inference Example")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to save output image")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--size", type=int, default=640, help="Input size")

    args = parser.parse_args()

    print(f"Initializing RT-DETR Inference Engine...")
    print(f"  Model: {args.model}")
    print(f"  Precision: {args.precision}")
    print(f"  Input size: {args.size}x{args.size}")

    # Create inference engine
    engine = InferenceEngine(
        model_path=args.model,
        precision=args.precision,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=(args.size, args.size)
    )

    print(f"\nLoading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Failed to load image: {args.image}")
        return

    print(f"  Image size: {image.shape[1]}x{image.shape[0]}")

    # Run inference
    print("\nRunning inference...")
    detections = engine.infer(image)

    print(f"\nResults:")
    print(f"  Found {len(detections)} detections")

    # Print detections
    for i, det in enumerate(detections):
        print(f"    [{i+1}] {det}")

    # Visualize if output path provided
    if args.output:
        print(f"\nVisualizing detections...")
        visualizer = DetectionVisualizer()
        output_image = visualizer.draw_detections(image, detections)

        cv2.imwrite(args.output, output_image)
        print(f"  Saved output to: {args.output}")


if __name__ == "__main__":
    main()
