"""
Video RT-DETR Inference Example
Demonstrates real-time video inference with FPS counter
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import argparse
import time
from inference.inference_engine import InferenceEngine
from postprocessing.nms_filter import DetectionVisualizer
from benchmarks.fps_benchmark import FPSMeter


def main():
    parser = argparse.ArgumentParser(description="RT-DETR Video Inference Example")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--input", required=True, help="Path to video file or camera index (0, 1, etc.)")
    parser.add_argument("--output", help="Path to save output video")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--size", type=int, default=640, help="Input size")
    parser.add_argument("--no-display", action="store_true", help="Disable display window")

    args = parser.parse_args()

    print(f"Initializing RT-DETR Inference Engine...")

    # Create inference engine
    engine = InferenceEngine(
        model_path=args.model,
        precision=args.precision,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=(args.size, args.size)
    )

    # Open video
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
        print(f"\nOpened camera: {args.input}")
    else:
        cap = cv2.VideoCapture(args.input)
        print(f"\nOpened video: {args.input}")

    if not cap.isOpened():
        print(f"Error: Failed to open video source: {args.input}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    if frame_count > 0:
        print(f"  Frame count: {frame_count}")

    # Create video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"\nSaving output to: {args.output}")

    # Create visualizer and FPS meter
    visualizer = DetectionVisualizer()
    fps_meter = FPSMeter(window_size=30)

    print("\nProcessing video... (Press 'q' to quit)")
    frame_idx = 0

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Run inference
            detections = engine.infer(frame)

            # Update FPS meter
            fps_meter.update()
            current_fps = fps_meter.get_fps()

            # Draw detections
            output_frame = visualizer.draw_detections(frame, detections)

            # Draw FPS counter
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(
                output_frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            # Draw detection count
            det_text = f"Detections: {len(detections)}"
            cv2.putText(
                output_frame,
                det_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            # Display
            if not args.no_display:
                cv2.imshow('RT-DETR Inference', output_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break

            # Write to output video
            if writer:
                writer.write(output_frame)

            # Print progress
            if frame_idx % 30 == 0:
                print(f"  Frame: {frame_idx}, FPS: {current_fps:.2f}, Detections: {len(detections)}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Print summary
        print(f"\nSummary:")
        print(f"  Processed frames: {frame_idx}")
        print(f"  Average FPS: {fps_meter.get_fps():.2f}")


if __name__ == "__main__":
    main()
