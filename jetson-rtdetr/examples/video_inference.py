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
import os
import sys
from inference.inference_engine import InferenceEngine
from postprocessing.nms_filter import DetectionVisualizer
from benchmarks.fps_benchmark import FPSMeter


def print_camera_diagnostics(camera_index):
    """Print diagnostic information for camera troubleshooting"""
    print("\n" + "="*60)
    print("CAMERA DIAGNOSTICS")
    print("="*60)
    
    # Check available video devices on Linux
    if sys.platform.startswith('linux'):
        print("\nAvailable video devices:")
        video_devices = []
        for i in range(10):
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path):
                video_devices.append(device_path)
                print(f"  ✓ {device_path} exists")
        
        if not video_devices:
            print("  ✗ No video devices found in /dev/")
    
    # Check OpenCV build info
    print(f"\nOpenCV version: {cv2.__version__}")
    print(f"Video I/O backends: {cv2.getBuildInformation()}")
    
    print("\nTroubleshooting suggestions:")
    print("  1. Check if camera is physically connected")
    print("  2. Verify camera permissions: ls -l /dev/video*")
    print("  3. Add user to video group: sudo usermod -a -G video $USER")
    print("  4. Check if camera is in use: sudo lsof /dev/video*")
    print("  5. Try different camera indices (0, 1, 2, etc.)")
    print("  6. For USB cameras, try unplugging and reconnecting")
    print("  7. Check dmesg for hardware errors: dmesg | grep -i video")
    print("="*60 + "\n")


def open_video_source(source, max_retries=3, retry_delay=1.0):
    """
    Open video source with retry mechanism and detailed error reporting
    
    Args:
        source: Video source (camera index as int or string, or video file path)
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        cv2.VideoCapture object if successful, None otherwise
    """
    is_camera = False
    source_desc = source
    
    # Determine if source is a camera index or file
    if isinstance(source, str) and source.isdigit():
        source = int(source)
        is_camera = True
        source_desc = f"camera {source}"
    elif isinstance(source, int):
        is_camera = True
        source_desc = f"camera {source}"
    else:
        source_desc = f"video file '{source}'"
    
    print(f"\nAttempting to open {source_desc}...")
    
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            print(f"\nRetry attempt {attempt}/{max_retries}...")
            time.sleep(retry_delay)
        
        # Try to open video source
        cap = cv2.VideoCapture(source)
        
        # Check if successfully opened
        if cap.isOpened():
            # Verify we can actually read a frame
            ret, frame = cap.read()
            if ret:
                # Reset to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print(f"✓ Successfully opened {source_desc}")
                return cap
            else:
                print(f"✗ Opened {source_desc} but cannot read frames")
                cap.release()
        else:
            print(f"✗ Failed to open {source_desc} (attempt {attempt}/{max_retries})")
    
    # All retries failed
    print(f"\n{'='*60}")
    print(f"ERROR: Failed to open {source_desc} after {max_retries} attempts")
    print(f"{'='*60}")
    
    if is_camera:
        print_camera_diagnostics(source)
    else:
        print("\nPossible issues with video file:")
        print(f"  1. File does not exist: {source}")
        print(f"  2. File format not supported by OpenCV")
        print(f"  3. File is corrupted or incomplete")
        print(f"  4. Missing codecs for this video format")
        print("\nSupported formats: .mp4, .avi, .mov, .mkv, .webm")
        
        if not os.path.exists(source):
            print(f"\n✗ File not found: {source}")
        else:
            print(f"\n✓ File exists: {source}")
            file_size = os.path.getsize(source)
            print(f"  File size: {file_size:,} bytes")
    
    return None


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

    # Open video source with retry mechanism
    cap = open_video_source(args.input, max_retries=3, retry_delay=1.0)
    
    if cap is None:
        print("\n✗ Cannot proceed without valid video source")
        print("Please check the diagnostics above and try again.\n")
        return 1

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps if fps > 0 else 'Unknown'}")
    if frame_count > 0:
        print(f"  Frame count: {frame_count}")
        duration = frame_count / fps if fps > 0 else 0
        if duration > 0:
            print(f"  Duration: {duration:.2f} seconds")

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
        
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
