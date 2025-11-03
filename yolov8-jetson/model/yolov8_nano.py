"""
YOLOv8 Nano Model Implementation
Optimized architecture for maximum FPS on Jetson devices
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch/Ultralytics not available for model export")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOv8NanoExporter:
    """
    YOLOv8 Nano model exporter for TensorRT optimization
    Handles ONNX export with optimizations for Jetson inference
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (640, 640),
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize YOLOv8 Nano exporter

        Args:
            model_path: Path to pretrained weights (downloads if None)
            input_size: Model input size (height, width)
            device: Device for export ('cuda' or 'cpu')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and Ultralytics required for export")

        self.input_size = input_size
        self.device = device

        # Load model
        if model_path and Path(model_path).exists():
            logger.info(f"Loading YOLOv8 nano from: {model_path}")
            self.model = YOLO(model_path)
        else:
            logger.info("Downloading YOLOv8 nano pretrained model...")
            self.model = YOLO("yolov8n.pt")

        logger.info(f"Model loaded on device: {self.device}")
        logger.info(f"Input size: {input_size}")

    def export_to_onnx(
        self,
        output_path: str = "yolov8n.onnx",
        simplify: bool = True,
        dynamic: bool = False,
        opset: int = 11
    ) -> str:
        """
        Export YOLOv8 model to ONNX format

        Args:
            output_path: Path to save ONNX model
            simplify: Simplify ONNX graph
            dynamic: Use dynamic axes
            opset: ONNX opset version

        Returns:
            Path to exported ONNX model
        """
        logger.info(f"Exporting YOLOv8 nano to ONNX...")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Simplify: {simplify}")
        logger.info(f"  Dynamic: {dynamic}")
        logger.info(f"  Opset: {opset}")

        # Export using Ultralytics API
        export_path = self.model.export(
            format="onnx",
            imgsz=self.input_size,
            simplify=simplify,
            dynamic=dynamic,
            opset=opset
        )

        # Verify export
        output_path = Path(output_path)
        if export_path != str(output_path):
            import shutil
            shutil.move(export_path, output_path)

        logger.info(f"✓ ONNX model exported to: {output_path}")

        # Verify ONNX model
        self._verify_onnx(str(output_path))

        return str(output_path)

    def _verify_onnx(self, onnx_path: str):
        """Verify ONNX model"""
        try:
            import onnx
            logger.info("Verifying ONNX model...")

            # Load and check
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # Print model info
            logger.info("ONNX Model Information:")
            logger.info(f"  Opset version: {onnx_model.opset_import[0].version}")

            # Print inputs
            for input in onnx_model.graph.input:
                shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
                logger.info(f"  Input: {input.name}, shape: {shape}")

            # Print outputs
            for output in onnx_model.graph.output:
                shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
                logger.info(f"  Output: {output.name}, shape: {shape}")

            logger.info("✓ ONNX model verified successfully")

        except ImportError:
            logger.warning("ONNX package not available for verification")
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")

    def optimize_onnx(self, onnx_path: str, output_path: Optional[str] = None):
        """
        Optimize ONNX model for TensorRT

        Args:
            onnx_path: Path to input ONNX model
            output_path: Path to save optimized model
        """
        try:
            import onnx
            from onnxsim import simplify

            logger.info(f"Optimizing ONNX model: {onnx_path}")

            # Load model
            onnx_model = onnx.load(onnx_path)

            # Simplify
            logger.info("Simplifying ONNX graph...")
            model_simplified, check = simplify(onnx_model)

            if not check:
                logger.warning("Simplified model validation failed")
                return

            # Save
            if output_path is None:
                output_path = onnx_path.replace(".onnx", "_optimized.onnx")

            onnx.save(model_simplified, output_path)
            logger.info(f"✓ Optimized model saved to: {output_path}")

        except ImportError:
            logger.warning("onnx-simplifier not available, skipping optimization")
        except Exception as e:
            logger.error(f"ONNX optimization failed: {e}")

    @staticmethod
    def benchmark_pytorch(
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        iterations: int = 100,
        warmup: int = 10
    ):
        """
        Benchmark PyTorch model before export

        Args:
            model_path: Path to model weights
            input_size: Input size
            iterations: Number of iterations
            warmup: Warmup iterations
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for benchmarking")

        import time

        logger.info(f"Benchmarking PyTorch YOLOv8 nano...")

        # Load model
        model = YOLO(model_path)
        model.model.eval()

        # Create dummy input
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dummy_input = torch.randn(1, 3, *input_size).to(device)

        # Warmup
        logger.info(f"Warming up ({warmup} iterations)...")
        with torch.no_grad():
            for _ in range(warmup):
                _ = model.model(dummy_input)

        # Benchmark
        logger.info(f"Running benchmark ({iterations} iterations)...")
        latencies = []

        with torch.no_grad():
            for i in range(iterations):
                start = time.time()
                _ = model.model(dummy_input)
                if device == "cuda":
                    torch.cuda.synchronize()
                latency = time.time() - start
                latencies.append(latency)

                if (i + 1) % 10 == 0:
                    logger.info(f"  Progress: {i + 1}/{iterations}")

        # Statistics
        avg_latency = np.mean(latencies) * 1000
        std_latency = np.std(latencies) * 1000
        min_latency = np.min(latencies) * 1000
        max_latency = np.max(latencies) * 1000
        fps = 1000 / avg_latency

        logger.info("\nPyTorch Benchmark Results:")
        logger.info(f"  Average latency: {avg_latency:.2f} ms")
        logger.info(f"  Std latency: {std_latency:.2f} ms")
        logger.info(f"  Min latency: {min_latency:.2f} ms")
        logger.info(f"  Max latency: {max_latency:.2f} ms")
        logger.info(f"  FPS: {fps:.2f}")


class YOLOv8OutputParser:
    """
    Parse YOLOv8 ONNX output format
    """

    def __init__(self, num_classes: int = 80):
        """
        Initialize output parser

        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes

    def parse_output(self, output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse YOLOv8 output

        YOLOv8 output format: [batch, 4 + num_classes, num_anchors]
        - 4: bounding box (x, y, w, h)
        - num_classes: class probabilities

        Args:
            output: Raw model output

        Returns:
            Tuple of (boxes, scores, class_ids)
        """
        # Transpose: [batch, num_anchors, 4 + num_classes]
        output = output.transpose(0, 2, 1)

        # Extract boxes and scores
        boxes = output[..., :4]  # [batch, num_anchors, 4]
        class_scores = output[..., 4:]  # [batch, num_anchors, num_classes]

        # Get max class score and id
        scores = np.max(class_scores, axis=-1)  # [batch, num_anchors]
        class_ids = np.argmax(class_scores, axis=-1)  # [batch, num_anchors]

        # Convert from xywh to xyxy
        boxes_xyxy = self.xywh2xyxy(boxes)

        return boxes_xyxy, scores, class_ids

    @staticmethod
    def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
        """
        Convert boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2]

        Args:
            boxes: Boxes in xywh format [..., 4]

        Returns:
            Boxes in xyxy format
        """
        boxes_xyxy = boxes.copy()
        boxes_xyxy[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1
        boxes_xyxy[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1
        boxes_xyxy[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2
        boxes_xyxy[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2
        return boxes_xyxy


# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export YOLOv8 nano to ONNX")
    parser.add_argument("--model", help="Path to YOLOv8 weights")
    parser.add_argument("--output", default="yolov8n.onnx", help="Output ONNX path")
    parser.add_argument("--size", type=int, default=640, help="Input size")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch size")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark PyTorch model")

    args = parser.parse_args()

    # Benchmark if requested
    if args.benchmark:
        if args.model:
            YOLOv8NanoExporter.benchmark_pytorch(
                model_path=args.model,
                input_size=(args.size, args.size)
            )
        else:
            logger.error("--model required for benchmarking")
        exit(0)

    # Export to ONNX
    exporter = YOLOv8NanoExporter(
        model_path=args.model,
        input_size=(args.size, args.size)
    )

    onnx_path = exporter.export_to_onnx(
        output_path=args.output,
        simplify=args.simplify,
        dynamic=args.dynamic
    )

    logger.info(f"\n✓ Export complete: {onnx_path}")
