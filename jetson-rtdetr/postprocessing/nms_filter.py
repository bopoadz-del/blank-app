"""
Postprocessing for RT-DETR
Non-Maximum Suppression (NMS), confidence filtering, and result formatting
"""

import logging
from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Detection:
    """Single detection result"""

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],  # x1, y1, x2, y2
        confidence: float,
        class_id: int,
        class_name: Optional[str] = None
    ):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name or f"class_{class_id}"

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name
        }

    def __repr__(self) -> str:
        return (
            f"Detection(class={self.class_name}, "
            f"conf={self.confidence:.3f}, "
            f"bbox=[{self.x1:.1f}, {self.y1:.1f}, {self.x2:.1f}, {self.y2:.1f}])"
        )


class RTDETRPostprocessor:
    """
    Postprocessing for RT-DETR outputs
    Handles NMS, confidence filtering, and coordinate transformation
    """

    def __init__(
        self,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        class_names: Optional[List[str]] = None,
        agnostic_nms: bool = False  # Class-agnostic NMS
    ):
        """
        Initialize postprocessor

        Args:
            conf_threshold: Confidence threshold for filtering
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections to keep
            class_names: List of class names
            agnostic_nms: Apply NMS across all classes
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.class_names = class_names
        self.agnostic_nms = agnostic_nms

        logger.info(f"Initialized RTDETRPostprocessor")
        logger.info(f"  Confidence threshold: {conf_threshold}")
        logger.info(f"  IoU threshold: {iou_threshold}")
        logger.info(f"  Max detections: {max_detections}")

    def process(
        self,
        outputs: List[np.ndarray],
        metadata: Optional[dict] = None
    ) -> List[Detection]:
        """
        Process RT-DETR model outputs

        Args:
            outputs: List of model outputs [boxes, scores, labels]
            metadata: Preprocessing metadata for coordinate transformation

        Returns:
            List of Detection objects
        """
        # RT-DETR outputs:
        # - boxes: [N, 4] in format [x1, y1, x2, y2] normalized [0, 1]
        # - scores: [N,] confidence scores
        # - labels: [N,] class indices

        if len(outputs) == 1:
            # Combined output format: [N, 6] (x1, y1, x2, y2, conf, class)
            combined = outputs[0]
            boxes = combined[:, :4]
            scores = combined[:, 4]
            labels = combined[:, 5].astype(int)
        elif len(outputs) == 3:
            # Separate outputs
            boxes, scores, labels = outputs
            labels = labels.astype(int)
        else:
            raise ValueError(f"Unexpected number of outputs: {len(outputs)}")

        # Filter by confidence
        mask = scores >= self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        if len(boxes) == 0:
            return []

        # Apply NMS
        if self.agnostic_nms:
            # Class-agnostic NMS
            keep_indices = self.nms(boxes, scores, self.iou_threshold)
        else:
            # Per-class NMS
            keep_indices = self.multiclass_nms(boxes, scores, labels, self.iou_threshold)

        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]

        # Limit number of detections
        if len(boxes) > self.max_detections:
            # Keep top-k by confidence
            top_k = np.argsort(scores)[::-1][:self.max_detections]
            boxes = boxes[top_k]
            scores = scores[top_k]
            labels = labels[top_k]

        # Transform coordinates if metadata provided
        if metadata is not None:
            boxes = self.transform_boxes(boxes, metadata)

        # Create Detection objects
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            class_name = (
                self.class_names[label] if self.class_names and label < len(self.class_names)
                else f"class_{label}"
            )
            detection = Detection(
                bbox=tuple(box),
                confidence=float(score),
                class_id=int(label),
                class_name=class_name
            )
            detections.append(detection)

        return detections

    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute IoU between two boxes

        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]

        Returns:
            IoU value
        """
        # Intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    @staticmethod
    def compute_iou_matrix(boxes: np.ndarray) -> np.ndarray:
        """
        Compute IoU matrix for all pairs of boxes

        Args:
            boxes: [N, 4] array of boxes

        Returns:
            [N, N] IoU matrix
        """
        N = len(boxes)
        iou_matrix = np.zeros((N, N), dtype=np.float32)

        for i in range(N):
            for j in range(i + 1, N):
                iou = RTDETRPostprocessor.compute_iou(boxes[i], boxes[j])
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou

        return iou_matrix

    def nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float
    ) -> np.ndarray:
        """
        Non-Maximum Suppression

        Args:
            boxes: [N, 4] boxes in format [x1, y1, x2, y2]
            scores: [N,] confidence scores
            iou_threshold: IoU threshold

        Returns:
            Indices of boxes to keep
        """
        if CV2_AVAILABLE:
            # Use OpenCV NMS (faster)
            indices = cv2.dnn.NMSBoxes(
                bboxes=boxes.tolist(),
                scores=scores.tolist(),
                score_threshold=0.0,  # Already filtered
                nms_threshold=iou_threshold
            )
            if len(indices) > 0:
                return indices.flatten()
            return np.array([], dtype=int)

        # Python NMS implementation
        # Sort by score
        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU with remaining boxes
            ious = np.array([
                self.compute_iou(boxes[i], boxes[j])
                for j in order[1:]
            ])

            # Keep boxes with IoU < threshold
            mask = ious <= iou_threshold
            order = order[1:][mask]

        return np.array(keep, dtype=int)

    def multiclass_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        iou_threshold: float
    ) -> np.ndarray:
        """
        Per-class Non-Maximum Suppression

        Args:
            boxes: [N, 4] boxes
            scores: [N,] confidence scores
            labels: [N,] class labels
            iou_threshold: IoU threshold

        Returns:
            Indices of boxes to keep
        """
        keep_indices = []

        # Apply NMS per class
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            class_boxes = boxes[mask]
            class_scores = scores[mask]
            class_indices = np.where(mask)[0]

            # Apply NMS
            keep = self.nms(class_boxes, class_scores, iou_threshold)
            keep_indices.extend(class_indices[keep])

        return np.array(keep_indices, dtype=int)

    def transform_boxes(
        self,
        boxes: np.ndarray,
        metadata: dict
    ) -> np.ndarray:
        """
        Transform boxes from model space to original image space

        Args:
            boxes: [N, 4] normalized boxes [x1, y1, x2, y2] in [0, 1]
            metadata: Preprocessing metadata with original_size, scale, pad

        Returns:
            Transformed boxes in original image coordinates
        """
        original_h, original_w = metadata['original_size']
        target_h, target_w = metadata['target_size']

        # Denormalize to target size
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= target_w
        boxes[:, [1, 3]] *= target_h

        # Remove padding
        if 'pad' in metadata:
            top, bottom, left, right = metadata['pad']
            boxes[:, [0, 2]] -= left
            boxes[:, [1, 3]] -= top

        # Scale to original size
        if 'scale' in metadata:
            scale = metadata['scale']
            if isinstance(scale, (int, float)):
                boxes /= scale
            else:  # Tuple of (scale_h, scale_w)
                scale_h, scale_w = scale
                boxes[:, [0, 2]] /= scale_w
                boxes[:, [1, 3]] /= scale_h

        # Clip to image bounds
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_h)

        return boxes

    def filter_by_area(
        self,
        detections: List[Detection],
        min_area: float = 0,
        max_area: float = float('inf')
    ) -> List[Detection]:
        """
        Filter detections by area

        Args:
            detections: List of detections
            min_area: Minimum area
            max_area: Maximum area

        Returns:
            Filtered detections
        """
        return [
            det for det in detections
            if min_area <= det.area <= max_area
        ]

    def filter_by_class(
        self,
        detections: List[Detection],
        class_ids: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None
    ) -> List[Detection]:
        """
        Filter detections by class

        Args:
            detections: List of detections
            class_ids: List of class IDs to keep
            class_names: List of class names to keep

        Returns:
            Filtered detections
        """
        if class_ids is not None:
            detections = [det for det in detections if det.class_id in class_ids]

        if class_names is not None:
            detections = [det for det in detections if det.class_name in class_names]

        return detections


class DetectionVisualizer:
    """
    Visualization utilities for detections
    """

    def __init__(
        self,
        class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        font_scale: float = 0.5,
        thickness: int = 2
    ):
        """
        Initialize visualizer

        Args:
            class_colors: Dictionary mapping class_id to BGR color
            font_scale: Font scale for text
            thickness: Line thickness
        """
        self.class_colors = class_colors or {}
        self.font_scale = font_scale
        self.thickness = thickness

    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for class"""
        if class_id in self.class_colors:
            return self.class_colors[class_id]

        # Generate random color (seeded by class_id)
        np.random.seed(class_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        self.class_colors[class_id] = color
        return color

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        show_conf: bool = True
    ) -> np.ndarray:
        """
        Draw detections on image

        Args:
            image: Image (H, W, 3) in BGR format
            detections: List of detections
            show_conf: Show confidence scores

        Returns:
            Image with detections drawn
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for visualization")

        image = image.copy()

        for det in detections:
            # Get color
            color = self.get_color(det.class_id)

            # Draw bounding box
            pt1 = (int(det.x1), int(det.y1))
            pt2 = (int(det.x2), int(det.y2))
            cv2.rectangle(image, pt1, pt2, color, self.thickness)

            # Draw label
            label = det.class_name
            if show_conf:
                label = f"{label} {det.confidence:.2f}"

            # Get text size for background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness
            )

            # Draw background rectangle
            cv2.rectangle(
                image,
                (pt1[0], pt1[1] - text_h - baseline - 5),
                (pt1[0] + text_w, pt1[1]),
                color,
                -1
            )

            # Draw text
            cv2.putText(
                image,
                label,
                (pt1[0], pt1[1] - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                self.thickness
            )

        return image


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Test postprocessing")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")

    args = parser.parse_args()

    # Create postprocessor
    postprocessor = RTDETRPostprocessor(
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    # Create dummy detections
    boxes = np.array([
        [100, 100, 200, 200],
        [150, 150, 250, 250],
        [300, 300, 400, 400]
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7])
    labels = np.array([0, 0, 1])

    # Test NMS
    keep = postprocessor.nms(boxes, scores, args.iou)
    print(f"NMS keep indices: {keep}")
    print(f"Kept boxes: {boxes[keep]}")
