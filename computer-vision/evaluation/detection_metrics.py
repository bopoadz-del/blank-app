"""
Object Detection Evaluation Metrics
mAP, IoU, Precision-Recall calculations
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes

    Args:
        box1: Tensor of shape (4,) in format [x1, y1, x2, y2]
        box2: Tensor of shape (4,) in format [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection area
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection = inter_width * inter_height

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-6)

    return float(iou)


def calculate_iou_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes

    Args:
        boxes1: Tensor of shape (N, 4)
        boxes2: Tensor of shape (M, 4)

    Returns:
        IoU matrix of shape (N, M)
    """
    # Calculate intersection
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter_width = torch.clamp(x2 - x1, min=0)
    inter_height = torch.clamp(y2 - y1, min=0)
    intersection = inter_width * inter_height

    # Calculate areas
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Calculate union
    union = boxes1_area[:, None] + boxes2_area - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-6)

    return iou


def non_max_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    Apply Non-Maximum Suppression

    Args:
        boxes: Tensor of shape (N, 4) in format [x1, y1, x2, y2]
        scores: Tensor of shape (N,) with confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)

    # Sort by scores
    sorted_indices = torch.argsort(scores, descending=True)

    keep_indices = []

    while len(sorted_indices) > 0:
        # Keep highest score box
        current = sorted_indices[0]
        keep_indices.append(current)

        if len(sorted_indices) == 1:
            break

        # Calculate IoU with remaining boxes
        current_box = boxes[current:current+1]
        remaining_boxes = boxes[sorted_indices[1:]]

        ious = calculate_iou_batch(current_box, remaining_boxes).squeeze(0)

        # Keep boxes with IoU less than threshold
        mask = ious < iou_threshold
        sorted_indices = sorted_indices[1:][mask]

    return torch.tensor(keep_indices, dtype=torch.long)


def calculate_precision_recall(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 80
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate precision and recall for each class

    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        ground_truths: List of ground truth dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes

    Returns:
        Tuple of (precision, recall) arrays of shape (num_classes,)
    """
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)

    for class_id in range(num_classes):
        # Collect all predictions and ground truths for this class
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []

        for pred, gt in zip(predictions, ground_truths):
            # Get predictions for this class
            pred_mask = pred['labels'] == class_id
            if pred_mask.any():
                all_pred_boxes.append(pred['boxes'][pred_mask])
                all_pred_scores.append(pred['scores'][pred_mask])

            # Get ground truths for this class
            gt_mask = gt['labels'] == class_id
            if gt_mask.any():
                all_gt_boxes.append(gt['boxes'][gt_mask])

        if len(all_pred_boxes) == 0 or len(all_gt_boxes) == 0:
            continue

        # Concatenate all boxes
        pred_boxes = torch.cat(all_pred_boxes, dim=0)
        pred_scores = torch.cat(all_pred_scores, dim=0)
        gt_boxes = torch.cat(all_gt_boxes, dim=0)

        # Sort predictions by score
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]

        # Match predictions to ground truths
        tp = torch.zeros(len(pred_boxes))
        fp = torch.zeros(len(pred_boxes))

        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

        for i, pred_box in enumerate(pred_boxes):
            # Calculate IoU with all ground truths
            ious = calculate_iou_batch(pred_box.unsqueeze(0), gt_boxes).squeeze(0)

            # Find best match
            max_iou, max_idx = ious.max(dim=0)

            if max_iou >= iou_threshold and not gt_matched[max_idx]:
                tp[i] = 1
                gt_matched[max_idx] = True
            else:
                fp[i] = 1

        # Calculate precision and recall
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)

        if len(tp_cumsum) > 0:
            precision[class_id] = tp_cumsum[-1] / (tp_cumsum[-1] + fp_cumsum[-1] + 1e-6)
            recall[class_id] = tp_cumsum[-1] / (len(gt_boxes) + 1e-6)

    return precision, recall


def calculate_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 80
) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP)

    Args:
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts
        iou_threshold: IoU threshold
        num_classes: Number of classes

    Returns:
        Dictionary with mAP and per-class AP
    """
    ap_per_class = np.zeros(num_classes)

    for class_id in range(num_classes):
        # Collect predictions and ground truths for this class
        class_predictions = []
        class_ground_truths = []

        for pred, gt in zip(predictions, ground_truths):
            # Filter by class
            pred_mask = pred['labels'] == class_id
            gt_mask = gt['labels'] == class_id

            if pred_mask.any():
                class_predictions.append({
                    'boxes': pred['boxes'][pred_mask],
                    'scores': pred['scores'][pred_mask]
                })
            else:
                class_predictions.append({'boxes': torch.empty(0, 4), 'scores': torch.empty(0)})

            if gt_mask.any():
                class_ground_truths.append({'boxes': gt['boxes'][gt_mask]})
            else:
                class_ground_truths.append({'boxes': torch.empty(0, 4)})

        # Calculate AP for this class
        ap = calculate_ap_single_class(class_predictions, class_ground_truths, iou_threshold)
        ap_per_class[class_id] = ap

    # Calculate mAP
    valid_classes = ap_per_class > 0
    if valid_classes.any():
        map_value = ap_per_class[valid_classes].mean()
    else:
        map_value = 0.0

    return {
        'mAP': float(map_value),
        'AP_per_class': ap_per_class.tolist()
    }


def calculate_ap_single_class(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate Average Precision for a single class

    Args:
        predictions: List of predictions for one class
        ground_truths: List of ground truths for one class
        iou_threshold: IoU threshold

    Returns:
        Average Precision value
    """
    # Collect all predictions
    all_boxes = []
    all_scores = []
    all_image_ids = []

    for img_id, pred in enumerate(predictions):
        if len(pred['boxes']) > 0:
            all_boxes.append(pred['boxes'])
            all_scores.append(pred['scores'])
            all_image_ids.extend([img_id] * len(pred['boxes']))

    if len(all_boxes) == 0:
        return 0.0

    # Concatenate
    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_image_ids = torch.tensor(all_image_ids)

    # Sort by score
    sorted_indices = torch.argsort(all_scores, descending=True)
    all_boxes = all_boxes[sorted_indices]
    all_scores = all_scores[sorted_indices]
    all_image_ids = all_image_ids[sorted_indices]

    # Match predictions to ground truths
    tp = []
    fp = []

    gt_matched = {img_id: torch.zeros(len(gt['boxes']), dtype=torch.bool)
                  for img_id, gt in enumerate(ground_truths)}

    for box, score, img_id in zip(all_boxes, all_scores, all_image_ids):
        img_id = int(img_id)
        gt_boxes = ground_truths[img_id]['boxes']

        if len(gt_boxes) == 0:
            fp.append(1)
            tp.append(0)
            continue

        # Calculate IoU
        ious = calculate_iou_batch(box.unsqueeze(0), gt_boxes).squeeze(0)
        max_iou, max_idx = ious.max(dim=0)

        if max_iou >= iou_threshold and not gt_matched[img_id][max_idx]:
            tp.append(1)
            fp.append(0)
            gt_matched[img_id][max_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    # Calculate precision-recall curve
    tp = torch.tensor(tp, dtype=torch.float)
    fp = torch.tensor(fp, dtype=torch.float)

    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    # Total ground truths
    total_gt = sum(len(gt['boxes']) for gt in ground_truths)

    recalls = tp_cumsum / (total_gt + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            p = precisions[mask].max()
            ap += p / 11

    return float(ap)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Object Detection Metrics Test")
    print("=" * 70)

    # Test IoU
    print("\n1. IoU Calculation")
    print("-" * 70)
    box1 = torch.tensor([0, 0, 100, 100], dtype=torch.float)
    box2 = torch.tensor([50, 50, 150, 150], dtype=torch.float)
    iou = calculate_iou(box1, box2)
    print(f"Box 1: {box1.tolist()}")
    print(f"Box 2: {box2.tolist()}")
    print(f"IoU: {iou:.4f}")

    # Test NMS
    print("\n2. Non-Maximum Suppression")
    print("-" * 70)
    boxes = torch.tensor([
        [0, 0, 100, 100],
        [10, 10, 110, 110],
        [200, 200, 300, 300]
    ], dtype=torch.float)
    scores = torch.tensor([0.9, 0.8, 0.95])

    keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
    print(f"Input boxes: {len(boxes)}")
    print(f"Kept boxes: {len(keep)}")
    print(f"Kept indices: {keep.tolist()}")

    # Test mAP
    print("\n3. mAP Calculation")
    print("-" * 70)

    # Dummy predictions and ground truths
    predictions = [
        {
            'boxes': torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=torch.float),
            'scores': torch.tensor([0.9, 0.85]),
            'labels': torch.tensor([0, 1])
        }
    ]

    ground_truths = [
        {
            'boxes': torch.tensor([[12, 12, 52, 52], [65, 65, 105, 105]], dtype=torch.float),
            'labels': torch.tensor([0, 1])
        }
    ]

    map_results = calculate_map(predictions, ground_truths, iou_threshold=0.5, num_classes=80)
    print(f"mAP@0.5: {map_results['mAP']:.4f}")

    print("\n" + "=" * 70)
    print("Detection metrics tested successfully!")
