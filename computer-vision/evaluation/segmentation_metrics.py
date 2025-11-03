"""
Segmentation Evaluation Metrics
Dice coefficient, IoU, Pixel Accuracy
"""

import torch
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0
) -> float:
    """
    Calculate Dice Coefficient (F1 Score for segmentation)

    Args:
        pred: Predicted mask (H, W) or (N, H, W)
        target: Ground truth mask (H, W) or (N, H, W)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient value between 0 and 1
    """
    pred = pred.flatten()
    target = target.flatten()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return float(dice)


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0
) -> float:
    """
    Calculate Intersection over Union (IoU) for segmentation

    Args:
        pred: Predicted mask
        target: Ground truth mask
        smooth: Smoothing factor

    Returns:
        IoU value between 0 and 1
    """
    pred = pred.flatten()
    target = target.flatten()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return float(iou)


def pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    Calculate pixel-wise accuracy

    Args:
        pred: Predicted mask
        target: Ground truth mask

    Returns:
        Pixel accuracy value between 0 and 1
    """
    correct = (pred == target).sum()
    total = torch.numel(pred)

    accuracy = correct / total

    return float(accuracy)


def mean_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None
) -> float:
    """
    Calculate mean IoU across all classes

    Args:
        pred: Predicted segmentation map (H, W) with class indices
        target: Ground truth segmentation map (H, W)
        num_classes: Number of classes
        ignore_index: Class index to ignore

    Returns:
        Mean IoU value
    """
    ious = []

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()

        if union == 0:
            continue

        iou = intersection / union
        ious.append(float(iou))

    return np.mean(ious) if ious else 0.0


def confusion_matrix_multiclass(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Calculate confusion matrix for multiclass segmentation

    Args:
        pred: Predicted segmentation (H, W)
        target: Ground truth segmentation (H, W)
        num_classes: Number of classes

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    mask = (target >= 0) & (target < num_classes)
    conf_matrix = torch.bincount(
        num_classes * target[mask].int() + pred[mask].int(),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    return conf_matrix


def precision_recall_segmentation(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> tuple:
    """
    Calculate precision and recall for each class

    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        num_classes: Number of classes

    Returns:
        Tuple of (precision, recall) arrays
    """
    conf_matrix = confusion_matrix_multiclass(pred, target, num_classes)

    # Precision = TP / (TP + FP)
    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(dim=0) - tp
    precision = tp / (tp + fp + 1e-6)

    # Recall = TP / (TP + FN)
    fn = conf_matrix.sum(dim=1) - tp
    recall = tp / (tp + fn + 1e-6)

    return precision.numpy(), recall.numpy()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Segmentation Metrics Test")
    print("=" * 70)

    # Create dummy segmentation masks
    pred = torch.randint(0, 3, (256, 256))
    target = torch.randint(0, 3, (256, 256))

    # Test Dice coefficient
    print("\n1. Dice Coefficient")
    print("-" * 70)
    dice = dice_coefficient(pred, target)
    print(f"Dice: {dice:.4f}")

    # Test IoU
    print("\n2. IoU Score")
    print("-" * 70)
    iou = iou_score(pred, target)
    print(f"IoU: {iou:.4f}")

    # Test Pixel Accuracy
    print("\n3. Pixel Accuracy")
    print("-" * 70)
    acc = pixel_accuracy(pred, target)
    print(f"Pixel Accuracy: {acc:.4f}")

    # Test Mean IoU
    print("\n4. Mean IoU (multiclass)")
    print("-" * 70)
    miou = mean_iou(pred, target, num_classes=3)
    print(f"Mean IoU: {miou:.4f}")

    # Test Precision/Recall
    print("\n5. Precision and Recall")
    print("-" * 70)
    precision, recall = precision_recall_segmentation(pred, target, num_classes=3)
    for cls in range(3):
        print(f"Class {cls}: Precision={precision[cls]:.4f}, Recall={recall[cls]:.4f}")

    print("\n" + "=" * 70)
    print("Segmentation metrics tested successfully!")
