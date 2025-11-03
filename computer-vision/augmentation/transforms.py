"""
Data Augmentation Transforms
Comprehensive augmentation pipeline for computer vision training
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Optional, Tuple, List, Union
import random
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComposedTransforms:
    """
    Compose multiple transforms together
    """

    def __init__(self, transforms: List):
        """
        Initialize composed transforms

        Args:
            transforms: List of transform functions
        """
        self.transforms = transforms

    def __call__(self, image, target=None):
        """Apply all transforms"""
        for transform in self.transforms:
            if target is not None:
                image, target = transform(image, target)
            else:
                image = transform(image)

        if target is not None:
            return image, target
        return image


class RandomRotation:
    """
    Random rotation augmentation
    """

    def __init__(self, degrees: Union[float, Tuple[float, float]], p: float = 0.5):
        """
        Initialize random rotation

        Args:
            degrees: Range of degrees (single value or tuple)
            p: Probability of applying transform
        """
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.p = p

    def __call__(self, image, target=None):
        """Apply random rotation"""
        if random.random() < self.p:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            image = TF.rotate(image, angle)

            if target is not None and 'boxes' in target:
                # Rotate bounding boxes (simplified - may need proper rotation matrix)
                logger.warning("Box rotation not implemented - boxes remain unchanged")

        if target is not None:
            return image, target
        return image


class RandomHorizontalFlip:
    """
    Random horizontal flip
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize random horizontal flip

        Args:
            p: Probability of flipping
        """
        self.p = p

    def __call__(self, image, target=None):
        """Apply random horizontal flip"""
        if random.random() < self.p:
            image = TF.hflip(image)

            if target is not None:
                # Flip boxes
                if 'boxes' in target:
                    boxes = target['boxes']
                    width = image.width if isinstance(image, Image.Image) else image.shape[2]
                    boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                    target['boxes'] = boxes

                # Flip masks
                if 'masks' in target:
                    target['masks'] = TF.hflip(target['masks'])

        if target is not None:
            return image, target
        return image


class RandomVerticalFlip:
    """
    Random vertical flip
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize random vertical flip

        Args:
            p: Probability of flipping
        """
        self.p = p

    def __call__(self, image, target=None):
        """Apply random vertical flip"""
        if random.random() < self.p:
            image = TF.vflip(image)

            if target is not None:
                # Flip boxes
                if 'boxes' in target:
                    boxes = target['boxes']
                    height = image.height if isinstance(image, Image.Image) else image.shape[1]
                    boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                    target['boxes'] = boxes

                # Flip masks
                if 'masks' in target:
                    target['masks'] = TF.vflip(target['masks'])

        if target is not None:
            return image, target
        return image


class RandomCrop:
    """
    Random crop augmentation
    """

    def __init__(self, size: Union[int, Tuple[int, int]], padding: Optional[int] = None):
        """
        Initialize random crop

        Args:
            size: Output size (height, width)
            padding: Padding before crop
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding

    def __call__(self, image, target=None):
        """Apply random crop"""
        if self.padding:
            image = TF.pad(image, self.padding)

        i, j, h, w = T.RandomCrop.get_params(image, self.size)
        image = TF.crop(image, i, j, h, w)

        if target is not None and 'boxes' in target:
            # Adjust boxes for crop
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] -= j
            boxes[:, [1, 3]] -= i

            # Clip boxes to image boundaries
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)

            # Remove boxes with zero area
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            target['boxes'] = boxes[keep]

            if 'labels' in target:
                target['labels'] = target['labels'][keep]

        if target is not None:
            return image, target
        return image


class ColorJitter:
    """
    Random color jittering
    """

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        p: float = 0.5
    ):
        """
        Initialize color jitter

        Args:
            brightness: Brightness adjustment factor
            contrast: Contrast adjustment factor
            saturation: Saturation adjustment factor
            hue: Hue adjustment factor
            p: Probability of applying transform
        """
        self.transform = T.ColorJitter(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, image, target=None):
        """Apply color jitter"""
        if random.random() < self.p:
            image = self.transform(image)

        if target is not None:
            return image, target
        return image


class Resize:
    """
    Resize image and annotations
    """

    def __init__(self, size: Union[int, Tuple[int, int]]):
        """
        Initialize resize

        Args:
            size: Output size
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, image, target=None):
        """Apply resize"""
        orig_width, orig_height = image.size if isinstance(image, Image.Image) else (image.shape[2], image.shape[1])

        image = TF.resize(image, self.size)

        new_height, new_width = self.size

        if target is not None and 'boxes' in target:
            # Scale boxes
            boxes = target['boxes'].clone()
            scale_x = new_width / orig_width
            scale_y = new_height / orig_height

            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            target['boxes'] = boxes

        if target is not None:
            return image, target
        return image


class Normalize:
    """
    Normalize image with mean and std
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize normalize

        Args:
            mean: Mean values for each channel
            std: Standard deviation for each channel
        """
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        """Apply normalization"""
        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)

        image = TF.normalize(image, self.mean, self.std)

        if target is not None:
            return image, target
        return image


class ToTensor:
    """
    Convert PIL Image or numpy array to tensor
    """

    def __call__(self, image, target=None):
        """Convert to tensor"""
        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if target is not None:
            return image, target
        return image


def get_classification_transforms(
    mode: str = 'train',
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
):
    """
    Get classification transforms

    Args:
        mode: 'train' or 'val'
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std

    Returns:
        Composed transforms
    """
    if mode == 'train':
        transforms = T.Compose([
            T.RandomResizedCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    else:
        transforms = T.Compose([
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    return transforms


def get_detection_transforms(
    mode: str = 'train',
    image_size: int = 640
):
    """
    Get object detection transforms

    Args:
        mode: 'train' or 'val'
        image_size: Target image size

    Returns:
        Composed transforms
    """
    if mode == 'train':
        transforms = ComposedTransforms([
            Resize((image_size, image_size)),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ToTensor(),
            Normalize()
        ])
    else:
        transforms = ComposedTransforms([
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize()
        ])

    return transforms


def get_segmentation_transforms(
    mode: str = 'train',
    image_size: int = 512
):
    """
    Get segmentation transforms

    Args:
        mode: 'train' or 'val'
        image_size: Target image size

    Returns:
        Composed transforms
    """
    if mode == 'train':
        transforms = ComposedTransforms([
            Resize((image_size, image_size)),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.2),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ToTensor(),
            Normalize()
        ])
    else:
        transforms = ComposedTransforms([
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize()
        ])

    return transforms


# Example usage
if __name__ == "__main__":
    from PIL import Image

    print("=" * 70)
    print("Data Augmentation Test")
    print("=" * 70)

    # Create dummy image
    image = Image.new('RGB', (640, 480), color=(100, 150, 200))

    # Test classification transforms
    print("\n1. Classification Transforms")
    print("-" * 70)
    train_transforms = get_classification_transforms(mode='train', image_size=224)
    augmented = train_transforms(image)
    print(f"Input shape: (640, 480)")
    print(f"Output shape: {augmented.shape}")

    # Test individual transforms
    print("\n2. Individual Transforms")
    print("-" * 70)

    # Rotation
    rotation = RandomRotation(degrees=30, p=1.0)
    rotated = rotation(image)
    print(f"Rotation applied: {rotated.size}")

    # Flip
    flip = RandomHorizontalFlip(p=1.0)
    flipped = flip(image)
    print(f"Horizontal flip applied: {flipped.size}")

    # Color jitter
    jitter = ColorJitter(brightness=0.3, contrast=0.3, p=1.0)
    jittered = jitter(image)
    print(f"Color jitter applied: {jittered.size}")

    print("\n" + "=" * 70)
    print("Data augmentation tested successfully!")
