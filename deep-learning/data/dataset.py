"""
Dataset and DataLoader Utilities
Custom datasets for images and sequences with augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """
    Custom Image Dataset
    Loads images from directory with optional transforms
    """

    def __init__(
        self,
        image_dir: str,
        labels: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ):
        """
        Initialize image dataset

        Args:
            image_dir: Directory containing images
            labels: Optional list of labels (one per image)
            transform: Optional transform to apply to images
            extensions: Valid image file extensions
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.extensions = extensions

        # Find all image files
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(self.image_dir.glob(f'*{ext}'))
            self.image_files.extend(self.image_dir.glob(f'*{ext.upper()}'))

        self.image_files = sorted(self.image_files)

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")

        # Labels
        if labels is not None:
            if len(labels) != len(self.image_files):
                raise ValueError(
                    f"Number of labels ({len(labels)}) must match "
                    f"number of images ({len(self.image_files)})"
                )
            self.labels = labels
        else:
            # No labels provided - use dummy labels
            self.labels = [0] * len(self.image_files)

        logger.info(f"Loaded {len(self.image_files)} images from {image_dir}")

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label at index

        Args:
            idx: Index

        Returns:
            Tuple of (image_tensor, label)
        """
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')

        # Apply transform
        if self.transform:
            image = self.transform(image)

        # Get label
        label = self.labels[idx]

        return image, label


class ImageFolderDataset(Dataset):
    """
    Image dataset with class folders
    Expects directory structure:
        root/class1/img1.jpg
        root/class1/img2.jpg
        root/class2/img3.jpg
        ...
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ):
        """
        Initialize image folder dataset

        Args:
            root: Root directory
            transform: Optional transform
            extensions: Valid image extensions
        """
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions

        # Find all classes (subdirectories)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Find all images
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            class_idx = self.class_to_idx[class_name]

            for ext in extensions:
                for img_path in class_dir.glob(f'*{ext}'):
                    self.samples.append((img_path, class_idx))
                for img_path in class_dir.glob(f'*{ext.upper()}'):
                    self.samples.append((img_path, class_idx))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root}")

        logger.info(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and label"""
        image_path, label = self.samples[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label


class SequenceDataset(Dataset):
    """
    Custom Sequence Dataset for RNN/LSTM/Transformer
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: List[int],
        max_len: Optional[int] = None,
        padding_value: float = 0.0
    ):
        """
        Initialize sequence dataset

        Args:
            sequences: List of sequences (variable length)
            labels: List of labels
            max_len: Maximum sequence length (for padding)
            padding_value: Value to use for padding
        """
        if len(sequences) != len(labels):
            raise ValueError("Number of sequences must match number of labels")

        self.sequences = sequences
        self.labels = labels
        self.padding_value = padding_value

        # Determine max length
        if max_len is None:
            self.max_len = max(len(seq) for seq in sequences)
        else:
            self.max_len = max_len

        logger.info(f"Loaded {len(sequences)} sequences")
        logger.info(f"Max sequence length: {self.max_len}")

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get sequence, label, and actual length

        Returns:
            Tuple of (padded_sequence, label, actual_length)
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        actual_len = len(sequence)

        # Pad or truncate sequence
        if len(sequence) < self.max_len:
            # Pad
            padding = np.full(
                (self.max_len - len(sequence), sequence.shape[1]),
                self.padding_value
            )
            sequence = np.vstack([sequence, padding])
        elif len(sequence) > self.max_len:
            # Truncate
            sequence = sequence[:self.max_len]

        # Convert to tensor
        sequence = torch.FloatTensor(sequence)

        return sequence, label, actual_len


def get_transforms(
    mode: str = 'train',
    image_size: int = 224,
    augment: bool = True
) -> transforms.Compose:
    """
    Get image transforms for training or validation

    Args:
        mode: 'train' or 'val'
        image_size: Target image size
        augment: Apply data augmentation

    Returns:
        Composed transforms
    """
    if mode == 'train' and augment:
        # Training transforms with augmentation
        transform_list = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    else:
        # Validation/test transforms (no augmentation)
        transform_list = [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]

    return transforms.Compose(transform_list)


def create_data_loaders(
    dataset: Dataset,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders with automatic splitting

    Args:
        dataset: Dataset to split
        batch_size: Batch size
        train_split: Fraction for training
        val_split: Fraction for validation
        num_workers: Number of worker processes
        shuffle: Shuffle training data
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


class CustomCollate:
    """
    Custom collate function for variable-length sequences
    """

    def __init__(self, padding_value: float = 0.0):
        """
        Initialize custom collate

        Args:
            padding_value: Value to use for padding
        """
        self.padding_value = padding_value

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate batch of variable-length sequences

        Args:
            batch: List of (sequence, label, length) tuples

        Returns:
            Tuple of (padded_sequences, labels, lengths)
        """
        sequences, labels, lengths = zip(*batch)

        # Find max length in batch
        max_len = max(lengths)

        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            if seq.size(0) < max_len:
                padding = torch.full(
                    (max_len - seq.size(0), seq.size(1)),
                    self.padding_value
                )
                seq = torch.cat([seq, padding], dim=0)
            padded_sequences.append(seq)

        # Stack
        sequences = torch.stack(padded_sequences)
        labels = torch.LongTensor(labels)
        lengths = torch.LongTensor(lengths)

        return sequences, labels, lengths


# Example usage
if __name__ == "__main__":
    print("Testing Image Dataset:")
    print("-" * 50)

    # Create dummy image directory
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy images
        for i in range(10):
            img = Image.new('RGB', (224, 224), color=(i * 25, i * 25, i * 25))
            img.save(os.path.join(tmpdir, f'img_{i}.jpg'))

        # Create dataset
        transform = get_transforms(mode='train', image_size=224)
        dataset = ImageDataset(
            image_dir=tmpdir,
            labels=list(range(10)),
            transform=transform
        )

        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset,
            batch_size=4,
            train_split=0.6,
            val_split=0.2,
            num_workers=0
        )

        # Test batch
        for images, labels in train_loader:
            print(f"Batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            break

    print("\n" + "=" * 50)
    print("Testing Sequence Dataset:")
    print("-" * 50)

    # Create dummy sequences
    sequences = [np.random.randn(np.random.randint(10, 50), 20) for _ in range(100)]
    labels = list(range(100))

    # Create dataset
    seq_dataset = SequenceDataset(sequences, labels, max_len=50)

    # Create data loader
    seq_loader = DataLoader(
        seq_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=CustomCollate()
    )

    # Test batch
    for sequences, labels, lengths in seq_loader:
        print(f"Sequences shape: {sequences.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Lengths: {lengths}")
        break

    print("\nDataset tests completed successfully!")
