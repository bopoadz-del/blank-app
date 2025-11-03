"""
Example: Train ResNet on CIFAR-10
Complete training script with mixed precision and data augmentation
"""

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from architectures.cnn import ResNet18
from training.trainer import Trainer, TrainingConfig
from utils.gpu_utils import get_device
from data.dataset import get_transforms

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    logger.info("=" * 70)
    logger.info("Training ResNet-18 on CIFAR-10")
    logger.info("=" * 70)

    # Configuration
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    NUM_WORKERS = 4

    # Get device
    device = get_device()

    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    # Load CIFAR-10 dataset
    logger.info("Loading CIFAR-10 dataset...")

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create model
    logger.info("Creating ResNet-18 model...")
    model = ResNet18(num_classes=10, in_channels=3)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 75],
        gamma=0.1
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training configuration
    config = TrainingConfig(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=NUM_EPOCHS,
        device=str(device),
        use_amp=True,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        save_dir='./checkpoints/resnet18_cifar10',
        save_every=10,
        save_best_only=True,
        log_every=50,
        val_every=1,
        early_stopping_patience=20,
        early_stopping_delta=0.001
    )

    # Create trainer
    trainer = Trainer(config)

    # Train
    logger.info("\nStarting training...")
    history = trainer.train()

    # Print final results
    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")

    if history['val_acc']:
        best_val_acc = max(history['val_acc'])
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Save final model
    final_path = './checkpoints/resnet18_cifar10/final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, final_path)

    logger.info(f"\nFinal model saved to: {final_path}")


if __name__ == "__main__":
    main()
