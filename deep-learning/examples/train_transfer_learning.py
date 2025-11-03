"""
Example: Transfer Learning with Pretrained ResNet
Fine-tune pretrained ResNet on custom dataset
"""

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader

from training.transfer_learning import TransferLearning, FineTuner
from training.trainer import Trainer, TrainingConfig
from utils.gpu_utils import get_device
from data.dataset import get_transforms

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training function for transfer learning"""
    logger.info("=" * 70)
    logger.info("Transfer Learning Example: Fine-tuning ResNet-50")
    logger.info("=" * 70)

    # Configuration
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_CLASSES = 10
    NUM_WORKERS = 4

    # Get device
    device = get_device()

    # Data transforms
    train_transform = get_transforms(mode='train', image_size=224, augment=True)
    val_transform = get_transforms(mode='val', image_size=224, augment=False)

    # Load dataset (using CIFAR-10 as example, but works with any ImageFolder dataset)
    logger.info("Loading dataset...")

    # For CIFAR-10, we need to resize images to 224x224 for pretrained models
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

    # Load pretrained ResNet-50
    logger.info("\nLoading pretrained ResNet-50...")
    model = TransferLearning.load_pretrained_resnet(
        model_name='resnet50',
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True
    )

    # Get discriminative learning rate parameters
    logger.info("\nSetting up discriminative learning rates...")
    param_groups = TransferLearning.get_discriminative_lr_params(
        model,
        base_lr=LEARNING_RATE,
        layer_lr_decay=0.5
    )

    # Create optimizer with discriminative learning rates
    optimizer = optim.Adam(param_groups)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Phase 1: Train only the classifier (frozen backbone)
    logger.info("\n" + "=" * 70)
    logger.info("Phase 1: Training classifier with frozen backbone")
    logger.info("=" * 70)

    config_phase1 = TrainingConfig(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=10,
        device=str(device),
        use_amp=True,
        save_dir='./checkpoints/transfer_learning/phase1',
        log_every=50,
        val_every=1
    )

    trainer_phase1 = Trainer(config_phase1)
    history_phase1 = trainer_phase1.train()

    # Phase 2: Gradual unfreezing and fine-tuning
    logger.info("\n" + "=" * 70)
    logger.info("Phase 2: Gradual unfreezing and fine-tuning")
    logger.info("=" * 70)

    # Create fine-tuner
    fine_tuner = FineTuner(model, optimizer, scheduler)

    # Unfreeze layer4
    TransferLearning.freeze_layers(model, freeze_until='layer4')

    # Lower learning rate for fine-tuning
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1

    config_phase2 = TrainingConfig(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=20,
        device=str(device),
        use_amp=True,
        save_dir='./checkpoints/transfer_learning/phase2',
        log_every=50,
        val_every=1,
        early_stopping_patience=10
    )

    trainer_phase2 = Trainer(config_phase2)
    history_phase2 = trainer_phase2.train()

    # Phase 3: Fine-tune entire network
    logger.info("\n" + "=" * 70)
    logger.info("Phase 3: Fine-tuning entire network")
    logger.info("=" * 70)

    # Unfreeze all layers
    fine_tuner.unfreeze_all()

    # Even lower learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1

    config_phase3 = TrainingConfig(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=20,
        device=str(device),
        use_amp=True,
        save_dir='./checkpoints/transfer_learning/phase3',
        log_every=50,
        val_every=1,
        early_stopping_patience=10
    )

    trainer_phase3 = Trainer(config_phase3)
    history_phase3 = trainer_phase3.train()

    # Print final results
    logger.info("\n" + "=" * 70)
    logger.info("Transfer Learning Complete!")
    logger.info("=" * 70)

    logger.info(f"\nPhase 1 best val loss: {trainer_phase1.best_val_loss:.4f}")
    logger.info(f"Phase 2 best val loss: {trainer_phase2.best_val_loss:.4f}")
    logger.info(f"Phase 3 best val loss: {trainer_phase3.best_val_loss:.4f}")

    # Save final model
    final_path = './checkpoints/transfer_learning/final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history_phase1': history_phase1,
        'history_phase2': history_phase2,
        'history_phase3': history_phase3
    }, final_path)

    logger.info(f"\nFinal model saved to: {final_path}")


if __name__ == "__main__":
    main()
