"""
Complete Image Classification Training Example
Demonstrates end-to-end training pipeline with data augmentation and evaluation
"""

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import v2 as transforms
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationTrainer:
    """
    Complete trainer for image classification
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        num_epochs: int = 100,
        save_dir: str = './checkpoints',
        use_amp: bool = True
    ):
        """
        Initialize trainer

        Args:
            model: Classification model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            use_amp: Use automatic mixed precision
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0

        logger.info(f"Trainer initialized on device: {device}")
        logger.info(f"Mixed precision: {use_amp}")

    def train_epoch(self, epoch: int) -> tuple:
        """Train for one epoch"""
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Log progress
            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{self.num_epochs}] "
                    f"Batch [{batch_idx+1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Acc: {100.*correct/total:.2f}%"
                )

        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, epoch: int) -> tuple:
        """Validate model"""
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Calculate metrics
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def train(self):
        """Main training loop"""
        logger.info("=" * 70)
        logger.info("Starting Training")
        logger.info("=" * 70)

        start_time = time.time()

        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Log epoch results
            logger.info(
                f"\nEpoch [{epoch+1}/{self.num_epochs}] Summary:\n"
                f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n"
                f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"  New best model! Val Acc: {val_acc:.2f}%")

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

        # Training complete
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("Training Complete!")
        logger.info("=" * 70)
        logger.info(f"Total time: {elapsed/60:.2f} minutes")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'

        torch.save(checkpoint, path)


def main():
    """Main training function"""
    logger.info("=" * 70)
    logger.info("Image Classification Training")
    logger.info("=" * 70)

    # Configuration
    NUM_CLASSES = 10
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.1
    WEIGHT_DECAY = 5e-4
    NUM_WORKERS = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data transforms
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    # Load CIFAR-10 dataset
    logger.info("\nLoading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transforms
    )

    val_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=val_transforms
    )

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

    # Create model (ResNet18)
    logger.info("\nCreating ResNet-18 model...")
    model = models.resnet18(pretrained=False, num_classes=NUM_CLASSES)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
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

    # Create trainer
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=NUM_EPOCHS,
        save_dir='./checkpoints/resnet18_cifar10',
        use_amp=True
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
