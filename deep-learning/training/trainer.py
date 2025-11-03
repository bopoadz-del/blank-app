"""
Training Loop Implementation
Complete trainer with forward/backward pass, mixed precision, and gradient accumulation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Callable, Dict, List, Any
from dataclasses import dataclass
import time
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration"""

    # Model and data
    model: nn.Module
    train_loader: DataLoader
    val_loader: Optional[DataLoader] = None

    # Optimization
    optimizer: Optional[optim.Optimizer] = None
    criterion: Optional[nn.Module] = None
    scheduler: Optional[Any] = None

    # Training parameters
    num_epochs: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Mixed precision
    use_amp: bool = True

    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Checkpointing
    save_dir: Optional[str] = None
    save_every: int = 1
    save_best_only: bool = True

    # Logging
    log_every: int = 100
    val_every: int = 1

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_delta: float = 0.001


class Trainer:
    """
    Complete training loop with:
    - Forward and backward pass
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpointing
    - Early stopping
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer

        Args:
            config: Training configuration
        """
        self.config = config
        self.model = config.model.to(config.device)
        self.train_loader = config.train_loader
        self.val_loader = config.val_loader

        # Optimizer
        if config.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            self.optimizer = config.optimizer

        # Loss function
        if config.criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = config.criterion

        # Learning rate scheduler
        self.scheduler = config.scheduler

        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None

        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Save directory
        if config.save_dir:
            self.save_dir = Path(config.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

        logger.info(f"Trainer initialized")
        logger.info(f"  Device: {config.device}")
        logger.info(f"  Mixed precision: {config.use_amp}")
        logger.info(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")

    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop

        Returns:
            Dictionary of training history
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)

            # Validation
            if self.val_loader and (epoch + 1) % self.config.val_every == 0:
                val_loss, val_acc = self.validate(epoch)

                # Save best model
                if val_loss < self.best_val_loss - self.config.early_stopping_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0

                    if self.save_dir:
                        self.save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            # Save checkpoint
            if self.save_dir and (epoch + 1) % self.config.save_every == 0:
                if not self.config.save_best_only:
                    self.save_checkpoint(epoch, is_best=False)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if self.val_loader else train_loss)
                else:
                    self.scheduler.step()

        logger.info("Training completed!")

        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accs,
            'val_acc': self.val_accs
        }

    def train_epoch(self, epoch: int) -> tuple:
        """
        Train for one epoch

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Progress bar
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
        )

        for batch_idx, (data, target) in pbar:
            # Move to device
            data = data.to(self.config.device)
            target = target.to(self.config.device)

            # Forward pass with mixed precision
            if self.config.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Zero gradients
                self.optimizer.zero_grad()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)
            total_loss += loss.item() * self.config.gradient_accumulation_steps

            # Update progress bar
            if (batch_idx + 1) % self.config.log_every == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_acc = 100.0 * total_correct / total_samples
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{avg_acc:.2f}%'
                })

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100.0 * total_correct / total_samples

        self.train_losses.append(avg_loss)
        self.train_accs.append(avg_acc)

        logger.info(
            f"Epoch {epoch + 1} - "
            f"Train Loss: {avg_loss:.4f}, "
            f"Train Acc: {avg_acc:.2f}%"
        )

        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self, epoch: int) -> tuple:
        """
        Validate model

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, target in tqdm(self.val_loader, desc="Validation"):
            # Move to device
            data = data.to(self.config.device)
            target = target.to(self.config.device)

            # Forward pass
            if self.config.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
            else:
                output = self.model(data)
                loss = self.criterion(output, target)

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)
            total_loss += loss.item()

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100.0 * total_correct / total_samples

        self.val_losses.append(avg_loss)
        self.val_accs.append(avg_acc)

        logger.info(
            f"Epoch {epoch + 1} - "
            f"Val Loss: {avg_loss:.4f}, "
            f"Val Acc: {avg_acc:.2f}%"
        )

        return avg_loss, avg_acc

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accs,
            'val_acc': self.val_accs,
            'best_val_loss': self.best_val_loss
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save checkpoint
        if is_best:
            path = self.save_dir / 'best_model.pth'
            logger.info(f"Saving best model to {path}")
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            logger.info(f"Saving checkpoint to {path}")

        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to checkpoint
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.train_losses = checkpoint.get('train_loss', [])
        self.val_losses = checkpoint.get('val_loss', [])
        self.train_accs = checkpoint.get('train_acc', [])
        self.val_accs = checkpoint.get('val_acc', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        logger.info(f"Checkpoint loaded (epoch {checkpoint['epoch'] + 1})")


# Example usage
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import sys
    sys.path.append('..')
    from architectures.cnn import ResNet18

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    val_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Model
    model = ResNet18(num_classes=10, in_channels=1)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training config
    config = TrainingConfig(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=10,
        device=device,
        use_amp=True,
        save_dir='./checkpoints',
        log_every=50
    )

    # Train
    trainer = Trainer(config)
    history = trainer.train()

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
