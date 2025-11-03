"""
Advanced Training Infrastructure
Complete training loop with early stopping, schedulers, checkpointing, and more
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Callable, Any, Tuple
import logging
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving

    Features:
    - Monitors validation metric
    - Saves best model
    - Configurable patience
    - Supports both min and max mode (loss vs accuracy)
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize early stopping

        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop

        Args:
            score: Current validation metric
            epoch: Current epoch number

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.monitor_op(score - self.min_delta, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                logger.info(f"EarlyStopping: Validation metric improved to {score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"Early stopping triggered! Best epoch: {self.best_epoch}")
                return True

        return False

    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class ModelCheckpoint:
    """
    Save model checkpoints during training

    Features:
    - Save best model based on metric
    - Save periodic checkpoints
    - Save full training state (optimizer, scheduler, epoch)
    - Model versioning
    - Automatic cleanup of old checkpoints
    """

    def __init__(
        self,
        save_dir: str,
        filename: str = 'checkpoint',
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_top_k: int = 3,
        every_n_epochs: int = 1,
        verbose: bool = True
    ):
        """
        Initialize model checkpoint

        Args:
            save_dir: Directory to save checkpoints
            filename: Checkpoint filename pattern
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save best model
            save_top_k: Keep top k checkpoints
            every_n_epochs: Save every n epochs
            verbose: Print messages
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_top_k = save_top_k
        self.every_n_epochs = every_n_epochs
        self.verbose = verbose

        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.checkpoints = []  # List of (score, path) tuples

    def should_save(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Check if checkpoint should be saved"""
        if self.save_best_only:
            if self.monitor not in metrics:
                return False

            score = metrics[self.monitor]

            if self.mode == 'min':
                is_better = score < self.best_score
            else:
                is_better = score > self.best_score

            if is_better:
                self.best_score = score
                return True
            return False
        else:
            return epoch % self.every_n_epochs == 0

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        metrics: Dict[str, float],
        extra_state: Optional[Dict] = None
    ):
        """
        Save checkpoint

        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            metrics: Current metrics
            extra_state: Additional state to save
        """
        if not self.should_save(epoch, metrics):
            return

        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if extra_state is not None:
            checkpoint.update(extra_state)

        # Create filename
        score = metrics.get(self.monitor, 0.0)
        filename = f"{self.filename}_epoch{epoch}_{self.monitor}={score:.4f}.pt"
        filepath = self.save_dir / filename

        # Save checkpoint
        torch.save(checkpoint, filepath)

        if self.verbose:
            logger.info(f"Checkpoint saved: {filepath}")

        # Track checkpoint
        self.checkpoints.append((score, filepath))

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only top k"""
        if len(self.checkpoints) <= self.save_top_k:
            return

        # Sort by score
        if self.mode == 'min':
            self.checkpoints.sort(key=lambda x: x[0])
        else:
            self.checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Remove worst checkpoints
        for _, filepath in self.checkpoints[self.save_top_k:]:
            if filepath.exists():
                filepath.unlink()
                if self.verbose:
                    logger.info(f"Removed old checkpoint: {filepath}")

        self.checkpoints = self.checkpoints[:self.save_top_k]

    def load_best(self) -> Optional[Path]:
        """Get path to best checkpoint"""
        if not self.checkpoints:
            return None
        return self.checkpoints[0][1]


class AdvancedTrainer:
    """
    Advanced training loop with all modern features

    Features:
    - Training and validation loops
    - Early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Gradient clipping
    - Progress tracking
    - Metric logging
    - TensorBoard integration
    - Custom callbacks
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cuda',
        scheduler: Optional[Any] = None,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint: Optional[ModelCheckpoint] = None,
        callbacks: Optional[List[Callable]] = None,
        log_interval: int = 10
    ):
        """
        Initialize advanced trainer

        Args:
            model: Model to train
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler
            use_amp: Use automatic mixed precision
            gradient_accumulation_steps: Accumulate gradients over n steps
            max_grad_norm: Maximum gradient norm for clipping
            early_stopping: Early stopping callback
            checkpoint: Model checkpoint callback
            callbacks: List of custom callbacks
            log_interval: Log metrics every n batches
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.early_stopping = early_stopping
        self.checkpoint = checkpoint
        self.callbacks = callbacks or []
        self.log_interval = log_interval

        # AMP scaler
        if use_amp:
            self.scaler = GradScaler()

        # Metrics tracking
        self.history = defaultdict(list)
        self.current_epoch = 0

        # TensorBoard (optional)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path('runs') / datetime.now().strftime('%Y%m%d-%H%M%S')
            self.writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging to: {log_dir}")
        except ImportError:
            logger.warning("TensorBoard not available")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        epoch_loss = 0.0
        num_batches = len(train_loader)

        # Metrics
        metrics = defaultdict(float)

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            if isinstance(batch, (tuple, list)):
                batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
            elif isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            else:
                batch = batch.to(self.device)

            # Forward pass
            if self.use_amp:
                with autocast():
                    loss, batch_metrics = self._forward_step(batch)
                    loss = loss / self.gradient_accumulation_steps
            else:
                loss, batch_metrics = self._forward_step(batch)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm is not None:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Scheduler step (for step-based schedulers)
                if self.scheduler is not None and hasattr(self.scheduler, 'step_update'):
                    self.scheduler.step_update(epoch * num_batches + batch_idx)

            # Update metrics
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            for key, value in batch_metrics.items():
                metrics[key] += value

            # Log progress
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                logger.info(
                    f"Epoch {epoch} [{batch_idx + 1}/{num_batches}] "
                    f"Loss: {avg_loss:.6f}"
                )

        # Average metrics
        metrics['loss'] = epoch_loss / num_batches
        for key in list(metrics.keys()):
            if key != 'loss':
                metrics[key] /= num_batches

        return dict(metrics)

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        val_loss = 0.0
        num_batches = len(val_loader)

        metrics = defaultdict(float)

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                if isinstance(batch, (tuple, list)):
                    batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b
                            for b in batch]
                elif isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)

                # Forward pass
                if self.use_amp:
                    with autocast():
                        loss, batch_metrics = self._forward_step(batch)
                else:
                    loss, batch_metrics = self._forward_step(batch)

                # Update metrics
                val_loss += loss.item()
                for key, value in batch_metrics.items():
                    metrics[key] += value

        # Average metrics
        metrics['val_loss'] = val_loss / num_batches
        for key in list(metrics.keys()):
            if key != 'val_loss':
                metrics[f'val_{key}'] = metrics.pop(key) / num_batches

        return dict(metrics)

    def _forward_step(self, batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single forward step
        Override this method for custom forward pass

        Args:
            batch: Input batch

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Default implementation assumes batch is (inputs, targets)
        if isinstance(batch, (tuple, list)):
            inputs, targets = batch[0], batch[1]
        elif isinstance(batch, dict):
            inputs = batch['input']
            targets = batch['target']
        else:
            raise ValueError("Unsupported batch format")

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        # Calculate accuracy (for classification)
        metrics = {}
        if outputs.dim() > 1 and outputs.size(1) > 1:
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == targets).float().mean().item()
            metrics['accuracy'] = accuracy

        return loss, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        resume_from: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Complete training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            resume_from: Path to checkpoint to resume from

        Returns:
            Training history
        """
        # Resume from checkpoint
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)
            logger.info(f"Resumed training from epoch {start_epoch}")

        logger.info("=" * 70)
        logger.info("Starting Training")
        logger.info("=" * 70)
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed Precision: {self.use_amp}")
        logger.info(f"Gradient Accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"Epochs: {epochs}")
        logger.info("=" * 70)

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Training
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validation
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}

            # Update history
            for key, value in all_metrics.items():
                self.history[key].append(value)

            # Learning rate scheduler (epoch-based)
            if self.scheduler is not None and not hasattr(self.scheduler, 'step_update'):
                if 'val_loss' in all_metrics:
                    # ReduceLROnPlateau
                    if hasattr(self.scheduler, 'step') and 'metrics' in str(type(self.scheduler)):
                        self.scheduler.step(all_metrics['val_loss'])
                    else:
                        self.scheduler.step()
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Epoch summary
            epoch_time = time.time() - epoch_start
            logger.info("=" * 70)
            logger.info(f"Epoch {epoch + 1}/{epochs} - {epoch_time:.2f}s")
            logger.info(f"Learning Rate: {current_lr:.2e}")
            for key, value in all_metrics.items():
                logger.info(f"{key}: {value:.6f}")
            logger.info("=" * 70)

            # TensorBoard logging
            if self.writer is not None:
                for key, value in all_metrics.items():
                    self.writer.add_scalar(key, value, epoch)
                self.writer.add_scalar('learning_rate', current_lr, epoch)

            # Callbacks
            for callback in self.callbacks:
                callback(self, epoch, all_metrics)

            # Model checkpoint
            if self.checkpoint is not None:
                self.checkpoint.save(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=all_metrics,
                    extra_state={'history': dict(self.history)}
                )

            # Early stopping
            if self.early_stopping is not None:
                monitor_metric = all_metrics.get('val_loss', all_metrics.get('loss'))
                if self.early_stopping(monitor_metric, epoch):
                    logger.info("Training stopped by early stopping")
                    break

        logger.info("=" * 70)
        logger.info("Training Complete!")
        logger.info("=" * 70)

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()

        return dict(self.history)

    def save_checkpoint(self, path: str, extra_state: Optional[Dict] = None):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': dict(self.history),
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if extra_state is not None:
            checkpoint.update(extra_state)

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> int:
        """
        Load training checkpoint

        Returns:
            Epoch number to resume from
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'history' in checkpoint:
            self.history = defaultdict(list, checkpoint['history'])

        epoch = checkpoint.get('epoch', 0)
        logger.info(f"Checkpoint loaded: {path}")

        return epoch + 1  # Resume from next epoch


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Advanced Training Infrastructure Test")
    print("=" * 70)

    # Create dummy model and data
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    )

    # Dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            x = torch.randn(10)
            y = torch.randint(0, 2, (1,)).item()
            return x, y

    train_dataset = DummyDataset()
    val_dataset = DummyDataset()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

    # Setup training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=5, mode='min', verbose=True)

    # Model checkpoint
    checkpoint = ModelCheckpoint(
        save_dir='checkpoints',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        verbose=True
    )

    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        use_amp=True if device == 'cuda' else False,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        log_interval=5
    )

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20
    )

    print("\nTraining History:")
    for key, values in history.items():
        print(f"{key}: {values[-1]:.6f}")

    print("\n" + "=" * 70)
    print("Training infrastructure tested successfully!")
