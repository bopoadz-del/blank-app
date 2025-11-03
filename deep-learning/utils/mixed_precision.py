"""
Mixed Precision Training Utilities
Automatic Mixed Precision (AMP) for faster training with lower memory usage
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Optional, Callable
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MixedPrecisionTrainer:
    """
    Mixed Precision Training with Automatic Mixed Precision (AMP)

    Benefits:
    - 2-3x speedup on modern GPUs (Volta, Turing, Ampere)
    - 50% memory reduction
    - Minimal accuracy loss
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        enable_amp: bool = True,
        gradient_clip_value: Optional[float] = None
    ):
        """
        Initialize mixed precision trainer

        Args:
            model: Neural network model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            enable_amp: Enable automatic mixed precision
            gradient_clip_value: Gradient clipping value (None = no clipping)
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.enable_amp = enable_amp and device.type == 'cuda'
        self.gradient_clip_value = gradient_clip_value

        # Gradient scaler for mixed precision
        if self.enable_amp:
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            logger.info("Mixed precision training disabled")

    def train_step(
        self,
        data: torch.Tensor,
        target: torch.Tensor
    ) -> tuple:
        """
        Single training step with mixed precision

        Args:
            data: Input data
            target: Target labels

        Returns:
            Tuple of (loss, predictions)
        """
        # Move data to device
        data = data.to(self.device)
        target = target.to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass with autocast
        if self.enable_amp:
            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
        else:
            output = self.model(data)
            loss = self.criterion(output, target)

        # Backward pass
        if self.enable_amp:
            # Scale loss and backward
            self.scaler.scale(loss).backward()

            # Unscale gradients for clipping
            if self.gradient_clip_value:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_value
                )

            # Optimizer step with scaled gradients
            self.scaler.step(self.optimizer)

            # Update scaler
            self.scaler.update()
        else:
            # Standard backward
            loss.backward()

            # Gradient clipping
            if self.gradient_clip_value:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_value
                )

            # Optimizer step
            self.optimizer.step()

        return loss.item(), output

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> tuple:
        """
        Train for one epoch

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            leave=False
        )

        for data, target in pbar:
            # Training step
            loss, output = self.train_step(data, target)

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            target = target.to(self.device)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)
            total_loss += loss

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'acc': f'{100.0 * total_correct / total_samples:.2f}%'
            })

        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        avg_acc = 100.0 * total_correct / total_samples

        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> tuple:
        """
        Validate model

        Args:
            dataloader: Validation data loader
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, target in tqdm(dataloader, desc=f"Validation {epoch}", leave=False):
            # Move to device
            data = data.to(self.device)
            target = target.to(self.device)

            # Forward pass
            if self.enable_amp:
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
        avg_loss = total_loss / len(dataloader)
        avg_acc = 100.0 * total_correct / total_samples

        return avg_loss, avg_acc


class GradientAccumulator:
    """
    Gradient Accumulation for large effective batch sizes

    Useful when:
    - GPU memory is limited
    - Want large batch size for stability
    - Simulate larger batch size
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        accumulation_steps: int = 4,
        enable_amp: bool = True,
        gradient_clip_value: Optional[float] = None
    ):
        """
        Initialize gradient accumulator

        Args:
            model: Neural network model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            accumulation_steps: Number of steps to accumulate gradients
            enable_amp: Enable automatic mixed precision
            gradient_clip_value: Gradient clipping value
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.enable_amp = enable_amp and device.type == 'cuda'
        self.gradient_clip_value = gradient_clip_value

        # Gradient scaler
        if self.enable_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        logger.info(f"Gradient accumulation: {accumulation_steps} steps")
        logger.info(f"Mixed precision: {self.enable_amp}")

    def train_step(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        step: int
    ) -> tuple:
        """
        Single training step with gradient accumulation

        Args:
            data: Input data
            target: Target labels
            step: Current step number

        Returns:
            Tuple of (loss, predictions, should_update)
        """
        # Move data to device
        data = data.to(self.device)
        target = target.to(self.device)

        # Forward pass
        if self.enable_amp:
            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
                # Scale loss by accumulation steps
                loss = loss / self.accumulation_steps
        else:
            output = self.model(data)
            loss = self.criterion(output, target)
            loss = loss / self.accumulation_steps

        # Backward pass
        if self.enable_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Determine if should update
        should_update = (step + 1) % self.accumulation_steps == 0

        if should_update:
            # Gradient clipping and optimizer step
            if self.enable_amp:
                if self.gradient_clip_value:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_value
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.gradient_clip_value:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_value
                    )

                self.optimizer.step()

            # Zero gradients
            self.optimizer.zero_grad()

        return loss.item() * self.accumulation_steps, output, should_update


class DynamicLossScaler:
    """
    Dynamic loss scaling for mixed precision training
    Automatically adjusts scale factor to prevent overflow/underflow
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        scale_factor: float = 2.0,
        scale_window: int = 2000
    ):
        """
        Initialize dynamic loss scaler

        Args:
            init_scale: Initial scale factor
            scale_factor: Factor to scale up/down
            scale_window: Number of steps before scaling up
        """
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.steps_since_scale = 0
        self.overflow_count = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss"""
        return loss * self.scale

    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients"""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.scale)

    def check_overflow(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Check for gradient overflow

        Returns:
            True if overflow detected
        """
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        return True
        return False

    def update_scale(self, overflow: bool):
        """
        Update scale based on overflow

        Args:
            overflow: Whether overflow occurred
        """
        if overflow:
            # Scale down
            self.scale /= self.scale_factor
            self.overflow_count += 1
            self.steps_since_scale = 0
            logger.warning(f"Overflow detected, scaling down to {self.scale}")
        else:
            self.steps_since_scale += 1

            # Scale up if no overflow for scale_window steps
            if self.steps_since_scale >= self.scale_window:
                self.scale *= self.scale_factor
                self.steps_since_scale = 0
                logger.info(f"Scaling up to {self.scale}")


# Example usage
if __name__ == "__main__":
    print("Mixed Precision Training Test")
    print("=" * 50)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        exit(0)

    # Create simple model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )

    # Setup
    device = torch.device('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Create trainer
    trainer = MixedPrecisionTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        enable_amp=True,
        gradient_clip_value=1.0
    )

    # Dummy data
    data = torch.randn(32, 100)
    target = torch.randint(0, 10, (32,))

    # Training step
    loss, output = trainer.train_step(data, target)

    print(f"Loss: {loss:.4f}")
    print(f"Output shape: {output.shape}")

    print("\nMixed precision training test completed!")
