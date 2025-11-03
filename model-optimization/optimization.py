"""
Model Optimization
Pruning, quantization, and knowledge distillation for model compression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.quantization as quant
from typing import Optional, List, Tuple, Dict, Any
import logging
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPruner:
    """
    Neural network pruning for model compression

    Techniques:
    - Magnitude pruning: Remove smallest weights
    - Structured pruning: Remove entire channels/neurons
    - Iterative pruning: Gradually increase sparsity
    - Global pruning: Prune across all layers

    Benefits:
    - Smaller model size
    - Faster inference
    - Lower memory usage
    - Maintained accuracy (with fine-tuning)

    Usage:
        pruner = ModelPruner(model, amount=0.3)
        pruner.prune_magnitude()
        pruner.fine_tune(train_loader, epochs=5)
        pruner.make_permanent()
    """

    def __init__(
        self,
        model: nn.Module,
        amount: float = 0.3,
        structured: bool = False
    ):
        """
        Initialize model pruner

        Args:
            model: Model to prune
            amount: Fraction of weights to prune (0.0-1.0)
            structured: Use structured pruning (vs unstructured)
        """
        self.model = model
        self.amount = amount
        self.structured = structured

        self.pruned_modules = []
        self.sparsity = 0.0

    def prune_magnitude(self, layers: Optional[List[str]] = None):
        """
        Magnitude-based pruning

        Prunes weights with smallest absolute values.

        Args:
            layers: Specific layers to prune (None = all Conv2d and Linear)
        """
        logger.info(f"Applying magnitude pruning ({self.amount * 100:.1f}% sparsity)")

        if layers is None:
            # Prune all Conv2d and Linear layers
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if self.structured:
                        # Structured pruning (entire channels/neurons)
                        if isinstance(module, nn.Conv2d):
                            prune.ln_structured(
                                module,
                                name='weight',
                                amount=self.amount,
                                n=2,
                                dim=0  # Output channels
                            )
                        else:  # Linear
                            prune.ln_structured(
                                module,
                                name='weight',
                                amount=self.amount,
                                n=2,
                                dim=0  # Output features
                            )
                    else:
                        # Unstructured pruning (individual weights)
                        prune.l1_unstructured(
                            module,
                            name='weight',
                            amount=self.amount
                        )

                    self.pruned_modules.append((name, module))
        else:
            # Prune specific layers
            for name, module in self.model.named_modules():
                if name in layers:
                    if self.structured:
                        if isinstance(module, nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.amount, n=2, dim=0)
                        else:
                            prune.ln_structured(module, name='weight', amount=self.amount, n=2, dim=0)
                    else:
                        prune.l1_unstructured(module, name='weight', amount=self.amount)

                    self.pruned_modules.append((name, module))

        self._calculate_sparsity()
        logger.info(f"Pruning complete. Sparsity: {self.sparsity * 100:.2f}%")

    def prune_global(self):
        """
        Global magnitude pruning

        Prunes across all layers simultaneously based on global magnitude.
        Better than layer-wise pruning for maintaining accuracy.
        """
        logger.info(f"Applying global magnitude pruning ({self.amount * 100:.1f}% sparsity)")

        # Collect all prunable parameters
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
                self.pruned_modules.append((name, module))

        # Global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.amount
        )

        self._calculate_sparsity()
        logger.info(f"Global pruning complete. Sparsity: {self.sparsity * 100:.2f}%")

    def prune_iterative(
        self,
        train_fn: callable,
        num_iterations: int = 5,
        initial_sparsity: float = 0.1,
        final_sparsity: float = 0.7
    ):
        """
        Iterative pruning with fine-tuning

        Gradually increases sparsity over multiple iterations.

        Args:
            train_fn: Training function (receives model as argument)
            num_iterations: Number of pruning iterations
            initial_sparsity: Starting sparsity
            final_sparsity: Final target sparsity
        """
        logger.info("Starting iterative pruning")
        logger.info(f"Iterations: {num_iterations}")
        logger.info(f"Sparsity: {initial_sparsity:.2f} -> {final_sparsity:.2f}")

        # Calculate sparsity schedule
        sparsity_schedule = np.linspace(initial_sparsity, final_sparsity, num_iterations)

        for i, target_sparsity in enumerate(sparsity_schedule):
            logger.info(f"\nIteration {i+1}/{num_iterations} - Target sparsity: {target_sparsity:.2f}")

            # Prune
            self.amount = target_sparsity
            self.prune_global()

            # Fine-tune
            logger.info("Fine-tuning...")
            train_fn(self.model)

        logger.info("\nIterative pruning complete")

    def _calculate_sparsity(self):
        """Calculate actual sparsity of the model"""
        total_params = 0
        zero_params = 0

        for name, module in self.pruned_modules:
            if hasattr(module, 'weight_mask'):
                mask = module.weight_mask
                total_params += mask.numel()
                zero_params += (mask == 0).sum().item()

        if total_params > 0:
            self.sparsity = zero_params / total_params
        else:
            self.sparsity = 0.0

    def make_permanent(self):
        """
        Make pruning permanent

        Removes pruning reparameterization and makes sparsity permanent.
        """
        logger.info("Making pruning permanent")

        for name, module in self.pruned_modules:
            prune.remove(module, 'weight')

        logger.info("Pruning is now permanent")

    def get_model_size(self) -> Dict[str, float]:
        """
        Get model size information

        Returns:
            Dictionary with size metrics
        """
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024**2

        return {
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'size_mb': size_mb,
            'sparsity': self.sparsity
        }


class ModelQuantizer:
    """
    Model quantization for compression and acceleration

    Techniques:
    - Dynamic quantization: Quantize weights, activations stay float
    - Static quantization: Quantize weights and activations (INT8)
    - Quantization-aware training: Train with fake quantization

    Benefits:
    - 4x smaller model size (FP32 -> INT8)
    - 2-4x faster inference
    - Lower power consumption
    - Suitable for edge devices

    Usage:
        quantizer = ModelQuantizer(model)
        quantized_model = quantizer.quantize_dynamic()
        # or
        quantized_model = quantizer.quantize_static(calib_loader)
    """

    def __init__(self, model: nn.Module):
        """
        Initialize model quantizer

        Args:
            model: Model to quantize
        """
        self.model = model
        self.quantized_model = None

    def quantize_dynamic(
        self,
        qconfig_spec: Optional[Set] = None,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Dynamic quantization

        Quantizes weights statically, activations dynamically.
        Best for models where activation quantization is slow.

        Args:
            qconfig_spec: Modules to quantize (None = Linear and LSTM)
            dtype: Quantization dtype

        Returns:
            Quantized model
        """
        logger.info("Applying dynamic quantization")

        if qconfig_spec is None:
            qconfig_spec = {nn.Linear, nn.LSTM}

        self.quantized_model = quant.quantize_dynamic(
            self.model,
            qconfig_spec=qconfig_spec,
            dtype=dtype
        )

        logger.info("Dynamic quantization complete")
        self._print_size_comparison()

        return self.quantized_model

    def quantize_static(
        self,
        calibration_loader,
        backend: str = 'fbgemm'
    ) -> nn.Module:
        """
        Static quantization

        Quantizes both weights and activations.
        Requires calibration data.

        Args:
            calibration_loader: DataLoader for calibration
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)

        Returns:
            Quantized model
        """
        logger.info("Applying static quantization")

        # Set backend
        torch.backends.quantized.engine = backend

        # Prepare model
        self.model.eval()
        self.model.qconfig = quant.get_default_qconfig(backend)

        # Fuse modules (Conv-BN-ReLU, etc.)
        model_fused = self._fuse_modules()

        # Prepare for quantization
        model_prepared = quant.prepare(model_fused)

        # Calibrate
        logger.info("Calibrating...")
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(calibration_loader):
                model_prepared(inputs)
                if batch_idx >= 100:  # Use 100 batches for calibration
                    break

        # Convert to quantized model
        self.quantized_model = quant.convert(model_prepared)

        logger.info("Static quantization complete")
        self._print_size_comparison()

        return self.quantized_model

    def quantize_qat(
        self,
        train_fn: callable,
        backend: str = 'fbgemm'
    ) -> nn.Module:
        """
        Quantization-aware training (QAT)

        Trains with fake quantization to maintain accuracy.

        Args:
            train_fn: Training function (receives model)
            backend: Quantization backend

        Returns:
            Quantized model
        """
        logger.info("Applying quantization-aware training")

        # Set backend
        torch.backends.quantized.engine = backend

        # Prepare model for QAT
        self.model.train()
        self.model.qconfig = quant.get_default_qat_qconfig(backend)

        # Fuse modules
        model_fused = self._fuse_modules()

        # Prepare for QAT
        model_prepared = quant.prepare_qat(model_fused)

        # Train with fake quantization
        logger.info("Training with quantization-aware training...")
        train_fn(model_prepared)

        # Convert to quantized model
        model_prepared.eval()
        self.quantized_model = quant.convert(model_prepared)

        logger.info("QAT complete")
        self._print_size_comparison()

        return self.quantized_model

    def _fuse_modules(self) -> nn.Module:
        """Fuse common module patterns"""
        # This is model-specific
        # Example: fuse Conv2d-BatchNorm2d-ReLU
        model_fused = copy.deepcopy(self.model)

        # Add model-specific fusion here
        # For now, return as-is
        return model_fused

    def _print_size_comparison(self):
        """Print size comparison"""
        if self.quantized_model is None:
            return

        # Original size
        orig_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        orig_size_mb = orig_size / (1024 ** 2)

        # Quantized size
        quant_size = sum(p.numel() * p.element_size() for p in self.quantized_model.parameters())
        quant_size_mb = quant_size / (1024 ** 2)

        compression_ratio = orig_size / quant_size if quant_size > 0 else 0

        logger.info(f"Original size: {orig_size_mb:.2f} MB")
        logger.info(f"Quantized size: {quant_size_mb:.2f} MB")
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")


class KnowledgeDistiller:
    """
    Knowledge distillation for model compression

    Train a small "student" model to mimic a large "teacher" model.

    Process:
    1. Train large teacher model
    2. Use teacher's soft predictions as targets
    3. Train student model with both hard labels and soft targets

    Benefits:
    - Smaller, faster student model
    - Maintains accuracy close to teacher
    - Combines benefits of large and small models

    Usage:
        distiller = KnowledgeDistiller(teacher, student, temperature=4.0)
        distiller.train(train_loader, val_loader, epochs=100)
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
        device: str = 'cuda'
    ):
        """
        Initialize knowledge distiller

        Args:
            teacher: Pre-trained teacher model
            student: Student model to train
            temperature: Softmax temperature (higher = softer distributions)
            alpha: Weight for distillation loss (1-alpha for hard label loss)
            device: Device to train on
        """
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.temperature = temperature
        self.alpha = alpha
        self.device = device

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """
        Calculate distillation loss

        Loss = alpha * KL(teacher_soft || student_soft) + (1-alpha) * CE(student, labels)

        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            labels: True labels
            temperature: Softmax temperature

        Returns:
            Combined loss
        """
        # Soft targets (temperature scaling)
        soft_student = F.log_softmax(student_logits / temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=1)

        # Distillation loss (KL divergence)
        distill_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (temperature ** 2)

        # Hard label loss (standard cross-entropy)
        hard_loss = F.cross_entropy(student_logits, labels)

        # Combine losses
        loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss

        return loss

    def train_epoch(
        self,
        dataloader,
        optimizer,
        epoch: int
    ) -> float:
        """Train student for one epoch"""
        self.student.train()
        total_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Teacher predictions (no gradient)
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)

            # Student predictions
            student_logits = self.student(inputs)

            # Calculate distillation loss
            loss = self.distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                self.temperature
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def validate(self, dataloader) -> Tuple[float, float]:
        """Validate student model"""
        self.student.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Predictions
                teacher_logits = self.teacher(inputs)
                student_logits = self.student(inputs)

                # Loss
                loss = self.distillation_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    self.temperature
                )

                total_loss += loss.item()

                # Accuracy
                predictions = torch.argmax(student_logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader,
        val_loader,
        optimizer,
        epochs: int = 100,
        scheduler: Optional[Any] = None
    ):
        """
        Train student model with knowledge distillation

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for student
            epochs: Number of epochs
            scheduler: Learning rate scheduler
        """
        logger.info("=" * 70)
        logger.info("Starting Knowledge Distillation")
        logger.info("=" * 70)
        logger.info(f"Temperature: {self.temperature}")
        logger.info(f"Alpha: {self.alpha}")
        logger.info(f"Epochs: {epochs}")

        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, epoch)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Scheduler step
            if scheduler is not None:
                scheduler.step()

            # Log
            logger.info("=" * 70)
            logger.info(f"Epoch {epoch}/{epochs}")
            logger.info(f"Train Loss: {train_loss:.6f}")
            logger.info(f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.2f}%")
            logger.info("=" * 70)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")

        logger.info("=" * 70)
        logger.info("Knowledge Distillation Complete")
        logger.info(f"Best Val Accuracy: {best_val_acc:.2f}%")
        logger.info("=" * 70)


# Example usage
if __name__ == "__main__":
    import numpy as np

    print("=" * 70)
    print("Model Optimization Test")
    print("=" * 70)

    # Create a simple model
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(-1, 784)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = SimpleNet()

    # Test Pruning
    print("\n1. Model Pruning")
    print("-" * 70)

    pruner = ModelPruner(model, amount=0.3, structured=False)

    print("Before pruning:")
    size_info = pruner.get_model_size()
    print(f"  Total params: {size_info['total_params']:,}")
    print(f"  Size: {size_info['size_mb']:.2f} MB")

    pruner.prune_global()

    print("\nAfter pruning:")
    size_info = pruner.get_model_size()
    print(f"  Sparsity: {size_info['sparsity'] * 100:.2f}%")

    # Test Quantization
    print("\n2. Model Quantization")
    print("-" * 70)

    quantizer = ModelQuantizer(model)

    print("Applying dynamic quantization...")
    quantized_model = quantizer.quantize_dynamic()

    print("\n" + "=" * 70)
    print("Model optimization tested successfully!")
