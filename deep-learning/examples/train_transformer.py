"""
Example: Train Transformer for Text Classification
Using custom Transformer implementation on text data
"""

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from architectures.transformer import Transformer
from training.trainer import Trainer, TrainingConfig
from utils.gpu_utils import get_device

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple text dataset for demonstration"""

    def __init__(self, num_samples: int = 1000, seq_len: int = 50, vocab_size: int = 1000):
        """
        Initialize dataset

        Args:
            num_samples: Number of samples
            seq_len: Sequence length
            vocab_size: Vocabulary size
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Generate random sequences (token indices)
        self.sequences = torch.randint(0, vocab_size, (num_samples, seq_len))

        # Generate random labels (binary classification)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def main():
    """Main training function for Transformer"""
    logger.info("=" * 70)
    logger.info("Training Transformer for Text Classification")
    logger.info("=" * 70)

    # Configuration
    VOCAB_SIZE = 5000
    D_MODEL = 256
    NUM_HEADS = 8
    NUM_LAYERS = 6
    D_FF = 1024
    NUM_CLASSES = 2
    MAX_LEN = 100
    SEQ_LEN = 50

    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0001
    NUM_WORKERS = 4

    # Get device
    device = get_device()

    # Create datasets
    logger.info("Creating datasets...")

    train_dataset = TextDataset(
        num_samples=10000,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE
    )

    val_dataset = TextDataset(
        num_samples=2000,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE
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

    # Create Transformer model
    logger.info("\nCreating Transformer model...")
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        num_classes=NUM_CLASSES,
        max_len=MAX_LEN,
        dropout=0.1,
        pooling='mean'  # Use mean pooling over sequence
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer (Adam with warmup)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # Learning rate scheduler (warmup + cosine decay)
    def lr_lambda(current_step):
        warmup_steps = 4000
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, NUM_EPOCHS * len(train_loader) - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
        save_dir='./checkpoints/transformer_classification',
        save_every=10,
        save_best_only=True,
        log_every=50,
        val_every=1,
        early_stopping_patience=10,
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
    final_path = './checkpoints/transformer_classification/final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'vocab_size': VOCAB_SIZE,
            'd_model': D_MODEL,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS,
            'd_ff': D_FF,
            'num_classes': NUM_CLASSES,
            'max_len': MAX_LEN
        },
        'history': history
    }, final_path)

    logger.info(f"\nFinal model saved to: {final_path}")

    # Demonstrate inference
    logger.info("\n" + "=" * 70)
    logger.info("Inference Example")
    logger.info("=" * 70)

    model.eval()
    with torch.no_grad():
        # Get a sample
        sample, label = val_dataset[0]
        sample = sample.unsqueeze(0).to(device)

        # Forward pass
        output = model(sample)
        prediction = torch.argmax(output, dim=1).item()

        logger.info(f"True label: {label}")
        logger.info(f"Predicted label: {prediction}")
        logger.info(f"Prediction probabilities: {torch.softmax(output, dim=1).cpu().numpy()}")


if __name__ == "__main__":
    main()
