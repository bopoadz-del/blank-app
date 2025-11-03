"""
Text Classification Training Example
Complete training pipeline for TextCNN, BiLSTM, and BERT classifiers
"""

import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm

from models.text_classifier import TextCNN, BiLSTM, BERTClassifier
from preprocessing.tokenizer import WordTokenizer
from preprocessing.text_processor import TextProcessor

try:
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Dataset for text classification"""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
        is_bert: bool = False
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_bert = is_bert

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.is_bert:
            # BERT tokenization
            encoding = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            # Word-level tokenization
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)

            # Truncate or pad
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            else:
                token_ids = token_ids + [0] * (self.max_length - len(token_ids))

            return {
                'input_ids': torch.tensor(token_ids, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)
            }


class TextClassificationTrainer:
    """
    Complete training pipeline for text classification

    Supports:
    - TextCNN
    - BiLSTM
    - BERT-based classifiers
    - Mixed precision training
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        use_amp: bool = False
    ):
        """
        Initialize trainer

        Args:
            model: Classification model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            use_amp: Use automatic mixed precision
        """
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # Scheduler (will be set in train method)
        self.scheduler = None

        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            # Get data
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    if 'attention_mask' in batch:
                        # BERT model
                        attention_mask = batch['attention_mask'].to(self.device)
                        logits = self.model(input_ids, attention_mask)
                    else:
                        # CNN/LSTM model
                        logits = self.model(input_ids)

                    loss = self.criterion(logits, labels)

                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if 'attention_mask' in batch:
                    attention_mask = batch['attention_mask'].to(self.device)
                    logits = self.model(input_ids, attention_mask)
                else:
                    logits = self.model(input_ids)

                loss = self.criterion(logits, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Calculate metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })

        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                # Get data
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                if 'attention_mask' in batch:
                    attention_mask = batch['attention_mask'].to(self.device)
                    logits = self.model(input_ids, attention_mask)
                else:
                    logits = self.model(input_ids)

                loss = self.criterion(logits, labels)

                # Calculate metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        patience: int = 3,
        save_path: str = 'best_model.pt'
    ):
        """
        Train model with early stopping

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            patience: Early stopping patience
            save_path: Path to save best model
        """
        logger.info("=" * 70)
        logger.info("Training Text Classification Model")
        logger.info("=" * 70)

        # Setup scheduler
        total_steps = len(train_loader) * epochs
        if TRANSFORMERS_AVAILABLE:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=total_steps // 10,
                num_training_steps=total_steps
            )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_acc = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            logger.info(f"\nEpoch {epoch}/{epochs}")
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, save_path)

                logger.info(f"Best model saved to {save_path}")
            else:
                patience_counter += 1
                logger.info(f"Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break

        logger.info("\n" + "=" * 70)
        logger.info("Training Complete!")
        logger.info(f"Best Val Loss: {best_val_loss:.4f}")
        logger.info("=" * 70)


def main():
    """Main training function"""

    # Sample data (replace with real dataset)
    train_texts = [
        "This movie is absolutely fantastic! Best film I've seen all year.",
        "Terrible movie, waste of time and money.",
        "Great acting and cinematography. Highly recommended!",
        "Boring and predictable. Not worth watching.",
        "Amazing story with brilliant performances.",
        "Disappointing ending ruined the whole movie.",
        "Must watch! Entertaining from start to finish.",
        "Poor script and bad direction. Skip this one.",
    ] * 100  # Repeat for more training samples

    train_labels = [1, 0, 1, 0, 1, 0, 1, 0] * 100  # 1=positive, 0=negative

    val_texts = train_texts[:50]
    val_labels = train_labels[:50]

    # Hyperparameters
    VOCAB_SIZE = 5000
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    NUM_CLASSES = 2
    MAX_LENGTH = 64
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 1e-3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # ========== Train TextCNN ==========
    logger.info("\n" + "=" * 70)
    logger.info("Training TextCNN")
    logger.info("=" * 70)

    # Preprocess text
    text_processor = TextProcessor(lowercase=True, remove_punctuation=True)
    processed_texts = text_processor.process_batch(train_texts)

    # Build vocabulary
    tokenizer = WordTokenizer(method='whitespace', max_vocab_size=VOCAB_SIZE)
    tokenizer.fit(processed_texts)

    # Create datasets
    train_dataset = TextDataset(
        processed_texts,
        train_labels,
        tokenizer,
        max_length=MAX_LENGTH,
        is_bert=False
    )

    val_dataset = TextDataset(
        text_processor.process_batch(val_texts),
        val_labels,
        tokenizer,
        max_length=MAX_LENGTH,
        is_bert=False
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Create model
    cnn_model = TextCNN(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        kernel_sizes=[3, 4, 5],
        num_filters=100,
        dropout=0.5
    )

    # Train
    cnn_trainer = TextClassificationTrainer(
        cnn_model,
        device=device,
        learning_rate=LEARNING_RATE
    )

    cnn_trainer.train(
        train_loader,
        val_loader,
        epochs=EPOCHS,
        patience=3,
        save_path='textcnn_best.pt'
    )

    # ========== Train BiLSTM ==========
    logger.info("\n" + "=" * 70)
    logger.info("Training BiLSTM")
    logger.info("=" * 70)

    # Create model
    lstm_model = BiLSTM(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        num_classes=NUM_CLASSES,
        use_attention=True,
        dropout=0.5
    )

    # Train
    lstm_trainer = TextClassificationTrainer(
        lstm_model,
        device=device,
        learning_rate=LEARNING_RATE
    )

    lstm_trainer.train(
        train_loader,
        val_loader,
        epochs=EPOCHS,
        patience=3,
        save_path='bilstm_best.pt'
    )

    # ========== Train BERT (if available) ==========
    if TRANSFORMERS_AVAILABLE:
        logger.info("\n" + "=" * 70)
        logger.info("Training BERT Classifier")
        logger.info("=" * 70)

        # BERT tokenizer
        bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        # Create datasets
        bert_train_dataset = TextDataset(
            train_texts,  # Use original texts (BERT does its own preprocessing)
            train_labels,
            bert_tokenizer,
            max_length=MAX_LENGTH,
            is_bert=True
        )

        bert_val_dataset = TextDataset(
            val_texts,
            val_labels,
            bert_tokenizer,
            max_length=MAX_LENGTH,
            is_bert=True
        )

        bert_train_loader = DataLoader(bert_train_dataset, batch_size=8, shuffle=True)
        bert_val_loader = DataLoader(bert_val_dataset, batch_size=8)

        # Create model
        bert_model = BERTClassifier(
            model_name='distilbert-base-uncased',
            num_classes=NUM_CLASSES,
            hidden_dim=128,
            dropout=0.1
        )

        # Train with lower learning rate for BERT
        bert_trainer = TextClassificationTrainer(
            bert_model,
            device=device,
            learning_rate=2e-5,  # Lower LR for fine-tuning
            use_amp=True if device == 'cuda' else False
        )

        bert_trainer.train(
            bert_train_loader,
            bert_val_loader,
            epochs=5,  # Fewer epochs for BERT
            patience=2,
            save_path='bert_classifier_best.pt'
        )

    logger.info("\n" + "=" * 70)
    logger.info("All Models Trained Successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
