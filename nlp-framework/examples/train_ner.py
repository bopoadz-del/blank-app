"""
Named Entity Recognition Training Example
Complete training pipeline for BiLSTM-CRF and BERT-NER
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

from models.ner import BiLSTM_CRF, BERT_NER, NER_TAG_SCHEMES
from preprocessing.tokenizer import WordTokenizer

try:
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from seqeval.metrics import classification_report, f1_score
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False
    logging.warning("seqeval not available. Install with: pip install seqeval")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NERDataset(Dataset):
    """Dataset for NER"""

    def __init__(
        self,
        sentences: List[List[str]],
        tags: List[List[int]],
        tokenizer=None,
        max_length: int = 128,
        is_bert: bool = False
    ):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_bert = is_bert

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tag_seq = self.tags[idx]

        if self.is_bert:
            # BERT tokenization
            # Note: BERT may split words into subwords, need to align tags
            encoding = self.tokenizer(
                sentence,
                is_split_into_words=True,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Align tags with subword tokens
            word_ids = encoding.word_ids(batch_index=0)
            aligned_tags = []
            previous_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    # Special tokens get -100 (ignored in loss)
                    aligned_tags.append(-100)
                elif word_id != previous_word_id:
                    # First subword of a word gets the tag
                    if word_id < len(tag_seq):
                        aligned_tags.append(tag_seq[word_id])
                    else:
                        aligned_tags.append(-100)
                else:
                    # Other subwords get -100
                    aligned_tags.append(-100)

                previous_word_id = word_id

            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'tags': torch.tensor(aligned_tags, dtype=torch.long)
            }
        else:
            # Word-level tokenization
            token_ids = [self.tokenizer.vocab.get(word, self.tokenizer.vocab['<UNK>'])
                        for word in sentence]

            # Truncate or pad
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
                tag_seq = tag_seq[:self.max_length]
            else:
                pad_len = self.max_length - len(token_ids)
                token_ids = token_ids + [0] * pad_len
                tag_seq = tag_seq + [0] * pad_len

            # Create mask (1 for real tokens, 0 for padding)
            mask = [1] * len(sentence) + [0] * (self.max_length - len(sentence))
            mask = mask[:self.max_length]

            return {
                'input_ids': torch.tensor(token_ids, dtype=torch.long),
                'tags': torch.tensor(tag_seq, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.bool)
            }


class NERTrainer:
    """
    Complete training pipeline for NER

    Supports:
    - BiLSTM-CRF
    - BERT-NER
    - Token-level evaluation
    - Entity-level evaluation (with seqeval)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        tag_names: List[str] = None
    ):
        """
        Initialize trainer

        Args:
            model: NER model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
            tag_names: List of tag names (for evaluation)
        """
        self.model = model.to(device)
        self.device = device
        self.tag_names = tag_names or NER_TAG_SCHEMES['IOB2']['tags']

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = None

        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            # Get data
            input_ids = batch['input_ids'].to(self.device)
            tags = batch['tags'].to(self.device)

            # Handle mask
            if 'mask' in batch:
                mask = batch['mask'].to(self.device)
            elif 'attention_mask' in batch:
                mask = batch['attention_mask'].to(self.device)
            else:
                mask = None

            # Forward pass
            self.optimizer.zero_grad()

            # Check if model is BiLSTM-CRF or BERT-NER
            if hasattr(self.model, 'crf'):
                # Model with CRF (BiLSTM-CRF or BERT-NER with CRF)
                if 'attention_mask' in batch:
                    # BERT-NER
                    attention_mask = batch['attention_mask'].to(self.device)
                    loss = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        tags=tags
                    )
                else:
                    # BiLSTM-CRF
                    loss = self.model(x=input_ids, tags=tags, mask=mask)
            else:
                # BERT-NER without CRF
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Calculate loss (ignore -100 labels)
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(outputs.view(-1, len(self.tag_names)), tags.view(-1))

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                # Get data
                input_ids = batch['input_ids'].to(self.device)
                tags = batch['tags'].to(self.device)

                if 'mask' in batch:
                    mask = batch['mask'].to(self.device)
                elif 'attention_mask' in batch:
                    mask = batch['attention_mask'].to(self.device)
                else:
                    mask = None

                # Get predictions
                if hasattr(self.model, 'crf'):
                    if 'attention_mask' in batch:
                        # BERT-NER with CRF
                        attention_mask = batch['attention_mask'].to(self.device)
                        loss = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            tags=tags
                        )
                        predictions = self.model.predict(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                    else:
                        # BiLSTM-CRF
                        loss = self.model(x=input_ids, tags=tags, mask=mask)
                        predictions = self.model.predict(x=input_ids, mask=mask)
                else:
                    # BERT-NER without CRF
                    attention_mask = batch['attention_mask'].to(self.device)
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(outputs.view(-1, len(self.tag_names)), tags.view(-1))

                    predictions = torch.argmax(outputs, dim=-1)

                total_loss += loss.item()

                # Collect predictions and labels
                if isinstance(predictions, list):
                    # CRF output (list of lists)
                    batch_predictions = predictions
                else:
                    # Tensor output
                    batch_predictions = predictions.cpu().numpy()

                batch_labels = tags.cpu().numpy()

                # Convert to lists for seqeval
                for i in range(len(batch_labels)):
                    pred_tags = []
                    true_tags = []

                    if isinstance(batch_predictions, list):
                        preds = batch_predictions[i]
                    else:
                        preds = batch_predictions[i]

                    for j, (pred, true) in enumerate(zip(preds, batch_labels[i])):
                        if true != -100 and true != 0:  # Skip padding and special tokens
                            pred_tags.append(self.tag_names[pred])
                            true_tags.append(self.tag_names[true])

                    if pred_tags and true_tags:
                        all_predictions.append(pred_tags)
                        all_labels.append(true_tags)

        avg_loss = total_loss / len(dataloader)

        # Calculate F1 score
        if SEQEVAL_AVAILABLE and all_predictions:
            f1 = f1_score(all_labels, all_predictions)
        else:
            f1 = 0.0

        return avg_loss, f1

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        patience: int = 3,
        save_path: str = 'best_ner_model.pt'
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
        logger.info("Training NER Model")
        logger.info("=" * 70)

        # Setup scheduler
        from transformers import get_linear_schedule_with_warmup
        total_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )

        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_f1 = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)

            logger.info(f"\nEpoch {epoch}/{epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss:   {val_loss:.4f} | Val F1: {val_f1:.4f}")

            # Early stopping based on F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_f1': val_f1
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
        logger.info(f"Best Val F1: {best_val_f1:.4f}")
        logger.info("=" * 70)


def main():
    """Main training function"""

    # Sample NER data (replace with real dataset like CoNLL-2003)
    # Format: (sentence, tags)
    train_sentences = [
        ["Barack", "Obama", "was", "born", "in", "Hawaii"],
        ["Apple", "Inc.", "is", "located", "in", "Cupertino"],
        ["London", "is", "the", "capital", "of", "England"],
        ["Microsoft", "was", "founded", "by", "Bill", "Gates"],
    ] * 100  # Repeat for more samples

    # IOB2 tags: O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC
    tag2id = {tag: i for i, tag in enumerate(NER_TAG_SCHEMES['IOB2']['tags'])}

    train_tags = [
        [tag2id['B-PER'], tag2id['I-PER'], tag2id['O'], tag2id['O'], tag2id['O'], tag2id['B-LOC']],
        [tag2id['B-ORG'], tag2id['I-ORG'], tag2id['O'], tag2id['O'], tag2id['O'], tag2id['B-LOC']],
        [tag2id['B-LOC'], tag2id['O'], tag2id['O'], tag2id['O'], tag2id['O'], tag2id['B-LOC']],
        [tag2id['B-ORG'], tag2id['O'], tag2id['O'], tag2id['O'], tag2id['B-PER'], tag2id['I-PER']],
    ] * 100

    val_sentences = train_sentences[:20]
    val_tags = train_tags[:20]

    # Hyperparameters
    VOCAB_SIZE = 5000
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    NUM_TAGS = len(NER_TAG_SCHEMES['IOB2']['tags'])
    MAX_LENGTH = 64
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 1e-3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Build vocabulary
    all_words = [word for sent in train_sentences for word in sent]
    tokenizer = WordTokenizer(method='whitespace', max_vocab_size=VOCAB_SIZE)
    tokenizer.fit([' '.join(sent) for sent in train_sentences])

    logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    logger.info(f"Number of NER tags: {NUM_TAGS}")

    # ========== Train BiLSTM-CRF ==========
    try:
        from torchcrf import CRF

        logger.info("\n" + "=" * 70)
        logger.info("Training BiLSTM-CRF")
        logger.info("=" * 70)

        # Create datasets
        train_dataset = NERDataset(
            train_sentences,
            train_tags,
            tokenizer,
            max_length=MAX_LENGTH,
            is_bert=False
        )

        val_dataset = NERDataset(
            val_sentences,
            val_tags,
            tokenizer,
            max_length=MAX_LENGTH,
            is_bert=False
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Create model
        bilstm_crf_model = BiLSTM_CRF(
            vocab_size=tokenizer.get_vocab_size(),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
            num_tags=NUM_TAGS,
            dropout=0.5
        )

        # Train
        trainer = NERTrainer(
            bilstm_crf_model,
            device=device,
            learning_rate=LEARNING_RATE,
            tag_names=NER_TAG_SCHEMES['IOB2']['tags']
        )

        trainer.train(
            train_loader,
            val_loader,
            epochs=EPOCHS,
            patience=3,
            save_path='bilstm_crf_best.pt'
        )

    except ImportError:
        logger.warning("pytorch-crf not available, skipping BiLSTM-CRF training")

    logger.info("\n" + "=" * 70)
    logger.info("NER Training Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
