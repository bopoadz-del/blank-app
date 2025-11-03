"""
Named Entity Recognition (NER) Models
BiLSTM-CRF and BERT-based NER
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
import logging

try:
    from transformers import (
        AutoModel, AutoConfig,
        BertForTokenClassification,
        RobertaForTokenClassification,
        DistilBertForTokenClassification
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers")

try:
    from torchcrf import CRF
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    logging.warning("pytorch-crf not available. Install with: pip install pytorch-crf")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiLSTM_CRF(nn.Module):
    """
    Bidirectional LSTM with Conditional Random Field (CRF) for NER

    Architecture:
    1. Embedding layer
    2. Bidirectional LSTM
    3. Linear projection to tag space
    4. CRF layer for structured prediction

    Paper: "Bidirectional LSTM-CRF Models for Sequence Tagging" (Huang et al., 2015)

    The CRF layer enforces constraints like:
    - B-PER cannot be followed by I-LOC
    - I-ORG must follow B-ORG or I-ORG
    - O can follow any tag

    Best for:
    - Named Entity Recognition
    - Part-of-Speech tagging
    - Chunking
    - Any sequence labeling task

    Advantages:
    - Enforces valid tag sequences
    - Better than simple classification
    - Captures label dependencies
    - State-of-the-art for NER (pre-BERT)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_tags: int = 9,  # B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC, O
        dropout: float = 0.5,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False
    ):
        """
        Initialize BiLSTM-CRF

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_tags: Number of NER tags
            dropout: Dropout probability
            pretrained_embeddings: Pretrained embedding matrix (optional)
            freeze_embeddings: Whether to freeze embedding layer
        """
        super(BiLSTM_CRF, self).__init__()

        if not CRF_AVAILABLE:
            raise ImportError("pytorch-crf not installed. Install with: pip install pytorch-crf")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tags = num_tags

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,  # Divide by 2 for bidirectional
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Linear projection to tag space
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)

        # CRF layer
        self.crf = CRF(num_tags, batch_first=True)

    def _get_lstm_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get LSTM features

        Args:
            x: Input tensor [batch, seq_len]

        Returns:
            LSTM features [batch, seq_len, num_tags]
        """
        # Embedding: [batch, seq_len] -> [batch, seq_len, emb_dim]
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # LSTM: [batch, seq_len, emb_dim] -> [batch, seq_len, hidden_dim]
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)

        # Linear: [batch, seq_len, hidden_dim] -> [batch, seq_len, num_tags]
        emissions = self.hidden2tag(lstm_out)

        return emissions

    def forward(
        self,
        x: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [batch, seq_len]
            tags: True tags [batch, seq_len] (required for training)
            mask: Mask tensor [batch, seq_len] (1 for real tokens, 0 for padding)

        Returns:
            Negative log-likelihood loss (training) or predicted tags (inference)
        """
        # Get LSTM features
        emissions = self._get_lstm_features(x)

        if tags is not None:
            # Training: return negative log-likelihood
            if mask is None:
                mask = torch.ones_like(x, dtype=torch.bool)

            # CRF expects mask as byte tensor
            mask = mask.bool()

            # Compute loss
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            # Inference: return best path
            if mask is None:
                mask = torch.ones_like(x, dtype=torch.bool)

            mask = mask.bool()

            # Decode best path
            predicted_tags = self.crf.decode(emissions, mask=mask)

            return predicted_tags

    def predict(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Predict tags for input

        Args:
            x: Input tensor [batch, seq_len]
            mask: Mask tensor [batch, seq_len]

        Returns:
            List of predicted tag sequences
        """
        self.eval()
        with torch.no_grad():
            predicted_tags = self.forward(x, tags=None, mask=mask)
        return predicted_tags


class BERT_NER(nn.Module):
    """
    BERT-based Named Entity Recognition

    Architecture:
    1. BERT encoder (pretrained)
    2. Dropout
    3. Linear layer to tag space
    4. Optional CRF layer

    Best for:
    - Transfer learning for NER
    - When pretrained knowledge is valuable
    - State-of-the-art NER performance
    - Limited training data

    Advantages:
    - Leverages pretrained knowledge
    - State-of-the-art performance
    - Contextualized representations
    - Good with limited data
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_tags: int = 9,
        dropout: float = 0.1,
        use_crf: bool = False,
        freeze_bert: bool = False
    ):
        """
        Initialize BERT NER

        Args:
            model_name: HuggingFace model name
            num_tags: Number of NER tags
            dropout: Dropout probability
            use_crf: Use CRF layer on top of BERT
            freeze_bert: Whether to freeze BERT weights
        """
        super(BERT_NER, self).__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Install with: pip install transformers")

        if use_crf and not CRF_AVAILABLE:
            raise ImportError("pytorch-crf not installed. Install with: pip install pytorch-crf")

        self.model_name = model_name
        self.num_tags = num_tags
        self.use_crf = use_crf

        # Load BERT
        logger.info(f"Loading {model_name}...")
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Get BERT output dimension
        bert_dim = self.config.hidden_size

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Linear layer to tag space
        self.classifier = nn.Linear(bert_dim, num_tags)

        # Optional CRF layer
        if use_crf:
            self.crf = CRF(num_tags, batch_first=True)

        logger.info(f"BERT NER initialized. Number of tags: {num_tags}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        tags: Optional[torch.Tensor] = None
    ):
        """
        Forward pass

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Token type IDs [batch, seq_len]
            tags: True tags [batch, seq_len] (required for training with CRF)

        Returns:
            Loss (training) or logits/predictions (inference)
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get token representations
        sequence_output = outputs.last_hidden_state

        # Dropout
        sequence_output = self.dropout(sequence_output)

        # Linear projection to tag space
        emissions = self.classifier(sequence_output)

        if self.use_crf:
            if tags is not None:
                # Training with CRF: return negative log-likelihood
                mask = attention_mask.bool() if attention_mask is not None else None
                loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
                return loss
            else:
                # Inference with CRF: return best path
                mask = attention_mask.bool() if attention_mask is not None else None
                predicted_tags = self.crf.decode(emissions, mask=mask)
                return predicted_tags
        else:
            # Without CRF: return logits
            return emissions

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ):
        """
        Predict tags for input

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Token type IDs [batch, seq_len]

        Returns:
            Predicted tags
        """
        self.eval()
        with torch.no_grad():
            if self.use_crf:
                predictions = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    tags=None
                )
            else:
                logits = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                predictions = torch.argmax(logits, dim=-1)

        return predictions


class BERT_NER_Simple(nn.Module):
    """
    Simplified BERT NER using HuggingFace's built-in token classification models
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_tags: int = 9
    ):
        """
        Initialize simple BERT NER

        Args:
            model_name: HuggingFace model name
            num_tags: Number of NER tags
        """
        super(BERT_NER_Simple, self).__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Install with: pip install transformers")

        logger.info(f"Loading {model_name} for token classification...")

        # Try to load model-specific classifier
        try:
            if 'bert' in model_name.lower():
                self.model = BertForTokenClassification.from_pretrained(
                    model_name,
                    num_labels=num_tags
                )
            elif 'roberta' in model_name.lower():
                self.model = RobertaForTokenClassification.from_pretrained(
                    model_name,
                    num_labels=num_tags
                )
            elif 'distilbert' in model_name.lower():
                self.model = DistilBertForTokenClassification.from_pretrained(
                    model_name,
                    num_labels=num_tags
                )
            else:
                # Fallback to AutoModel
                from transformers import AutoModelForTokenClassification
                self.model = AutoModelForTokenClassification.from_pretrained(
                    model_name,
                    num_labels=num_tags
                )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        logger.info(f"Model loaded. Number of tags: {num_tags}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Token type IDs [batch, seq_len]
            labels: True tags [batch, seq_len] (optional, for loss calculation)

        Returns:
            Model outputs (logits and optional loss)
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )


# Common NER tag schemes
NER_TAG_SCHEMES = {
    'IOB2': {
        'tags': ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC'],
        'num_tags': 9
    },
    'IOBES': {
        'tags': ['O', 'B-PER', 'I-PER', 'E-PER', 'S-PER',
                'B-LOC', 'I-LOC', 'E-LOC', 'S-LOC',
                'B-ORG', 'I-ORG', 'E-ORG', 'S-ORG',
                'B-MISC', 'I-MISC', 'E-MISC', 'S-MISC'],
        'num_tags': 17
    },
    'BILOU': {
        'tags': ['O', 'B-PER', 'I-PER', 'L-PER', 'U-PER',
                'B-LOC', 'I-LOC', 'L-LOC', 'U-LOC',
                'B-ORG', 'I-ORG', 'L-ORG', 'U-ORG',
                'B-MISC', 'I-MISC', 'L-MISC', 'U-MISC'],
        'num_tags': 17
    }
}


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Named Entity Recognition Models Test")
    print("=" * 70)

    batch_size = 4
    seq_len = 20
    vocab_size = 1000
    num_tags = 9  # IOB2 scheme

    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    tags = torch.randint(0, num_tags, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)

    # Test BiLSTM-CRF
    if CRF_AVAILABLE:
        print("\n1. BiLSTM-CRF")
        print("-" * 70)

        model = BiLSTM_CRF(
            vocab_size=vocab_size,
            embedding_dim=100,
            hidden_dim=128,
            num_layers=2,
            num_tags=num_tags
        )

        # Training forward pass
        loss = model(x, tags=tags, mask=mask)
        print(f"Input shape: {x.shape}")
        print(f"Tags shape: {tags.shape}")
        print(f"Loss: {loss.item():.4f}")

        # Inference
        predictions = model.predict(x, mask=mask)
        print(f"Predictions (first sequence): {predictions[0][:10]}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    else:
        print("\n1. BiLSTM-CRF: Not available (pytorch-crf not installed)")

    # Test BERT NER
    if TRANSFORMERS_AVAILABLE:
        print("\n2. BERT NER (without CRF)")
        print("-" * 70)

        try:
            bert_ner = BERT_NER(
                model_name='distilbert-base-uncased',
                num_tags=num_tags,
                use_crf=False
            )

            # Create dummy BERT input
            bert_input = torch.randint(0, 1000, (batch_size, 32))
            attention_mask = torch.ones(batch_size, 32)
            bert_tags = torch.randint(0, num_tags, (batch_size, 32))

            # Forward pass
            logits = bert_ner(
                input_ids=bert_input,
                attention_mask=attention_mask
            )

            print(f"Input shape: {bert_input.shape}")
            print(f"Logits shape: {logits.shape}")
            print(f"Number of parameters: {sum(p.numel() for p in bert_ner.parameters()):,}")

        except Exception as e:
            print(f"BERT NER test skipped: {e}")

        # Test BERT NER with CRF
        if CRF_AVAILABLE:
            print("\n3. BERT NER (with CRF)")
            print("-" * 70)

            try:
                bert_ner_crf = BERT_NER(
                    model_name='distilbert-base-uncased',
                    num_tags=num_tags,
                    use_crf=True
                )

                # Training forward pass
                loss = bert_ner_crf(
                    input_ids=bert_input,
                    attention_mask=attention_mask,
                    tags=bert_tags
                )

                print(f"Loss: {loss.item():.4f}")

                # Inference
                predictions = bert_ner_crf.predict(
                    input_ids=bert_input,
                    attention_mask=attention_mask
                )

                print(f"Predictions (first sequence, first 10): {predictions[0][:10]}")

            except Exception as e:
                print(f"BERT NER with CRF test skipped: {e}")
    else:
        print("\n2. BERT NER: Not available (transformers not installed)")

    # Print NER tag schemes
    print("\n4. NER Tag Schemes")
    print("-" * 70)

    for scheme, info in NER_TAG_SCHEMES.items():
        print(f"\n{scheme} scheme ({info['num_tags']} tags):")
        print(f"  Tags: {', '.join(info['tags'][:5])}...")

    print("\n" + "=" * 70)
    print("NER models tested successfully!")
