"""
Text Classification Models
TextCNN, BiLSTM, and BERT-based classifiers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import logging

try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoConfig,
        BertForSequenceClassification,
        RobertaForSequenceClassification,
        DistilBertForSequenceClassification
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCNN(nn.Module):
    """
    Convolutional Neural Network for Text Classification

    Architecture:
    1. Embedding layer
    2. Multiple parallel Conv1D with different kernel sizes
    3. Max pooling
    4. Fully connected layer
    5. Dropout
    6. Output layer

    Paper: "Convolutional Neural Networks for Sentence Classification" (Kim, 2014)

    Best for:
    - Sentence classification
    - Sentiment analysis
    - Topic classification
    - Short text classification

    Advantages:
    - Fast training and inference
    - Captures local patterns (n-grams)
    - Good for fixed-length texts
    - Relatively small model size
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        num_classes: int = 2,
        kernel_sizes: List[int] = [3, 4, 5],
        num_filters: int = 100,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False
    ):
        """
        Initialize TextCNN

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            num_classes: Number of output classes
            kernel_sizes: List of kernel sizes for convolutions
            num_filters: Number of filters per kernel size
            dropout: Dropout probability
            pretrained_embeddings: Pretrained embedding matrix (optional)
            freeze_embeddings: Whether to freeze embedding layer
        """
        super(TextCNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k
            )
            for k in kernel_sizes
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, seq_len]

        Returns:
            Logits [batch_size, num_classes]
        """
        # Embedding: [batch, seq_len] -> [batch, seq_len, emb_dim]
        embedded = self.embedding(x)

        # Transpose for Conv1d: [batch, emb_dim, seq_len]
        embedded = embedded.transpose(1, 2)

        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            # Conv: [batch, emb_dim, seq_len] -> [batch, num_filters, seq_len-k+1]
            conv_out = F.relu(conv(embedded))

            # Max pool: [batch, num_filters, seq_len-k+1] -> [batch, num_filters, 1]
            pooled = F.max_pool1d(conv_out, conv_out.size(2))

            # Squeeze: [batch, num_filters]
            pooled = pooled.squeeze(2)

            conv_outputs.append(pooled)

        # Concatenate: [batch, num_filters * len(kernel_sizes)]
        concat = torch.cat(conv_outputs, dim=1)

        # Dropout
        concat = self.dropout(concat)

        # FC layer: [batch, num_classes]
        logits = self.fc(concat)

        return logits


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for Text Classification

    Architecture:
    1. Embedding layer
    2. Bidirectional LSTM
    3. Attention (optional) or pooling
    4. Fully connected layers
    5. Dropout
    6. Output layer

    Best for:
    - Sequential text understanding
    - Sentiment analysis
    - Document classification
    - Variable-length texts

    Advantages:
    - Captures long-range dependencies
    - Bidirectional context
    - Good for sequential patterns
    - Flexible with text length
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        use_attention: bool = False,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False
    ):
        """
        Initialize BiLSTM

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
            use_attention: Use attention mechanism
            pretrained_embeddings: Pretrained embedding matrix (optional)
            freeze_embeddings: Whether to freeze embedding layer
        """
        super(BiLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Attention layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        if use_attention:
            self.attention = nn.Linear(lstm_output_dim, 1)

        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def attention_forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism

        Args:
            lstm_output: LSTM output [batch, seq_len, hidden_dim]

        Returns:
            Context vector [batch, hidden_dim]
        """
        # Attention scores: [batch, seq_len, 1]
        attention_scores = self.attention(lstm_output)

        # Attention weights: [batch, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)

        # Context: [batch, hidden_dim]
        context = torch.sum(attention_weights * lstm_output, dim=1)

        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, seq_len]

        Returns:
            Logits [batch_size, num_classes]
        """
        # Embedding: [batch, seq_len] -> [batch, seq_len, emb_dim]
        embedded = self.embedding(x)

        # LSTM: [batch, seq_len, emb_dim] -> [batch, seq_len, hidden_dim * 2]
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Apply attention or use last hidden state
        if self.use_attention:
            # Attention pooling
            pooled = self.attention_forward(lstm_out)
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                pooled = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                pooled = hidden[-1]

        # Dropout
        pooled = self.dropout(pooled)

        # FC layers
        out = F.relu(self.fc1(pooled))
        out = self.dropout(out)
        logits = self.fc2(out)

        return logits


class BERTClassifier(nn.Module):
    """
    BERT-based Text Classifier

    Uses pretrained BERT model with a classification head.

    Architecture:
    1. BERT encoder (pretrained)
    2. Pooling ([CLS] token or mean pooling)
    3. Optional intermediate layer
    4. Dropout
    5. Classification head

    Best for:
    - Transfer learning
    - When pretrained knowledge is valuable
    - High-quality classification
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
        num_classes: int = 2,
        dropout: float = 0.1,
        freeze_bert: bool = False,
        hidden_dim: Optional[int] = None
    ):
        """
        Initialize BERT classifier

        Args:
            model_name: HuggingFace model name
            num_classes: Number of output classes
            dropout: Dropout probability
            freeze_bert: Whether to freeze BERT weights
            hidden_dim: Optional intermediate layer dimension
        """
        super(BERTClassifier, self).__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Install with: pip install transformers")

        self.model_name = model_name
        self.num_classes = num_classes

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

        # Classification head
        if hidden_dim is not None:
            self.fc1 = nn.Linear(bert_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_classes)
            self.use_intermediate = True
        else:
            self.classifier = nn.Linear(bert_dim, num_classes)
            self.use_intermediate = False

        logger.info(f"BERT classifier initialized. Output classes: {num_classes}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Token type IDs [batch, seq_len]

        Returns:
            Logits [batch, num_classes]
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Dropout
        pooled_output = self.dropout(pooled_output)

        # Classification
        if self.use_intermediate:
            out = F.relu(self.fc1(pooled_output))
            out = self.dropout(out)
            logits = self.fc2(out)
        else:
            logits = self.classifier(pooled_output)

        return logits


class BERTClassifierSimple(nn.Module):
    """
    Simplified BERT Classifier using HuggingFace's built-in classification models
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_classes: int = 2
    ):
        """
        Initialize simple BERT classifier

        Args:
            model_name: HuggingFace model name
            num_classes: Number of output classes
        """
        super(BERTClassifierSimple, self).__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed. Install with: pip install transformers")

        logger.info(f"Loading {model_name} for sequence classification...")

        # Try to load model-specific classifier
        try:
            if 'bert' in model_name.lower():
                self.model = BertForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_classes
                )
            elif 'roberta' in model_name.lower():
                self.model = RobertaForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_classes
                )
            elif 'distilbert' in model_name.lower():
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_classes
                )
            else:
                # Fallback to AutoModel
                from transformers import AutoModelForSequenceClassification
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_classes
                )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        logger.info(f"Model loaded. Output classes: {num_classes}")

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
            labels: Optional labels for loss calculation

        Returns:
            Model outputs (logits and optional loss)
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Text Classification Models Test")
    print("=" * 70)

    batch_size = 4
    seq_len = 20
    vocab_size = 1000
    num_classes = 3

    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Test TextCNN
    print("\n1. TextCNN")
    print("-" * 70)

    cnn = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=100,
        num_classes=num_classes,
        kernel_sizes=[3, 4, 5],
        num_filters=50
    )

    output = cnn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in cnn.parameters()):,}")

    # Test BiLSTM
    print("\n2. BiLSTM")
    print("-" * 70)

    lstm = BiLSTM(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=64,
        num_layers=2,
        num_classes=num_classes,
        use_attention=False
    )

    output = lstm(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in lstm.parameters()):,}")

    # Test BiLSTM with attention
    print("\n3. BiLSTM with Attention")
    print("-" * 70)

    lstm_attn = BiLSTM(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=64,
        num_layers=2,
        num_classes=num_classes,
        use_attention=True
    )

    output = lstm_attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in lstm_attn.parameters()):,}")

    # Test BERT classifier (if transformers available)
    if TRANSFORMERS_AVAILABLE:
        print("\n4. BERT Classifier")
        print("-" * 70)

        try:
            bert_clf = BERTClassifier(
                model_name='distilbert-base-uncased',
                num_classes=num_classes,
                hidden_dim=128
            )

            # Create dummy BERT input
            bert_input = torch.randint(0, 1000, (batch_size, 32))
            attention_mask = torch.ones(batch_size, 32)

            output = bert_clf(bert_input, attention_mask)
            print(f"Input shape: {bert_input.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Number of parameters: {sum(p.numel() for p in bert_clf.parameters()):,}")

        except Exception as e:
            print(f"BERT test skipped: {e}")
    else:
        print("\n4. BERT Classifier: Not available (transformers not installed)")

    print("\n" + "=" * 70)
    print("Text classification models tested successfully!")
