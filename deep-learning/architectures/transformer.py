"""
Transformer Architecture
Multi-head attention and transformer blocks for sequence modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Mechanism
    Allows model to attend to different representation subspaces
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize Multi-Head Attention

        Args:
            d_model: Dimension of model
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # Output projection
        self.out_linear = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Forward pass through multi-head attention

        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_linear(query)  # (batch, seq_len, d_model)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # Split into multiple heads
        # (batch, seq_len, num_heads, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k)
        K = K.view(batch_size, -1, self.num_heads, self.d_k)
        V = V.view(batch_size, -1, self.num_heads, self.d_k)

        # Transpose for attention
        # (batch, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        # scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        # (batch, num_heads, seq_len, d_k)
        x = torch.matmul(attention_weights, V)

        # Concatenate heads
        # (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.d_model)

        # Output projection
        x = self.out_linear(x)

        return x, attention_weights


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    Applied to each position independently
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Initialize feed-forward network

        Args:
            d_model: Dimension of model
            d_ff: Dimension of feed-forward network
            dropout: Dropout probability
        """
        super(PositionwiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feed-forward network"""
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Positional Encoding
    Adds position information to input embeddings
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        Initialize positional encoding

        Args:
            d_model: Dimension of model
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block
    Multi-head attention + feed-forward with residual connections
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Initialize transformer block

        Args:
            d_model: Dimension of model
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout: Dropout probability
        """
        super(TransformerBlock, self).__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Multi-head attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


class Transformer(nn.Module):
    """
    Complete Transformer Model
    For sequence classification or generation tasks
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        num_classes: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        pooling: str = 'mean'  # 'mean', 'max', or 'cls'
    ):
        """
        Initialize Transformer

        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of model
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Dimension of feed-forward network
            num_classes: Number of output classes
            max_len: Maximum sequence length
            dropout: Dropout probability
            pooling: Pooling strategy ('mean', 'max', or 'cls')
        """
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.pooling = pooling

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Classifier
        self.fc = nn.Linear(d_model, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model ** -0.5)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer

        Args:
            x: Input tensor (batch, seq_len)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, num_classes)
        """
        # Embedding
        x = self.embedding(x) * math.sqrt(self.d_model)

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        # Pooling
        if self.pooling == 'mean':
            x = torch.mean(x, dim=1)
        elif self.pooling == 'max':
            x = torch.max(x, dim=1)[0]
        elif self.pooling == 'cls':
            x = x[:, 0, :]  # Take first token (CLS token)

        # Dropout
        x = self.dropout(x)

        # Classifier
        x = self.fc(x)

        return x

    def extract_features(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract features without classification

        Args:
            x: Input tensor (batch, seq_len)
            mask: Optional attention mask

        Returns:
            Feature tensor (batch, seq_len, d_model)
        """
        # Embedding
        x = self.embedding(x) * math.sqrt(self.d_model)

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        return x


class TransformerForSequenceToSequence(nn.Module):
    """
    Transformer for Sequence-to-Sequence Tasks
    With encoder and decoder
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        d_ff: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        Initialize Seq2Seq Transformer

        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Dimension of model
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Dimension of feed-forward network
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(TransformerForSequenceToSequence, self).__init__()

        # PyTorch built-in Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Output projection
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            src: Source sequence (batch, src_len)
            tgt: Target sequence (batch, tgt_len)
            src_mask: Source mask
            tgt_mask: Target mask

        Returns:
            Output tensor (batch, tgt_len, tgt_vocab_size)
        """
        # Embeddings
        src = self.src_embedding(src) * math.sqrt(self.transformer.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.transformer.d_model)

        # Positional encoding
        src = self.pos_encoding(src)
        tgt = self.pos_encoding(tgt)

        # Transformer
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

        # Output projection
        output = self.fc(output)

        return output


# Example usage
if __name__ == "__main__":
    # Parameters
    batch_size = 32
    seq_len = 50
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    num_classes = 10

    # Create Transformer model
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        num_classes=num_classes
    )

    # Dummy input (token indices)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    output = model(x)
    print(f"Transformer Input shape: {x.shape}")
    print(f"Transformer Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test multi-head attention
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    query = key = value = torch.randn(batch_size, seq_len, d_model)
    output, attention_weights = mha(query, key, value)
    print(f"\nMulti-Head Attention Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
