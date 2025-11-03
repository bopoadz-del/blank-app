"""
Recurrent Neural Network Architectures
LSTM, BiLSTM, and GRU implementations for sequence modeling
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTM(nn.Module):
    """
    LSTM (Long Short-Term Memory) Network
    For sequence classification or prediction tasks
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.5,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through LSTM

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional initial hidden state

        Returns:
            Output tensor of shape (batch, num_classes)
        """
        batch_size = x.size(0)

        # Initialize hidden state if not provided
        if hidden is None:
            num_directions = 2 if self.bidirectional else 1
            h0 = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size
            ).to(x.device)
            c0 = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size
            ).to(x.device)
            hidden = (h0, c0)

        # LSTM forward pass
        # out: (batch, seq_len, hidden_size * num_directions)
        out, (hn, cn) = self.lstm(x, hidden)

        # Take output from last time step
        out = out[:, -1, :]

        # Dropout
        out = self.dropout(out)

        # Fully connected layer
        out = self.fc(out)

        return out

    def forward_sequence(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass returning full sequence output

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional initial hidden state

        Returns:
            Output tensor of shape (batch, seq_len, num_classes)
        """
        batch_size = x.size(0)

        # Initialize hidden state if not provided
        if hidden is None:
            num_directions = 2 if self.bidirectional else 1
            h0 = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size
            ).to(x.device)
            c0 = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size
            ).to(x.device)
            hidden = (h0, c0)

        # LSTM forward pass
        out, _ = self.lstm(x, hidden)

        # Apply FC to each time step
        # Reshape: (batch * seq_len, hidden_size)
        batch_size, seq_len, _ = out.size()
        out = out.contiguous().view(batch_size * seq_len, -1)

        # FC layer
        out = self.fc(out)

        # Reshape back: (batch, seq_len, num_classes)
        out = out.view(batch_size, seq_len, self.num_classes)

        return out


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM
    Processes sequence in both forward and backward directions
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.5
    ):
        """
        Initialize BiLSTM

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(BiLSTM, self).__init__()

        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through BiLSTM"""
        return self.lstm(x)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning full sequence"""
        return self.lstm.forward_sequence(x)


class GRU(nn.Module):
    """
    GRU (Gated Recurrent Unit) Network
    Simpler alternative to LSTM with fewer parameters
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.5,
        bidirectional: bool = False
    ):
        """
        Initialize GRU

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of GRU layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Use bidirectional GRU
        """
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GRU

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            hidden: Optional initial hidden state

        Returns:
            Output tensor of shape (batch, num_classes)
        """
        batch_size = x.size(0)

        # Initialize hidden state if not provided
        if hidden is None:
            num_directions = 2 if self.bidirectional else 1
            hidden = torch.zeros(
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size
            ).to(x.device)

        # GRU forward pass
        out, _ = self.gru(x, hidden)

        # Take output from last time step
        out = out[:, -1, :]

        # Dropout
        out = self.dropout(out)

        # Fully connected layer
        out = self.fc(out)

        return out


class LSTMWithAttention(nn.Module):
    """
    LSTM with Attention Mechanism
    Applies attention over sequence before classification
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.5
    ):
        """
        Initialize LSTM with Attention

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(LSTMWithAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # LSTM forward pass
        # out: (batch, seq_len, hidden_size * 2)
        out, _ = self.lstm(x)

        # Compute attention weights
        # attention_scores: (batch, seq_len, 1)
        attention_scores = self.attention(out)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Apply attention
        # context: (batch, hidden_size * 2)
        context = torch.sum(attention_weights * out, dim=1)

        # Dropout
        context = self.dropout(context)

        # Fully connected layer
        output = self.fc(context)

        return output


# Example usage
if __name__ == "__main__":
    # Parameters
    batch_size = 32
    seq_len = 100
    input_size = 50
    hidden_size = 128
    num_layers = 2
    num_classes = 10

    # Create LSTM model
    model = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes
    )

    # Dummy input
    x = torch.randn(batch_size, seq_len, input_size)

    # Forward pass
    output = model(x)
    print(f"LSTM Input shape: {x.shape}")
    print(f"LSTM Output shape: {output.shape}")

    # Create BiLSTM model
    bilstm_model = BiLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes
    )

    # Forward pass
    output = bilstm_model(x)
    print(f"\nBiLSTM Input shape: {x.shape}")
    print(f"BiLSTM Output shape: {output.shape}")

    # Create LSTM with Attention
    attention_model = LSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes
    )

    # Forward pass
    output = attention_model(x)
    print(f"\nLSTM+Attention Input shape: {x.shape}")
    print(f"LSTM+Attention Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nLSTM Total parameters: {total_params:,}")
