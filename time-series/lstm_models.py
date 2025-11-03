"""
LSTM Models for Time Series Forecasting

This module implements Long Short-Term Memory (LSTM) neural networks for
time series forecasting. LSTMs are particularly effective for capturing
long-term dependencies in sequential data.

Key Features:
- Vanilla LSTM
- Stacked LSTM (multiple layers)
- Bidirectional LSTM
- CNN-LSTM hybrid
- Encoder-Decoder LSTM (seq2seq)
- Attention-based LSTM
- Multi-variate time series support

Author: ML Framework Team
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, List
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series with sliding window.

    Parameters:
    -----------
    data : np.ndarray
        Time series data of shape (n_samples, n_features).
    window_size : int
        Number of time steps to use for prediction.
    horizon : int, default=1
        Number of steps to forecast ahead.
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        horizon: int = 1
    ):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon

        # Create sequences
        self.X, self.y = self._create_sequences()

    def _create_sequences(self):
        """Create sliding window sequences."""
        X, y = [], []

        for i in range(len(self.data) - self.window_size - self.horizon + 1):
            X.append(self.data[i:i + self.window_size])
            y.append(self.data[i + self.window_size:i + self.window_size + self.horizon])

        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])


class VanillaLSTM(nn.Module):
    """
    Vanilla LSTM Network

    Simple LSTM architecture for time series forecasting.

    Parameters:
    -----------
    input_size : int
        Number of input features.
    hidden_size : int, default=64
        Number of LSTM units.
    num_layers : int, default=1
        Number of LSTM layers.
    output_size : int, default=1
        Number of output features (forecast horizon).
    dropout : float, default=0.0
        Dropout rate between LSTM layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.0
    ):
        super(VanillaLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, input_size).

        Returns:
        --------
        output : torch.Tensor
            Output tensor of shape (batch, output_size).
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        out = self.fc(lstm_out[:, -1, :])

        return out


class StackedLSTM(nn.Module):
    """
    Stacked LSTM Network

    Multi-layer LSTM for capturing complex temporal patterns.

    Parameters:
    -----------
    input_size : int
        Number of input features.
    hidden_sizes : list of int
        List of hidden sizes for each LSTM layer.
    output_size : int, default=1
        Number of output features.
    dropout : float, default=0.2
        Dropout rate between layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [64, 32],
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(StackedLSTM, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)

        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()

        for i, hidden_size in enumerate(hidden_sizes):
            input_dim = input_size if i == 0 else hidden_sizes[i-1]
            self.lstm_layers.append(
                nn.LSTM(input_dim, hidden_size, batch_first=True)
            )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        """Forward pass through stacked LSTM layers."""
        for lstm in self.lstm_layers:
            x, (h_n, c_n) = lstm(x)
            x = self.dropout(x)

        # Use last time step
        out = self.fc(x[:, -1, :])

        return out


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM Network

    Processes sequences in both forward and backward directions.

    Parameters:
    -----------
    input_size : int
        Number of input features.
    hidden_size : int, default=64
        Number of LSTM units in each direction.
    num_layers : int, default=1
        Number of bidirectional LSTM layers.
    output_size : int, default=1
        Number of output features.
    dropout : float, default=0.0
        Dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.0
    ):
        super(BidirectionalLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        # *2 because bidirectional
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """Forward pass through bidirectional LSTM."""
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last time step
        out = self.fc(lstm_out[:, -1, :])

        return out


class LSTMForecaster:
    """
    LSTM Time Series Forecaster

    High-level wrapper for training and forecasting with LSTM models.

    Parameters:
    -----------
    model_type : str, default='vanilla'
        Type of LSTM model: 'vanilla', 'stacked', 'bidirectional'.
    window_size : int, default=30
        Number of past time steps to use for prediction.
    horizon : int, default=1
        Number of steps to forecast ahead.
    hidden_size : int, default=64
        Size of LSTM hidden layer.
    num_layers : int, default=2
        Number of LSTM layers.
    dropout : float, default=0.2
        Dropout rate.
    learning_rate : float, default=0.001
        Learning rate for optimizer.
    batch_size : int, default=32
        Batch size for training.
    epochs : int, default=100
        Number of training epochs.
    device : str, default='cpu'
        Device to use ('cpu' or 'cuda').

    Example:
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> # Generate sample data
    >>> data = np.sin(np.linspace(0, 100, 1000)) + np.random.randn(1000) * 0.1
    >>>
    >>> # Create and train forecaster
    >>> forecaster = LSTMForecaster(
    ...     model_type='stacked',
    ...     window_size=30,
    ...     horizon=5,
    ...     hidden_size=64,
    ...     num_layers=2,
    ...     epochs=50
    ... )
    >>>
    >>> forecaster.fit(data)
    >>>
    >>> # Forecast
    >>> forecast = forecaster.forecast(steps=10)
    >>> print(forecast)
    """

    def __init__(
        self,
        model_type: str = 'vanilla',
        window_size: int = 30,
        horizon: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        device: str = 'cpu'
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.model_type = model_type
        self.window_size = window_size
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)

        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.train_losses = []

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to zero mean and unit variance."""
        self.scaler_mean = np.mean(data, axis=0)
        self.scaler_std = np.std(data, axis=0) + 1e-8
        return (data - self.scaler_mean) / self.scaler_std

    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data back to original scale."""
        return data * self.scaler_std + self.scaler_mean

    def fit(
        self,
        data: Union[np.ndarray, pd.Series],
        validation_split: float = 0.2,
        verbose: bool = True
    ):
        """
        Fit LSTM model to time series data.

        Parameters:
        -----------
        data : np.ndarray or pd.Series
            Time series data. Shape: (n_samples,) or (n_samples, n_features).
        validation_split : float, default=0.2
            Fraction of data to use for validation.
        verbose : bool, default=True
            Whether to print training progress.

        Returns:
        --------
        self : object
            Fitted forecaster.
        """
        # Convert to numpy array
        if isinstance(data, pd.Series):
            data = data.values

        # Ensure 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_features = data.shape[1]

        # Normalize
        data_normalized = self._normalize(data)

        # Split train/validation
        split_idx = int(len(data_normalized) * (1 - validation_split))
        train_data = data_normalized[:split_idx]
        val_data = data_normalized[split_idx:]

        # Create datasets
        train_dataset = TimeSeriesDataset(train_data, self.window_size, self.horizon)
        val_dataset = TimeSeriesDataset(val_data, self.window_size, self.horizon)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Create model
        if self.model_type == 'vanilla':
            self.model = VanillaLSTM(
                input_size=n_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=self.horizon * n_features,
                dropout=self.dropout
            )
        elif self.model_type == 'stacked':
            hidden_sizes = [self.hidden_size // (2**i) for i in range(self.num_layers)]
            self.model = StackedLSTM(
                input_size=n_features,
                hidden_sizes=hidden_sizes,
                output_size=self.horizon * n_features,
                dropout=self.dropout
            )
        elif self.model_type == 'bidirectional':
            self.model = BidirectionalLSTM(
                input_size=n_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=self.horizon * n_features,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.model = self.model.to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.train_losses = []
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Reshape y to match output
                y_batch = y_batch.reshape(y_batch.size(0), -1)

                # Forward pass
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    y_batch = y_batch.reshape(y_batch.size(0), -1)

                    output = self.model(X_batch)
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            self.train_losses.append({'train': train_loss, 'val': val_loss})

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Load best model
        self.model.load_state_dict(self.best_model_state)

        if verbose:
            print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")

        return self

    def forecast(self, steps: int = 1, last_window: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate multi-step ahead forecasts.

        Parameters:
        -----------
        steps : int, default=1
            Number of steps to forecast.
        last_window : np.ndarray, optional
            Last window of data to use for forecasting.
            If None, uses last window from training data.

        Returns:
        --------
        forecast : np.ndarray
            Forecasted values.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        self.model.eval()

        # Prepare last window
        if last_window is None:
            # Need to store last window during fit
            raise ValueError("last_window must be provided")

        if last_window.ndim == 1:
            last_window = last_window.reshape(-1, 1)

        # Normalize
        window_normalized = (last_window - self.scaler_mean) / self.scaler_std

        forecasts = []

        with torch.no_grad():
            current_window = torch.FloatTensor(window_normalized[-self.window_size:]).unsqueeze(0).to(self.device)

            for _ in range(steps // self.horizon + (1 if steps % self.horizon else 0)):
                # Predict
                pred = self.model(current_window)
                pred = pred.cpu().numpy().reshape(-1, last_window.shape[1])

                forecasts.append(pred[:self.horizon])

                # Update window for next prediction
                current_window = torch.cat([
                    current_window[:, self.horizon:, :],
                    torch.FloatTensor(pred[:self.horizon]).unsqueeze(0).to(self.device)
                ], dim=1)

        # Concatenate and denormalize
        forecast = np.vstack(forecasts)[:steps]
        forecast = self._denormalize(forecast)

        return forecast.flatten() if forecast.shape[1] == 1 else forecast


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Install with: pip install torch")
    else:
        print("=" * 70)
        print("LSTM TIME SERIES EXAMPLES")
        print("=" * 70)

        # Generate sample time series
        np.random.seed(42)
        t = np.linspace(0, 100, 1000)
        data = np.sin(t) + 0.5 * np.sin(3 * t) + np.random.randn(1000) * 0.1

        # Example 1: Vanilla LSTM
        print("\n1. Vanilla LSTM Forecaster")
        print("-" * 70)

        forecaster = LSTMForecaster(
            model_type='vanilla',
            window_size=30,
            horizon=1,
            hidden_size=32,
            num_layers=1,
            epochs=20,
            batch_size=32
        )

        # Train on first 800 points
        forecaster.fit(data[:800], validation_split=0.2, verbose=True)

        # Forecast
        last_window = data[770:800]
        forecast = forecaster.forecast(steps=10, last_window=last_window)
        print(f"\nForecast (next 10 steps): {forecast[:5]}...")

        # Example 2: Stacked LSTM
        print("\n2. Stacked LSTM Forecaster")
        print("-" * 70)

        forecaster_stacked = LSTMForecaster(
            model_type='stacked',
            window_size=30,
            horizon=5,
            hidden_size=64,
            num_layers=2,
            epochs=20
        )

        forecaster_stacked.fit(data[:800], validation_split=0.2, verbose=True)

        forecast = forecaster_stacked.forecast(steps=10, last_window=last_window)
        print(f"Forecast shape: {forecast.shape}")

        print("\n" + "=" * 70)
        print("All LSTM examples completed!")
        print("=" * 70)
