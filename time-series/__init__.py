"""
Time Series Framework

A comprehensive framework for time series analysis and forecasting including:
- ARIMA/SARIMA
- Prophet
- LSTM models
- Classical forecasting (Exponential Smoothing, Holt-Winters)
- Seasonality detection and decomposition
- Preprocessing and evaluation utilities

Author: ML Framework Team
Version: 1.0.0
"""

# ARIMA models
from .arima import ARIMAModel, SARIMAModel, AutoARIMA, check_stationarity, difference_series

# Prophet
try:
    from .prophet_model import ProphetModel
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# LSTM models
try:
    from .lstm_models import LSTMForecaster, VanillaLSTM, StackedLSTM, BidirectionalLSTM
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# Forecasting models
from .forecasting import (
    SimpleExponentialSmoothing,
    HoltLinearTrend,
    HoltWinters,
    MovingAverage,
    NaiveForecaster
)

# Seasonality
from .seasonality import (
    SeasonalDecomposition,
    STLDecomposition,
    SeasonalityDetector,
    detrend
)

# Utilities
from .utils import (
    train_test_split_ts,
    create_lagged_features,
    normalize_ts,
    denormalize_ts,
    fill_missing,
    evaluate_forecast,
    mse,
    rmse,
    mae,
    mape,
    smape,
    r2_score
)

__all__ = [
    # ARIMA
    'ARIMAModel',
    'SARIMAModel',
    'AutoARIMA',
    'check_stationarity',
    'difference_series',

    # Prophet
    'ProphetModel',
    'PROPHET_AVAILABLE',

    # LSTM
    'LSTMForecaster',
    'VanillaLSTM',
    'StackedLSTM',
    'BidirectionalLSTM',
    'LSTM_AVAILABLE',

    # Forecasting
    'SimpleExponentialSmoothing',
    'HoltLinearTrend',
    'HoltWinters',
    'MovingAverage',
    'NaiveForecaster',

    # Seasonality
    'SeasonalDecomposition',
    'STLDecomposition',
    'SeasonalityDetector',
    'detrend',

    # Utilities
    'train_test_split_ts',
    'create_lagged_features',
    'normalize_ts',
    'denormalize_ts',
    'fill_missing',
    'evaluate_forecast',
    'mse',
    'rmse',
    'mae',
    'mape',
    'smape',
    'r2_score',
]

__version__ = '1.0.0'
__author__ = 'ML Framework Team'
