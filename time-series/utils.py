"""
Time Series Utilities

Preprocessing, evaluation metrics, and utility functions for time series analysis.

Author: ML Framework Team
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
import warnings


# ============================================================================
# PREPROCESSING
# ============================================================================

def train_test_split_ts(
    data: Union[pd.Series, np.ndarray],
    test_size: float = 0.2,
    return_index: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple]:
    """
    Split time series into train and test sets (preserving temporal order).

    Parameters:
    -----------
    data : pd.Series or np.ndarray
        Time series data.
    test_size : float, default=0.2
        Proportion of data to use for testing.
    return_index : bool, default=False
        Whether to return indices.

    Returns:
    --------
    train, test : arrays or Series
        Training and test data.
    """
    split_idx = int(len(data) * (1 - test_size))

    if isinstance(data, pd.Series):
        train = data.iloc[:split_idx]
        test = data.iloc[split_idx:]
    else:
        train = data[:split_idx]
        test = data[split_idx:]

    if return_index:
        return train, test, split_idx
    return train, test


def create_lagged_features(
    data: Union[pd.Series, np.ndarray],
    lags: int = 1
) -> pd.DataFrame:
    """
    Create lagged features for time series.

    Parameters:
    -----------
    data : pd.Series or np.ndarray
        Time series data.
    lags : int, default=1
        Number of lags to create.

    Returns:
    --------
    df : pd.DataFrame
        DataFrame with lagged features.
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    df = pd.DataFrame()
    df['value'] = data.values

    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = data.shift(lag).values

    return df.dropna()


def normalize_ts(
    data: Union[pd.Series, np.ndarray],
    method: str = 'minmax'
) -> Tuple[np.ndarray, dict]:
    """
    Normalize time series data.

    Parameters:
    -----------
    data : pd.Series or np.ndarray
        Time series data.
    method : str, default='minmax'
        Normalization method: 'minmax', 'zscore', 'robust'.

    Returns:
    --------
    normalized : np.ndarray
        Normalized data.
    params : dict
        Normalization parameters for inverse transform.
    """
    if isinstance(data, pd.Series):
        data = data.values

    if method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val, 'method': 'minmax'}

    elif method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        normalized = (data - mean) / (std + 1e-8)
        params = {'mean': mean, 'std': std, 'method': 'zscore'}

    elif method == 'robust':
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        normalized = (data - median) / (iqr + 1e-8)
        params = {'median': median, 'iqr': iqr, 'method': 'robust'}

    else:
        raise ValueError(f"Unknown method: {method}")

    return normalized, params


def denormalize_ts(data: np.ndarray, params: dict) -> np.ndarray:
    """
    Denormalize time series data.

    Parameters:
    -----------
    data : np.ndarray
        Normalized data.
    params : dict
        Normalization parameters from normalize_ts().

    Returns:
    --------
    denormalized : np.ndarray
        Original scale data.
    """
    method = params['method']

    if method == 'minmax':
        return data * (params['max'] - params['min']) + params['min']

    elif method == 'zscore':
        return data * params['std'] + params['mean']

    elif method == 'robust':
        return data * params['iqr'] + params['median']

    else:
        raise ValueError(f"Unknown method: {method}")


def fill_missing(
    data: Union[pd.Series, np.ndarray],
    method: str = 'forward'
) -> Union[pd.Series, np.ndarray]:
    """
    Fill missing values in time series.

    Parameters:
    -----------
    data : pd.Series or np.ndarray
        Time series with missing values.
    method : str, default='forward'
        Fill method: 'forward', 'backward', 'linear', 'mean'.

    Returns:
    --------
    filled : same type as input
        Time series with missing values filled.
    """
    is_series = isinstance(data, pd.Series)

    if not is_series:
        data = pd.Series(data)

    if method == 'forward':
        filled = data.fillna(method='ffill')
    elif method == 'backward':
        filled = data.fillna(method='bfill')
    elif method == 'linear':
        filled = data.interpolate(method='linear')
    elif method == 'mean':
        filled = data.fillna(data.mean())
    else:
        raise ValueError(f"Unknown method: {method}")

    return filled if is_series else filled.values


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (Coefficient of Determination)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[list] = None
) -> dict:
    """
    Comprehensive forecast evaluation.

    Parameters:
    -----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.
    metrics : list, optional
        List of metrics to compute.
        Options: 'mse', 'rmse', 'mae', 'mape', 'smape', 'r2'.
        If None, computes all.

    Returns:
    --------
    results : dict
        Dictionary of metric values.
    """
    if metrics is None:
        metrics = ['mse', 'rmse', 'mae', 'mape', 'smape', 'r2']

    results = {}

    if 'mse' in metrics:
        results['mse'] = mse(y_true, y_pred)

    if 'rmse' in metrics:
        results['rmse'] = rmse(y_true, y_pred)

    if 'mae' in metrics:
        results['mae'] = mae(y_true, y_pred)

    if 'mape' in metrics:
        try:
            results['mape'] = mape(y_true, y_pred)
        except:
            results['mape'] = np.nan

    if 'smape' in metrics:
        try:
            results['smape'] = smape(y_true, y_pred)
        except:
            results['smape'] = np.nan

    if 'r2' in metrics:
        results['r2'] = r2_score(y_true, y_pred)

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TIME SERIES UTILITIES EXAMPLES")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    data = np.cumsum(np.random.randn(100)) + 100

    # Example 1: Train-Test Split
    print("\n1. Train-Test Split")
    print("-" * 70)
    train, test = train_test_split_ts(data, test_size=0.2)
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    # Example 2: Create Lagged Features
    print("\n2. Lagged Features")
    print("-" * 70)
    df = create_lagged_features(data[:10], lags=3)
    print(df.head())

    # Example 3: Normalization
    print("\n3. Normalization")
    print("-" * 70)
    normalized, params = normalize_ts(data, method='minmax')
    print(f"Original range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")

    denormalized = denormalize_ts(normalized, params)
    print(f"Denormalized matches original: {np.allclose(data, denormalized)}")

    # Example 4: Evaluation Metrics
    print("\n4. Evaluation Metrics")
    print("-" * 70)
    y_true = data[:50]
    y_pred = data[:50] + np.random.randn(50) * 2

    metrics = evaluate_forecast(y_true, y_pred)
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper()}: {value:.4f}")

    print("\n" + "=" * 70)
    print("All utility examples completed!")
    print("=" * 70)
