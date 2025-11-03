"""
Classical Forecasting Models

This module implements classical statistical forecasting methods including
Exponential Smoothing, Holt's Linear Trend, and Holt-Winters Seasonal methods.

Key Features:
- Simple Exponential Smoothing (SES)
- Holt's Linear Trend Model
- Holt-Winters Seasonal Model (additive and multiplicative)
- Moving Average
- Weighted Moving Average
- Naive forecasting methods

Author: ML Framework Team
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
import warnings

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
    from statsmodels.tsa.forecasting.theta import ThetaModel
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Install with: pip install statsmodels")


class SimpleExponentialSmoothing:
    """
    Simple Exponential Smoothing (SES)

    Suitable for forecasting data with no trend or seasonality.
    Uses weighted averages where weights decay exponentially.

    Parameters:
    -----------
    alpha : float, optional
        Smoothing parameter (0 < alpha < 1).
        If None, optimized automatically.

    Attributes:
    -----------
    model_ : SimpleExpSmoothing
        Fitted statsmodels SimpleExpSmoothing model.
    fitted_values_ : pd.Series
        Fitted values.
    alpha_ : float
        Smoothing parameter used.

    Example:
    --------
    >>> ses = SimpleExponentialSmoothing(alpha=0.3)
    >>> ses.fit(data)
    >>> forecast = ses.forecast(steps=10)
    """

    def __init__(self, alpha: Optional[float] = None):
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required. Install with: pip install statsmodels")

        self.alpha = alpha
        self.model_ = None
        self.result_ = None
        self.fitted_values_ = None
        self.alpha_ = None

    def fit(self, y: Union[pd.Series, np.ndarray], verbose: bool = True):
        """
        Fit Simple Exponential Smoothing model.

        Parameters:
        -----------
        y : pd.Series or np.ndarray
            Time series data.
        verbose : bool, default=True
            Print fitting information.

        Returns:
        --------
        self : object
            Fitted model.
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        self.model_ = SimpleExpSmoothing(y)

        if self.alpha is not None:
            self.result_ = self.model_.fit(smoothing_level=self.alpha)
        else:
            self.result_ = self.model_.fit(optimized=True)

        self.fitted_values_ = self.result_.fittedvalues
        self.alpha_ = self.result_.params['smoothing_level']

        if verbose:
            print(f"Simple Exponential Smoothing fitted")
            print(f"Alpha: {self.alpha_:.4f}")

        return self

    def forecast(self, steps: int = 1) -> pd.Series:
        """Forecast future values."""
        if self.result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.result_.forecast(steps=steps)


class HoltLinearTrend:
    """
    Holt's Linear Trend Model

    Extension of SES to capture linear trends.

    Parameters:
    -----------
    alpha : float, optional
        Level smoothing parameter.
    beta : float, optional
        Trend smoothing parameter.
    damped : bool, default=False
        Whether to use damped trend.

    Example:
    --------
    >>> holt = HoltLinearTrend(damped=True)
    >>> holt.fit(data)
    >>> forecast = holt.forecast(steps=20)
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        damped: bool = False
    ):
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required")

        self.alpha = alpha
        self.beta = beta
        self.damped = damped
        self.model_ = None
        self.result_ = None

    def fit(self, y: Union[pd.Series, np.ndarray], verbose: bool = True):
        """Fit Holt's Linear Trend model."""
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        self.model_ = Holt(y, damped_trend=self.damped)

        if self.alpha is not None and self.beta is not None:
            self.result_ = self.model_.fit(
                smoothing_level=self.alpha,
                smoothing_trend=self.beta
            )
        else:
            self.result_ = self.model_.fit(optimized=True)

        if verbose:
            print(f"Holt's Linear Trend fitted")
            print(f"Alpha: {self.result_.params['smoothing_level']:.4f}")
            print(f"Beta: {self.result_.params['smoothing_trend']:.4f}")
            if self.damped:
                print(f"Phi: {self.result_.params['damping_trend']:.4f}")

        return self

    def forecast(self, steps: int = 1) -> pd.Series:
        """Forecast future values."""
        if self.result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.result_.forecast(steps=steps)


class HoltWinters:
    """
    Holt-Winters Seasonal Model

    Handles both trend and seasonality.

    Parameters:
    -----------
    seasonal : str, default='additive'
        Type of seasonal component: 'additive' or 'multiplicative'.
    seasonal_periods : int, optional
        Number of periods in a season (e.g., 12 for monthly data).
    trend : str, optional
        Type of trend: 'additive', 'multiplicative', or None.
    damped : bool, default=False
        Whether to use damped trend.

    Example:
    --------
    >>> hw = HoltWinters(seasonal='additive', seasonal_periods=12)
    >>> hw.fit(monthly_data)
    >>> forecast = hw.forecast(steps=24)
    """

    def __init__(
        self,
        seasonal: str = 'additive',
        seasonal_periods: Optional[int] = None,
        trend: Optional[str] = 'additive',
        damped: bool = False
    ):
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required")

        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.damped = damped
        self.model_ = None
        self.result_ = None

    def fit(self, y: Union[pd.Series, np.ndarray], verbose: bool = True):
        """Fit Holt-Winters model."""
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        self.model_ = ExponentialSmoothing(
            y,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            trend=self.trend,
            damped_trend=self.damped
        )

        self.result_ = self.model_.fit(optimized=True)

        if verbose:
            print(f"Holt-Winters model fitted")
            print(f"Trend: {self.trend}")
            print(f"Seasonal: {self.seasonal}")
            print(f"Periods: {self.seasonal_periods}")

        return self

    def forecast(self, steps: int = 1) -> pd.Series:
        """Forecast future values."""
        if self.result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.result_.forecast(steps=steps)


class MovingAverage:
    """
    Moving Average Forecaster

    Simple moving average for smoothing and forecasting.

    Parameters:
    -----------
    window : int
        Size of the moving window.
    center : bool, default=False
        Whether to center the window.

    Example:
    --------
    >>> ma = MovingAverage(window=7)
    >>> ma.fit(data)
    >>> forecast = ma.forecast(steps=5)
    """

    def __init__(self, window: int, center: bool = False):
        self.window = window
        self.center = center
        self.data_ = None
        self.ma_ = None

    def fit(self, y: Union[pd.Series, np.ndarray]):
        """Fit moving average."""
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        self.data_ = y
        self.ma_ = y.rolling(window=self.window, center=self.center).mean()

        return self

    def forecast(self, steps: int = 1) -> np.ndarray:
        """Forecast by repeating last moving average value."""
        if self.ma_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        last_value = self.ma_.dropna().iloc[-1]
        return np.array([last_value] * steps)


class NaiveForecaster:
    """
    Naive Forecasting Methods

    Simple baseline forecasting methods.

    Parameters:
    -----------
    method : str, default='last'
        Forecasting method:
        - 'last': Use last observed value
        - 'mean': Use mean of all values
        - 'seasonal': Use value from same season last year

    Example:
    --------
    >>> naive = NaiveForecaster(method='last')
    >>> naive.fit(data)
    >>> forecast = naive.forecast(steps=10)
    """

    def __init__(self, method: str = 'last', seasonal_period: int = 12):
        self.method = method
        self.seasonal_period = seasonal_period
        self.data_ = None

    def fit(self, y: Union[pd.Series, np.ndarray]):
        """Fit naive forecaster."""
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        self.data_ = y
        return self

    def forecast(self, steps: int = 1) -> np.ndarray:
        """Generate forecast."""
        if self.data_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.method == 'last':
            return np.array([self.data_.iloc[-1]] * steps)

        elif self.method == 'mean':
            return np.array([self.data_.mean()] * steps)

        elif self.method == 'seasonal':
            forecast = []
            for i in range(steps):
                idx = len(self.data_) - self.seasonal_period + (i % self.seasonal_period)
                if idx >= 0:
                    forecast.append(self.data_.iloc[idx])
                else:
                    forecast.append(self.data_.mean())
            return np.array(forecast)

        else:
            raise ValueError(f"Unknown method: {self.method}")


if __name__ == "__main__":
    print("=" * 70)
    print("FORECASTING MODELS EXAMPLES")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')

    # Data with trend
    trend = np.linspace(100, 150, 200)
    noise = np.random.randn(200) * 5
    data_trend = pd.Series(trend + noise, index=dates)

    # Data with seasonality
    t = np.arange(200)
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    data_seasonal = pd.Series(trend + seasonal + noise, index=dates)

    if STATSMODELS_AVAILABLE:
        # Example 1: Simple Exponential Smoothing
        print("\n1. Simple Exponential Smoothing")
        print("-" * 70)
        ses = SimpleExponentialSmoothing()
        ses.fit(data_trend[:150], verbose=True)
        forecast = ses.forecast(steps=50)
        print(f"Forecast (next 50): {forecast[:5].values}...")

        # Example 2: Holt's Linear Trend
        print("\n2. Holt's Linear Trend")
        print("-" * 70)
        holt = HoltLinearTrend(damped=True)
        holt.fit(data_trend[:150], verbose=True)
        forecast = holt.forecast(steps=50)
        print(f"Forecast shape: {forecast.shape}")

        # Example 3: Holt-Winters
        print("\n3. Holt-Winters Seasonal")
        print("-" * 70)
        hw = HoltWinters(seasonal='additive', seasonal_periods=12)
        hw.fit(data_seasonal[:150], verbose=True)
        forecast = hw.forecast(steps=50)
        print(f"Forecast (first 5): {forecast[:5].values}")

    # Example 4: Moving Average
    print("\n4. Moving Average")
    print("-" * 70)
    ma = MovingAverage(window=7)
    ma.fit(data_trend)
    forecast = ma.forecast(steps=10)
    print(f"MA Forecast: {forecast[:5]}")

    # Example 5: Naive Forecaster
    print("\n5. Naive Forecaster")
    print("-" * 70)
    naive = NaiveForecaster(method='last')
    naive.fit(data_trend)
    forecast = naive.forecast(steps=10)
    print(f"Naive Forecast: {forecast[:5]}")

    print("\n" + "=" * 70)
    print("All forecasting examples completed!")
    print("=" * 70)
