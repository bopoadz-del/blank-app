"""
Seasonality Detection and Decomposition

This module provides tools for detecting and decomposing seasonal patterns
in time series data.

Key Features:
- Classical decomposition (additive/multiplicative)
- STL decomposition (Seasonal-Trend decomposition using LOESS)
- Seasonal period detection
- Autocorrelation-based seasonality detection
- Fourier transform for periodicity detection

Author: ML Framework Team
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, Dict, List
import warnings

try:
    from statsmodels.tsa.seasonal import seasonal_decompose, STL
    from statsmodels.tsa.stattools import acf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Install with: pip install statsmodels")

try:
    from scipy import signal, stats
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Install with: pip install scipy")


class SeasonalDecomposition:
    """
    Classical Seasonal Decomposition

    Decomposes a time series into trend, seasonal, and residual components.

    Parameters:
    -----------
    model : str, default='additive'
        Type of seasonal component: 'additive' or 'multiplicative'.
    period : int, optional
        Period of the seasonal component. If None, auto-detected.
    extrapolate_trend : str, default='freq'
        How to handle boundaries when decomposing.

    Attributes:
    -----------
    trend_ : pd.Series
        Trend component.
    seasonal_ : pd.Series
        Seasonal component.
    residual_ : pd.Series
        Residual component.

    Example:
    --------
    >>> decomp = SeasonalDecomposition(model='additive', period=12)
    >>> decomp.fit(monthly_sales)
    >>> trend = decomp.trend_
    >>> seasonal = decomp.seasonal_
    >>> residual = decomp.residual_
    >>> decomp.plot()
    """

    def __init__(
        self,
        model: str = 'additive',
        period: Optional[int] = None,
        extrapolate_trend: str = 'freq'
    ):
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required. Install with: pip install statsmodels")

        self.model = model
        self.period = period
        self.extrapolate_trend = extrapolate_trend
        self.result_ = None
        self.trend_ = None
        self.seasonal_ = None
        self.residual_ = None

    def fit(self, y: Union[pd.Series, np.ndarray], verbose: bool = True):
        """
        Fit seasonal decomposition.

        Parameters:
        -----------
        y : pd.Series or np.ndarray
            Time series data.
        verbose : bool, default=True
            Print decomposition information.

        Returns:
        --------
        self : object
            Fitted decomposer.
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        # Auto-detect period if not provided
        if self.period is None:
            self.period = self._detect_period(y)
            if verbose:
                print(f"Auto-detected period: {self.period}")

        # Perform decomposition
        self.result_ = seasonal_decompose(
            y,
            model=self.model,
            period=self.period,
            extrapolate_trend=self.extrapolate_trend
        )

        self.trend_ = self.result_.trend
        self.seasonal_ = self.result_.seasonal
        self.residual_ = self.result_.resid

        if verbose:
            print(f"Seasonal decomposition completed")
            print(f"Model: {self.model}")
            print(f"Period: {self.period}")

        return self

    def _detect_period(self, y: pd.Series) -> int:
        """Auto-detect seasonal period using ACF."""
        if not STATSMODELS_AVAILABLE:
            return 12  # Default

        acf_vals = acf(y, nlags=min(len(y)//2, 100))

        # Find first significant peak after lag 1
        for lag in range(2, len(acf_vals)):
            if acf_vals[lag] > 0.5:  # Significant autocorrelation
                return lag

        return 12  # Default to 12 if no clear period found

    def plot(self):
        """Plot decomposition components."""
        if self.result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        try:
            import matplotlib.pyplot as plt
            self.result_.plot()
            plt.tight_layout()
            plt.show()
        except ImportError:
            warnings.warn("matplotlib not available. Cannot plot.")


class STLDecomposition:
    """
    STL (Seasonal-Trend decomposition using LOESS)

    More robust and flexible than classical decomposition.

    Parameters:
    -----------
    seasonal : int
        Length of the seasonal smoother. Must be odd.
    period : int, optional
        Periodicity of the sequence. If None, auto-detected.
    trend : int, optional
        Length of the trend smoother. Must be odd.
    robust : bool, default=False
        Flag indicating whether to use robust version.

    Example:
    --------
    >>> stl = STLDecomposition(period=12, seasonal=13)
    >>> stl.fit(data)
    >>> trend = stl.trend_
    >>> seasonal = stl.seasonal_
    """

    def __init__(
        self,
        seasonal: int = 7,
        period: Optional[int] = None,
        trend: Optional[int] = None,
        robust: bool = False
    ):
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required")

        self.seasonal = seasonal
        self.period = period
        self.trend = trend
        self.robust = robust
        self.result_ = None
        self.trend_ = None
        self.seasonal_ = None
        self.residual_ = None

    def fit(self, y: Union[pd.Series, np.ndarray], verbose: bool = True):
        """Fit STL decomposition."""
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        # Auto-detect period if needed
        if self.period is None:
            detector = SeasonalDecomposition()
            self.period = detector._detect_period(y)
            if verbose:
                print(f"Auto-detected period: {self.period}")

        # Fit STL
        stl = STL(
            y,
            seasonal=self.seasonal,
            period=self.period,
            trend=self.trend,
            robust=self.robust
        )

        self.result_ = stl.fit()
        self.trend_ = self.result_.trend
        self.seasonal_ = self.result_.seasonal
        self.residual_ = self.result_.resid

        if verbose:
            print(f"STL decomposition completed")
            print(f"Period: {self.period}")
            print(f"Robust: {self.robust}")

        return self

    def plot(self):
        """Plot STL decomposition."""
        if self.result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        try:
            import matplotlib.pyplot as plt
            self.result_.plot()
            plt.tight_layout()
            plt.show()
        except ImportError:
            warnings.warn("matplotlib not available.")


class SeasonalityDetector:
    """
    Detect Seasonality in Time Series

    Uses multiple methods to detect seasonal patterns:
    - Autocorrelation (ACF)
    - Fourier Transform
    - Statistical tests

    Example:
    --------
    >>> detector = SeasonalityDetector()
    >>> result = detector.detect(data)
    >>> if result['has_seasonality']:
    ...     print(f"Detected period: {result['period']}")
    """

    def detect(self, y: Union[pd.Series, np.ndarray], max_period: int = 50) -> Dict:
        """
        Detect seasonality in time series.

        Parameters:
        -----------
        y : pd.Series or np.ndarray
            Time series data.
        max_period : int, default=50
            Maximum period to check.

        Returns:
        --------
        result : dict
            Dictionary with detection results:
            - has_seasonality: bool
            - period: int or None
            - strength: float
            - method: str
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        results = {}

        # Method 1: ACF-based detection
        acf_result = self._detect_acf(y, max_period)
        results['acf'] = acf_result

        # Method 2: FFT-based detection
        fft_result = self._detect_fft(y, max_period)
        results['fft'] = fft_result

        # Combine results
        has_seasonality = acf_result['has_seasonality'] or fft_result['has_seasonality']

        period = None
        if acf_result['has_seasonality'] and fft_result['has_seasonality']:
            # Both methods agree
            if acf_result['period'] == fft_result['period']:
                period = acf_result['period']
            else:
                # Use the one with higher strength
                period = acf_result['period'] if acf_result['strength'] > fft_result['strength'] else fft_result['period']
        elif acf_result['has_seasonality']:
            period = acf_result['period']
        elif fft_result['has_seasonality']:
            period = fft_result['period']

        return {
            'has_seasonality': has_seasonality,
            'period': period,
            'strength': max(acf_result.get('strength', 0), fft_result.get('strength', 0)),
            'details': results
        }

    def _detect_acf(self, y: pd.Series, max_period: int) -> Dict:
        """Detect seasonality using autocorrelation."""
        if not STATSMODELS_AVAILABLE:
            return {'has_seasonality': False}

        nlags = min(len(y) // 2, max_period * 2)
        acf_vals = acf(y, nlags=nlags)

        # Find peaks in ACF
        peaks = []
        for i in range(2, len(acf_vals)):
            if acf_vals[i] > acf_vals[i-1] and acf_vals[i] > acf_vals[i+1] if i < len(acf_vals)-1 else False:
                if acf_vals[i] > 0.3:  # Significant correlation
                    peaks.append((i, acf_vals[i]))

        if peaks:
            # Use first significant peak as period
            period, strength = peaks[0]
            return {
                'has_seasonality': True,
                'period': period,
                'strength': strength
            }

        return {'has_seasonality': False}

    def _detect_fft(self, y: pd.Series, max_period: int) -> Dict:
        """Detect seasonality using Fourier Transform."""
        if not SCIPY_AVAILABLE:
            return {'has_seasonality': False}

        # Compute FFT
        n = len(y)
        yf = fft(y.values - y.mean())
        xf = fftfreq(n, 1)[:n//2]
        power = 2.0/n * np.abs(yf[0:n//2])

        # Find dominant frequency (excluding DC component)
        dominant_idx = np.argmax(power[1:]) + 1
        dominant_freq = xf[dominant_idx]

        if dominant_freq > 0:
            period = int(np.round(1.0 / dominant_freq))

            # Check if period is reasonable
            if 2 <= period <= max_period:
                strength = power[dominant_idx] / np.max(power)

                if strength > 0.1:  # Significant peak
                    return {
                        'has_seasonality': True,
                        'period': period,
                        'strength': strength
                    }

        return {'has_seasonality': False}


def detrend(y: Union[pd.Series, np.ndarray], method: str = 'linear') -> np.ndarray:
    """
    Remove trend from time series.

    Parameters:
    -----------
    y : pd.Series or np.ndarray
        Time series data.
    method : str, default='linear'
        Detrending method: 'linear', 'constant', or 'diff'.

    Returns:
    --------
    detrended : np.ndarray
        Detrended time series.
    """
    if isinstance(y, pd.Series):
        y = y.values

    if method == 'linear':
        if SCIPY_AVAILABLE:
            return signal.detrend(y, type='linear')
        else:
            # Manual linear detrend
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            trend = slope * x + intercept
            return y - trend

    elif method == 'constant':
        return y - np.mean(y)

    elif method == 'diff':
        return np.diff(y, prepend=y[0])

    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    print("=" * 70)
    print("SEASONALITY DETECTION EXAMPLES")
    print("=" * 70)

    # Generate sample seasonal data
    np.random.seed(42)
    t = np.arange(200)
    trend = 0.5 * t + 100
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise = np.random.randn(200) * 2
    data = trend + seasonal + noise

    ts = pd.Series(data)

    # Example 1: Seasonality Detection
    print("\n1. Seasonality Detection")
    print("-" * 70)

    detector = SeasonalityDetector()
    result = detector.detect(ts)

    print(f"Has seasonality: {result['has_seasonality']}")
    print(f"Detected period: {result['period']}")
    print(f"Strength: {result['strength']:.4f}")

    # Example 2: Classical Decomposition
    if STATSMODELS_AVAILABLE:
        print("\n2. Classical Seasonal Decomposition")
        print("-" * 70)

        decomp = SeasonalDecomposition(model='additive', period=12)
        decomp.fit(ts, verbose=True)

        print(f"Trend (first 5): {decomp.trend_.dropna().head().values}")
        print(f"Seasonal (first 5): {decomp.seasonal_.head().values}")

        # Example 3: STL Decomposition
        print("\n3. STL Decomposition")
        print("-" * 70)

        stl = STLDecomposition(period=12, seasonal=13, robust=True)
        stl.fit(ts, verbose=True)

        print(f"Trend (first 5): {stl.trend_.head().values}")

    # Example 4: Detrending
    print("\n4. Detrending")
    print("-" * 70)

    detrended = detrend(ts, method='linear')
    print(f"Original mean: {ts.mean():.2f}")
    print(f"Detrended mean: {detrended.mean():.2f}")
    print(f"Detrended std: {detrended.std():.2f}")

    print("\n" + "=" * 70)
    print("All seasonality examples completed!")
    print("=" * 70)
