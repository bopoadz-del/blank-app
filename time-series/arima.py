"""
ARIMA and SARIMA Time Series Models

This module implements ARIMA (AutoRegressive Integrated Moving Average) and
SARIMA (Seasonal ARIMA) models for time series forecasting.

Key Features:
- ARIMA: Non-seasonal time series modeling
- SARIMA: Seasonal time series modeling
- Auto ARIMA: Automatic parameter selection
- Stationarity testing
- Diagnostics and residual analysis
- Forecast with confidence intervals

Author: ML Framework Team
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, Dict, Any
import warnings

# Statistical tests
from scipy import stats

# ARIMA/SARIMA models
try:
    from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Install with: pip install statsmodels")

try:
    from pmdarima import auto_arima as pm_auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    warnings.warn("pmdarima not available. Install with: pip install pmdarima")


class ARIMAModel:
    """
    ARIMA (AutoRegressive Integrated Moving Average) Model

    ARIMA(p,d,q) combines three components:
    - AR(p): AutoRegressive component (p past values)
    - I(d): Integrated component (d times differencing to make stationary)
    - MA(q): Moving Average component (q past forecast errors)

    Parameters:
    -----------
    order : tuple (p, d, q)
        The (p,d,q) order of the model for the number of AR parameters,
        differences, and MA parameters.
    seasonal_order : tuple (P, D, Q, s), default=None
        The (P,D,Q,s) order of the seasonal component. If None, non-seasonal model.
    trend : str, default='c'
        Parameter controlling the deterministic trend polynomial.
        - 'n': No trend
        - 'c': Constant (default)
        - 't': Linear trend
        - 'ct': Constant and linear trend
    enforce_stationarity : bool, default=True
        Whether to ensure the AR parameters remain in the stationary region.
    enforce_invertibility : bool, default=True
        Whether to ensure the MA parameters remain in the invertible region.

    Attributes:
    -----------
    model_ : ARIMA or SARIMAX
        Fitted statsmodels ARIMA/SARIMAX model.
    aic_ : float
        Akaike Information Criterion.
    bic_ : float
        Bayesian Information Criterion.
    fitted_values_ : array
        In-sample fitted values.
    residuals_ : array
        Residuals of the fitted model.

    Example:
    --------
    >>> import numpy as np
    >>> from datetime import datetime, timedelta
    >>>
    >>> # Generate sample time series
    >>> dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    >>> values = np.cumsum(np.random.randn(100)) + 100
    >>> ts = pd.Series(values, index=dates)
    >>>
    >>> # Fit ARIMA model
    >>> arima = ARIMAModel(order=(1, 1, 1))
    >>> arima.fit(ts)
    >>>
    >>> # Forecast
    >>> forecast = arima.forecast(steps=10)
    >>> print(forecast)
    >>>
    >>> # Get confidence intervals
    >>> forecast, conf_int = arima.forecast(steps=10, return_conf_int=True)
    >>> print(f"Forecast: {forecast}")
    >>> print(f"95% CI: {conf_int}")
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        trend: str = 'c',
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True
    ):
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required. Install with: pip install statsmodels")

        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model_ = None
        self.result_ = None
        self.aic_ = None
        self.bic_ = None
        self.fitted_values_ = None
        self.residuals_ = None

    def fit(self, y: Union[pd.Series, np.ndarray], verbose: bool = True):
        """
        Fit ARIMA model to time series data.

        Parameters:
        -----------
        y : pd.Series or np.ndarray
            Time series data to fit.
        verbose : bool, default=True
            Whether to print fitting information.

        Returns:
        --------
        self : object
            Fitted model.
        """
        # Convert to pandas Series if needed
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        # Determine if seasonal
        if self.seasonal_order is not None:
            # Use SARIMAX
            self.model_ = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )
        else:
            # Use ARIMA
            self.model_ = SM_ARIMA(
                y,
                order=self.order,
                trend=self.trend,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility
            )

        # Fit model
        self.result_ = self.model_.fit(disp=verbose)

        # Store metrics
        self.aic_ = self.result_.aic
        self.bic_ = self.result_.bic
        self.fitted_values_ = self.result_.fittedvalues
        self.residuals_ = self.result_.resid

        if verbose:
            print(f"\nModel fitted successfully!")
            print(f"AIC: {self.aic_:.2f}")
            print(f"BIC: {self.bic_:.2f}")

        return self

    def forecast(
        self,
        steps: int = 1,
        return_conf_int: bool = False,
        alpha: float = 0.05
    ) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
        """
        Generate forecasts for future time periods.

        Parameters:
        -----------
        steps : int, default=1
            Number of steps to forecast ahead.
        return_conf_int : bool, default=False
            Whether to return confidence intervals.
        alpha : float, default=0.05
            Significance level for confidence intervals (default 95%).

        Returns:
        --------
        forecast : pd.Series
            Forecasted values.
        conf_int : pd.DataFrame (optional)
            Confidence intervals if return_conf_int=True.
        """
        if self.result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get forecast
        forecast_result = self.result_.get_forecast(steps=steps, alpha=alpha)
        forecast = forecast_result.predicted_mean

        if return_conf_int:
            conf_int = forecast_result.conf_int()
            return forecast, conf_int
        else:
            return forecast

    def predict(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        dynamic: bool = False
    ) -> pd.Series:
        """
        Generate in-sample predictions or out-of-sample forecasts.

        Parameters:
        -----------
        start : int, optional
            Zero-indexed observation number at which to start forecasting.
        end : int, optional
            Zero-indexed observation number at which to end forecasting.
        dynamic : bool, default=False
            If True, use dynamic prediction (forecasted values for lagged values).

        Returns:
        --------
        predictions : pd.Series
            Predicted values.
        """
        if self.result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.result_.predict(start=start, end=end, dynamic=dynamic)

    def get_residuals(self) -> pd.Series:
        """Get model residuals."""
        if self.result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.residuals_

    def summary(self) -> str:
        """Get model summary."""
        if self.result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.result_.summary()

    def diagnostics(self):
        """
        Plot diagnostic plots for residual analysis.

        Includes:
        - Standardized residuals
        - Histogram plus KDE
        - Q-Q plot
        - Correlogram (ACF)
        """
        if self.result_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        try:
            import matplotlib.pyplot as plt
            self.result_.plot_diagnostics(figsize=(12, 8))
            plt.tight_layout()
            plt.show()
        except ImportError:
            warnings.warn("matplotlib not available. Cannot plot diagnostics.")


class SARIMAModel(ARIMAModel):
    """
    SARIMA (Seasonal ARIMA) Model

    Convenience class for SARIMA models. Inherits from ARIMAModel.

    Parameters:
    -----------
    order : tuple (p, d, q)
        Non-seasonal (p,d,q) order.
    seasonal_order : tuple (P, D, Q, s)
        Seasonal (P,D,Q,s) order where s is the seasonal period.

    Example:
    --------
    >>> # Monthly data with yearly seasonality
    >>> sarima = SARIMAModel(
    ...     order=(1, 1, 1),
    ...     seasonal_order=(1, 1, 1, 12)  # s=12 for monthly data
    ... )
    >>> sarima.fit(monthly_sales)
    >>> forecast = sarima.forecast(steps=12)  # Forecast next year
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
        trend: str = 'c',
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True
    ):
        super().__init__(
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility
        )


class AutoARIMA:
    """
    Auto ARIMA - Automatic ARIMA parameter selection

    Automatically selects the best ARIMA(p,d,q) or SARIMA(p,d,q)(P,D,Q,s)
    parameters based on AIC, BIC, or other criteria.

    Parameters:
    -----------
    seasonal : bool, default=False
        Whether to fit a seasonal ARIMA model.
    m : int, default=1
        Seasonal period (e.g., 12 for monthly data with yearly seasonality).
    max_p : int, default=5
        Maximum value of p.
    max_q : int, default=5
        Maximum value of q.
    max_d : int, default=2
        Maximum value of d.
    max_P : int, default=2
        Maximum value of P (seasonal).
    max_Q : int, default=2
        Maximum value of Q (seasonal).
    max_D : int, default=1
        Maximum value of D (seasonal).
    information_criterion : str, default='aic'
        Information criterion to use for model selection ('aic', 'bic', 'hqic').
    trace : bool, default=True
        Whether to print model selection progress.
    stepwise : bool, default=True
        Whether to use stepwise algorithm (faster).

    Example:
    --------
    >>> # Automatic parameter selection
    >>> auto = AutoARIMA(seasonal=True, m=12, trace=True)
    >>> auto.fit(monthly_sales)
    >>> print(f"Best order: {auto.order_}")
    >>> print(f"Best seasonal order: {auto.seasonal_order_}")
    >>>
    >>> # Forecast
    >>> forecast = auto.forecast(steps=12)
    """

    def __init__(
        self,
        seasonal: bool = False,
        m: int = 1,
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 2,
        max_P: int = 2,
        max_Q: int = 2,
        max_D: int = 1,
        information_criterion: str = 'aic',
        trace: bool = True,
        stepwise: bool = True
    ):
        if not PMDARIMA_AVAILABLE:
            raise ImportError("pmdarima is required. Install with: pip install pmdarima")

        self.seasonal = seasonal
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.max_P = max_P
        self.max_Q = max_Q
        self.max_D = max_D
        self.information_criterion = information_criterion
        self.trace = trace
        self.stepwise = stepwise
        self.model_ = None
        self.order_ = None
        self.seasonal_order_ = None
        self.aic_ = None
        self.bic_ = None

    def fit(self, y: Union[pd.Series, np.ndarray]):
        """
        Fit Auto ARIMA model - automatically select best parameters.

        Parameters:
        -----------
        y : pd.Series or np.ndarray
            Time series data.

        Returns:
        --------
        self : object
            Fitted model.
        """
        # Fit auto ARIMA
        self.model_ = pm_auto_arima(
            y,
            seasonal=self.seasonal,
            m=self.m,
            max_p=self.max_p,
            max_q=self.max_q,
            max_d=self.max_d,
            max_P=self.max_P,
            max_Q=self.max_Q,
            max_D=self.max_D,
            information_criterion=self.information_criterion,
            trace=self.trace,
            stepwise=self.stepwise,
            suppress_warnings=True
        )

        # Store selected parameters
        self.order_ = self.model_.order
        self.seasonal_order_ = self.model_.seasonal_order
        self.aic_ = self.model_.aic()
        self.bic_ = self.model_.bic()

        if self.trace:
            print(f"\nBest model: ARIMA{self.order_}")
            if self.seasonal:
                print(f"Seasonal order: {self.seasonal_order_}")
            print(f"AIC: {self.aic_:.2f}")
            print(f"BIC: {self.bic_:.2f}")

        return self

    def forecast(
        self,
        steps: int = 1,
        return_conf_int: bool = False,
        alpha: float = 0.05
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate forecasts.

        Parameters:
        -----------
        steps : int, default=1
            Number of steps to forecast.
        return_conf_int : bool, default=False
            Whether to return confidence intervals.
        alpha : float, default=0.05
            Significance level for confidence intervals.

        Returns:
        --------
        forecast : np.ndarray
            Forecasted values.
        conf_int : np.ndarray (optional)
            Confidence intervals if return_conf_int=True.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model_.predict(n_periods=steps, return_conf_int=return_conf_int, alpha=alpha)

    def summary(self):
        """Get model summary."""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.summary()


def check_stationarity(
    series: Union[pd.Series, np.ndarray],
    test: str = 'adf',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test time series for stationarity.

    Parameters:
    -----------
    series : pd.Series or np.ndarray
        Time series data to test.
    test : str, default='adf'
        Statistical test to use:
        - 'adf': Augmented Dickey-Fuller test
        - 'kpss': Kwiatkowski-Phillips-Schmidt-Shin test
    verbose : bool, default=True
        Whether to print results.

    Returns:
    --------
    result : dict
        Dictionary containing test statistic, p-value, and conclusion.

    Example:
    --------
    >>> result = check_stationarity(sales_data, test='adf')
    >>> if result['stationary']:
    ...     print("Series is stationary")
    ... else:
    ...     print("Series is non-stationary, differencing required")
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required")

    result = {}

    if test == 'adf':
        # Augmented Dickey-Fuller test
        # H0: Series has a unit root (non-stationary)
        # H1: Series is stationary
        adf_result = adfuller(series, autolag='AIC')
        result['test'] = 'ADF'
        result['statistic'] = adf_result[0]
        result['p_value'] = adf_result[1]
        result['critical_values'] = adf_result[4]
        result['stationary'] = adf_result[1] < 0.05

        if verbose:
            print(f"Augmented Dickey-Fuller Test")
            print(f"{'='*50}")
            print(f"Test Statistic: {adf_result[0]:.4f}")
            print(f"P-value: {adf_result[1]:.4f}")
            print(f"Critical Values:")
            for key, value in adf_result[4].items():
                print(f"  {key}: {value:.4f}")
            print(f"\nConclusion: {'Stationary' if result['stationary'] else 'Non-stationary'}")

    elif test == 'kpss':
        # KPSS test
        # H0: Series is stationary
        # H1: Series has a unit root (non-stationary)
        kpss_result = kpss(series, regression='c', nlags='auto')
        result['test'] = 'KPSS'
        result['statistic'] = kpss_result[0]
        result['p_value'] = kpss_result[1]
        result['critical_values'] = kpss_result[3]
        result['stationary'] = kpss_result[1] > 0.05  # Note: reversed for KPSS

        if verbose:
            print(f"KPSS Test")
            print(f"{'='*50}")
            print(f"Test Statistic: {kpss_result[0]:.4f}")
            print(f"P-value: {kpss_result[1]:.4f}")
            print(f"Critical Values:")
            for key, value in kpss_result[3].items():
                print(f"  {key}: {value:.4f}")
            print(f"\nConclusion: {'Stationary' if result['stationary'] else 'Non-stationary'}")

    return result


def difference_series(
    series: Union[pd.Series, np.ndarray],
    order: int = 1,
    seasonal: bool = False,
    seasonal_period: int = 12
) -> pd.Series:
    """
    Difference a time series to make it stationary.

    Parameters:
    -----------
    series : pd.Series or np.ndarray
        Time series to difference.
    order : int, default=1
        Order of differencing.
    seasonal : bool, default=False
        Whether to apply seasonal differencing.
    seasonal_period : int, default=12
        Period for seasonal differencing.

    Returns:
    --------
    differenced : pd.Series
        Differenced time series.

    Example:
    --------
    >>> # First-order differencing
    >>> diff1 = difference_series(sales, order=1)
    >>>
    >>> # Seasonal differencing
    >>> seasonal_diff = difference_series(sales, seasonal=True, seasonal_period=12)
    """
    if isinstance(series, np.ndarray):
        series = pd.Series(series)

    differenced = series.copy()

    # Regular differencing
    for i in range(order):
        differenced = differenced.diff().dropna()

    # Seasonal differencing
    if seasonal:
        differenced = differenced.diff(seasonal_period).dropna()

    return differenced


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("ARIMA/SARIMA EXAMPLES")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')

    # Non-seasonal data with trend
    trend = np.linspace(100, 150, 200)
    noise = np.random.randn(200) * 5
    values = trend + noise + np.cumsum(np.random.randn(200) * 2)
    ts = pd.Series(values, index=dates)

    # Example 1: ARIMA Model
    print("\n1. ARIMA Model")
    print("-" * 70)

    arima = ARIMAModel(order=(1, 1, 1))
    arima.fit(ts, verbose=True)

    # Forecast
    forecast = arima.forecast(steps=30)
    print(f"\nForecast (next 30 days):")
    print(forecast.head())

    # Example 2: Check Stationarity
    print("\n2. Stationarity Test")
    print("-" * 70)
    result = check_stationarity(ts, test='adf', verbose=True)

    # Example 3: Seasonal data and SARIMA
    print("\n3. SARIMA Model (Seasonal Data)")
    print("-" * 70)

    # Generate seasonal data
    t = np.arange(200)
    seasonal = 10 * np.sin(2 * np.pi * t / 12)  # 12-period seasonality
    seasonal_ts = pd.Series(trend + seasonal + noise, index=dates)

    if STATSMODELS_AVAILABLE:
        sarima = SARIMAModel(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12)
        )
        sarima.fit(seasonal_ts, verbose=True)
        forecast = sarima.forecast(steps=12)
        print(f"\nSeasonal forecast (next 12 periods):")
        print(forecast)

    # Example 4: Auto ARIMA
    if PMDARIMA_AVAILABLE:
        print("\n4. Auto ARIMA - Automatic Parameter Selection")
        print("-" * 70)

        auto = AutoARIMA(seasonal=False, max_p=3, max_q=3, trace=True)
        auto.fit(ts[:150])  # Fit on first 150 points

        print(f"\nSelected order: {auto.order_}")
        forecast = auto.forecast(steps=50)
        print(f"Forecast shape: {forecast.shape}")

    print("\n" + "=" * 70)
    print("All ARIMA/SARIMA examples completed!")
    print("=" * 70)
