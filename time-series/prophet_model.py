"""
Facebook Prophet Time Series Model

This module provides a wrapper for Facebook Prophet, a forecasting tool designed
for business time series with strong seasonal patterns and several seasons of
historical data.

Key Features:
- Automatic seasonality detection (daily, weekly, yearly)
- Holiday effects
- Trend changepoints
- Multiple seasonalities
- Uncertainty intervals
- Outlier handling

Author: ML Framework Team
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any
import warnings

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    from prophet.plot import plot_cross_validation_metric
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Install with: pip install prophet")


class ProphetModel:
    """
    Facebook Prophet Time Series Forecasting Model

    Prophet is designed for forecasting time series data based on an additive model
    where non-linear trends are fit with yearly, weekly, and daily seasonality,
    plus holiday effects.

    Components:
    - Trend: Piecewise linear or logistic growth
    - Seasonality: Fourier series for flexible seasonal patterns
    - Holidays: User-provided list of holidays

    Parameters:
    -----------
    growth : str, default='linear'
        'linear' or 'logistic' to specify linear or logistic growth.
    changepoints : list, optional
        List of dates at which to include potential changepoints.
    n_changepoints : int, default=25
        Number of potential changepoints to include.
    changepoint_range : float, default=0.8
        Proportion of history in which trend changepoints will be estimated.
    yearly_seasonality : bool or int, default='auto'
        Fit yearly seasonality. Can be 'auto', True, False, or an integer Fourier order.
    weekly_seasonality : bool or int, default='auto'
        Fit weekly seasonality.
    daily_seasonality : bool or int, default='auto'
        Fit daily seasonality.
    seasonality_mode : str, default='additive'
        'additive' or 'multiplicative'.
    seasonality_prior_scale : float, default=10.0
        Parameter modulating the strength of the seasonality model.
    changepoint_prior_scale : float, default=0.05
        Parameter modulating the flexibility of the automatic changepoint selection.
    holidays_prior_scale : float, default=10.0
        Parameter modulating the strength of the holiday components model.
    interval_width : float, default=0.80
        Width of the uncertainty intervals (e.g., 0.80 for 80% intervals).

    Attributes:
    -----------
    model_ : Prophet
        Fitted Prophet model.
    forecast_ : pd.DataFrame
        Last forecast generated.

    Example:
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Create sample data
    >>> dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
    >>> values = np.cumsum(np.random.randn(365)) + 100 + 10 * np.sin(np.arange(365) * 2 * np.pi / 365)
    >>> df = pd.DataFrame({'ds': dates, 'y': values})
    >>>
    >>> # Fit Prophet model
    >>> prophet = ProphetModel(
    ...     yearly_seasonality=True,
    ...     weekly_seasonality=True,
    ...     daily_seasonality=False
    ... )
    >>> prophet.fit(df)
    >>>
    >>> # Forecast
    >>> future = prophet.make_future_dataframe(periods=90)
    >>> forecast = prophet.forecast(future)
    >>>
    >>> # Plot
    >>> prophet.plot(forecast)
    >>> prophet.plot_components(forecast)
    """

    def __init__(
        self,
        growth: str = 'linear',
        changepoints: Optional[List] = None,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        yearly_seasonality: Union[str, bool, int] = 'auto',
        weekly_seasonality: Union[str, bool, int] = 'auto',
        daily_seasonality: Union[str, bool, int] = 'auto',
        seasonality_mode: str = 'additive',
        seasonality_prior_scale: float = 10.0,
        changepoint_prior_scale: float = 0.05,
        holidays_prior_scale: float = 10.0,
        interval_width: float = 0.80
    ):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Install with: pip install prophet")

        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.interval_width = interval_width
        self.model_ = None
        self.forecast_ = None

    def fit(
        self,
        df: pd.DataFrame,
        verbose: bool = True
    ):
        """
        Fit Prophet model to historical data.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with columns 'ds' (datetime) and 'y' (value).
            Can also include 'cap' (for logistic growth) and additional regressors.
        verbose : bool, default=True
            Whether to print fitting progress.

        Returns:
        --------
        self : object
            Fitted model.
        """
        # Validate input
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must have 'ds' (datetime) and 'y' (value) columns")

        # Create Prophet model
        self.model_ = Prophet(
            growth=self.growth,
            changepoints=self.changepoints,
            n_changepoints=self.n_changepoints,
            changepoint_range=self.changepoint_range,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            seasonality_prior_scale=self.seasonality_prior_scale,
            changepoint_prior_scale=self.changepoint_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            interval_width=self.interval_width
        )

        # Fit model
        if not verbose:
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)

        self.model_.fit(df)

        if verbose:
            print("Prophet model fitted successfully!")

        return self

    def make_future_dataframe(
        self,
        periods: int,
        freq: str = 'D',
        include_history: bool = True
    ) -> pd.DataFrame:
        """
        Create a dataframe for future predictions.

        Parameters:
        -----------
        periods : int
            Number of periods to forecast forward.
        freq : str, default='D'
            Frequency of predictions ('D' for daily, 'M' for monthly, etc.).
        include_history : bool, default=True
            Whether to include historical dates.

        Returns:
        --------
        future : pd.DataFrame
            DataFrame with 'ds' column for forecasting.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model_.make_future_dataframe(
            periods=periods,
            freq=freq,
            include_history=include_history
        )

    def forecast(self, future: pd.DataFrame) -> pd.DataFrame:
        """
        Generate forecast for future dates.

        Parameters:
        -----------
        future : pd.DataFrame
            DataFrame with 'ds' column containing dates to forecast.

        Returns:
        --------
        forecast : pd.DataFrame
            DataFrame with forecast including yhat (prediction),
            yhat_lower, yhat_upper (uncertainty intervals), and components.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        self.forecast_ = self.model_.predict(future)
        return self.forecast_

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alias for forecast() method.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'ds' column.

        Returns:
        --------
        predictions : pd.DataFrame
            Forecast dataframe.
        """
        return self.forecast(df)

    def plot(self, forecast: Optional[pd.DataFrame] = None, **kwargs):
        """
        Plot the forecast.

        Parameters:
        -----------
        forecast : pd.DataFrame, optional
            Forecast dataframe. If None, uses last forecast.
        **kwargs : dict
            Additional arguments passed to Prophet's plot method.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if forecast is None:
            if self.forecast_ is None:
                raise ValueError("No forecast available. Call forecast() first.")
            forecast = self.forecast_

        try:
            import matplotlib.pyplot as plt
            fig = self.model_.plot(forecast, **kwargs)
            plt.show()
            return fig
        except ImportError:
            warnings.warn("matplotlib not available. Cannot plot forecast.")

    def plot_components(self, forecast: Optional[pd.DataFrame] = None, **kwargs):
        """
        Plot forecast components (trend, seasonality).

        Parameters:
        -----------
        forecast : pd.DataFrame, optional
            Forecast dataframe. If None, uses last forecast.
        **kwargs : dict
            Additional arguments passed to Prophet's plot_components method.
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if forecast is None:
            if self.forecast_ is None:
                raise ValueError("No forecast available. Call forecast() first.")
            forecast = self.forecast_

        try:
            import matplotlib.pyplot as plt
            fig = self.model_.plot_components(forecast, **kwargs)
            plt.show()
            return fig
        except ImportError:
            warnings.warn("matplotlib not available. Cannot plot components.")

    def add_seasonality(
        self,
        name: str,
        period: float,
        fourier_order: int,
        prior_scale: Optional[float] = None,
        mode: Optional[str] = None
    ):
        """
        Add a custom seasonality component.

        Parameters:
        -----------
        name : str
            Name of the seasonality component.
        period : float
            Period of the seasonality in days.
        fourier_order : int
            Number of Fourier components to use.
        prior_scale : float, optional
            Prior scale for this component.
        mode : str, optional
            'additive' or 'multiplicative'.

        Example:
        --------
        >>> # Add monthly seasonality
        >>> prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        """
        if self.model_ is None:
            # Create model if not exists
            self.model_ = Prophet()

        self.model_.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order,
            prior_scale=prior_scale,
            mode=mode
        )

        return self

    def add_regressor(
        self,
        name: str,
        prior_scale: Optional[float] = None,
        standardize: str = 'auto',
        mode: Optional[str] = None
    ):
        """
        Add an additional regressor to the model.

        Parameters:
        -----------
        name : str
            Name of the regressor column in the dataframe.
        prior_scale : float, optional
            Prior scale for this regressor.
        standardize : str, default='auto'
            Whether to standardize the regressor ('auto', True, False).
        mode : str, optional
            'additive' or 'multiplicative'.

        Example:
        --------
        >>> # Add external regressor (e.g., marketing spend)
        >>> prophet.add_regressor('marketing_spend')
        >>> prophet.fit(df)  # df must include 'marketing_spend' column
        """
        if self.model_ is None:
            self.model_ = Prophet()

        self.model_.add_regressor(
            name=name,
            prior_scale=prior_scale,
            standardize=standardize,
            mode=mode
        )

        return self

    def add_country_holidays(self, country_name: str):
        """
        Add country-specific holidays to the model.

        Parameters:
        -----------
        country_name : str
            Country name (e.g., 'US', 'UK', 'DE').

        Example:
        --------
        >>> prophet.add_country_holidays('US')
        """
        if self.model_ is None:
            self.model_ = Prophet()

        self.model_.add_country_holidays(country_name=country_name)

        return self

    def cross_validate(
        self,
        initial: str,
        period: str,
        horizon: str,
        parallel: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Perform cross-validation on the model.

        Parameters:
        -----------
        initial : str
            Size of the initial training period (e.g., '730 days').
        period : str
            Spacing between cutoff dates (e.g., '180 days').
        horizon : str
            Forecast horizon (e.g., '365 days').
        parallel : str, optional
            'processes' or 'threads' for parallel execution.

        Returns:
        --------
        cv_results : pd.DataFrame
            Cross-validation results.

        Example:
        --------
        >>> cv_results = prophet.cross_validate(
        ...     initial='730 days',
        ...     period='180 days',
        ...     horizon='365 days'
        ... )
        >>> metrics = prophet.get_performance_metrics(cv_results)
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return cross_validation(
            self.model_,
            initial=initial,
            period=period,
            horizon=horizon,
            parallel=parallel
        )

    def get_performance_metrics(
        self,
        cv_results: pd.DataFrame,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute performance metrics from cross-validation results.

        Parameters:
        -----------
        cv_results : pd.DataFrame
            Results from cross_validate().
        metrics : list, optional
            List of metrics to compute. Options: 'mse', 'rmse', 'mae', 'mape', 'coverage'.

        Returns:
        --------
        metrics_df : pd.DataFrame
            Performance metrics.
        """
        if metrics is None:
            metrics = ['mse', 'rmse', 'mae', 'mape', 'coverage']

        return performance_metrics(cv_results, metrics=metrics)


if __name__ == "__main__":
    if not PROPHET_AVAILABLE:
        print("Prophet not available. Install with: pip install prophet")
    else:
        print("=" * 70)
        print("PROPHET MODEL EXAMPLES")
        print("=" * 70)

        # Generate sample data with trend and seasonality
        np.random.seed(42)
        dates = pd.date_range(start='2019-01-01', periods=730, freq='D')

        # Create components
        trend = np.linspace(100, 200, 730)
        yearly_seasonality = 20 * np.sin(2 * np.pi * np.arange(730) / 365.25)
        weekly_seasonality = 5 * np.sin(2 * np.pi * np.arange(730) / 7)
        noise = np.random.randn(730) * 5

        values = trend + yearly_seasonality + weekly_seasonality + noise

        # Create dataframe in Prophet format
        df = pd.DataFrame({
            'ds': dates,
            'y': values
        })

        # Example 1: Basic Prophet Model
        print("\n1. Basic Prophet Model")
        print("-" * 70)

        prophet = ProphetModel(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        prophet.fit(df, verbose=True)

        # Forecast 90 days into future
        future = prophet.make_future_dataframe(periods=90)
        forecast = prophet.forecast(future)

        print(f"\nForecast columns: {forecast.columns.tolist()}")
        print(f"\nForecast for next 5 days:")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # Example 2: Prophet with Custom Seasonality
        print("\n2. Prophet with Custom Seasonality")
        print("-" * 70)

        prophet_custom = ProphetModel()
        prophet_custom.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        prophet_custom.fit(df, verbose=True)

        future = prophet_custom.make_future_dataframe(periods=90)
        forecast = prophet_custom.forecast(future)
        print(f"Forecast shape: {forecast.shape}")

        # Example 3: Prophet with Holidays
        print("\n3. Prophet with US Holidays")
        print("-" * 70)

        prophet_holidays = ProphetModel()
        prophet_holidays.add_country_holidays('US')
        prophet_holidays.fit(df, verbose=True)

        future = prophet_holidays.make_future_dataframe(periods=90)
        forecast = prophet_holidays.forecast(future)
        print("Model fitted with US holidays")

        print("\n" + "=" * 70)
        print("All Prophet examples completed!")
        print("=" * 70)
