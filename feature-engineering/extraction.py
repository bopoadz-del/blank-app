"""
Feature Extraction
Extract features from various data types: text, images, time series, and structured data
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Any
import logging
from datetime import datetime
from collections import Counter

try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextFeatureExtractor:
    """
    Extract features from text data

    Features:
    - Text statistics (length, word count, etc.)
    - N-gram features
    - Character-level features
    - Readability metrics
    - Sentiment features (with TextBlob)
    """

    def __init__(self, include_ngrams: bool = True, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize text feature extractor

        Args:
            include_ngrams: Extract n-gram features
            ngram_range: Range of n-grams to extract
        """
        self.include_ngrams = include_ngrams
        self.ngram_range = ngram_range

    def extract(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract features from texts

        Args:
            texts: List of text strings

        Returns:
            DataFrame with extracted features
        """
        features = []

        for text in texts:
            text_features = {}

            # Basic statistics
            text_features['char_count'] = len(text)
            text_features['word_count'] = len(text.split())
            text_features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
            text_features['avg_word_length'] = np.mean([len(w) for w in text.split()]) if text.split() else 0
            text_features['unique_word_count'] = len(set(text.lower().split()))

            # Character-level features
            text_features['uppercase_count'] = sum(1 for c in text if c.isupper())
            text_features['lowercase_count'] = sum(1 for c in text if c.islower())
            text_features['digit_count'] = sum(1 for c in text if c.isdigit())
            text_features['space_count'] = sum(1 for c in text if c.isspace())
            text_features['punctuation_count'] = sum(1 for c in text if c in '.,!?;:')

            # Special characters
            text_features['exclamation_count'] = text.count('!')
            text_features['question_count'] = text.count('?')
            text_features['hashtag_count'] = text.count('#')
            text_features['mention_count'] = text.count('@')

            # Readability (simple metrics)
            words = text.split()
            if words:
                text_features['avg_sentence_length'] = len(words) / max(text_features['sentence_count'], 1)

            features.append(text_features)

        return pd.DataFrame(features)

    def extract_ngrams(self, texts: List[str], n: int = 2) -> pd.DataFrame:
        """Extract n-gram features"""
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(ngram_range=(n, n), max_features=100)
        ngram_matrix = vectorizer.fit_transform(texts)

        feature_names = [f'ngram_{name}' for name in vectorizer.get_feature_names_out()]
        return pd.DataFrame(ngram_matrix.toarray(), columns=feature_names)


class TimeSeriesFeatureExtractor:
    """
    Extract features from time series data

    Features:
    - Statistical features (mean, std, min, max, etc.)
    - Trend features
    - Seasonality features
    - Autocorrelation features
    - Frequency domain features
    - Peak/valley features
    """

    def __init__(self):
        """Initialize time series feature extractor"""
        pass

    def extract(self, time_series: np.ndarray) -> Dict[str, float]:
        """
        Extract features from a single time series

        Args:
            time_series: 1D numpy array

        Returns:
            Dictionary of features
        """
        features = {}

        # Basic statistics
        features['mean'] = np.mean(time_series)
        features['std'] = np.std(time_series)
        features['min'] = np.min(time_series)
        features['max'] = np.max(time_series)
        features['median'] = np.median(time_series)
        features['range'] = features['max'] - features['min']
        features['iqr'] = np.percentile(time_series, 75) - np.percentile(time_series, 25)

        # Percentiles
        for p in [10, 25, 75, 90]:
            features[f'percentile_{p}'] = np.percentile(time_series, p)

        if SCIPY_AVAILABLE:
            # Skewness and kurtosis
            features['skewness'] = stats.skew(time_series)
            features['kurtosis'] = stats.kurtosis(time_series)

        # Trend features
        x = np.arange(len(time_series))
        slope, intercept = np.polyfit(x, time_series, 1)
        features['trend_slope'] = slope
        features['trend_intercept'] = intercept

        # Variation features
        features['coefficient_variation'] = features['std'] / features['mean'] if features['mean'] != 0 else 0
        features['mean_abs_change'] = np.mean(np.abs(np.diff(time_series)))
        features['mean_change'] = np.mean(np.diff(time_series))

        # Zero crossing
        features['zero_crossing_count'] = np.sum(np.diff(np.sign(time_series)) != 0)

        # Peak features
        if SCIPY_AVAILABLE:
            peaks, _ = find_peaks(time_series)
            features['peak_count'] = len(peaks)
            features['peak_mean_value'] = np.mean(time_series[peaks]) if len(peaks) > 0 else 0

        # Autocorrelation
        if len(time_series) > 1:
            features['autocorr_lag1'] = np.corrcoef(time_series[:-1], time_series[1:])[0, 1]

        # Energy
        features['energy'] = np.sum(time_series ** 2)
        features['abs_energy'] = np.sum(np.abs(time_series) ** 2)

        return features

    def extract_batch(self, time_series_list: List[np.ndarray]) -> pd.DataFrame:
        """
        Extract features from multiple time series

        Args:
            time_series_list: List of 1D numpy arrays

        Returns:
            DataFrame with extracted features
        """
        features = [self.extract(ts) for ts in time_series_list]
        return pd.DataFrame(features)


class DateTimeFeatureExtractor:
    """
    Extract features from datetime columns

    Features:
    - Year, month, day, day of week, hour, minute
    - Is weekend, is month start/end
    - Quarter, week of year
    - Cyclical encoding (sin/cos)
    - Time differences
    """

    def __init__(self, cyclical_encoding: bool = True):
        """
        Initialize datetime feature extractor

        Args:
            cyclical_encoding: Use sin/cos encoding for cyclical features
        """
        self.cyclical_encoding = cyclical_encoding

    def extract(self, datetimes: pd.Series) -> pd.DataFrame:
        """
        Extract features from datetime series

        Args:
            datetimes: Pandas Series of datetime objects

        Returns:
            DataFrame with extracted features
        """
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(datetimes):
            datetimes = pd.to_datetime(datetimes)

        features = pd.DataFrame()

        # Basic temporal features
        features['year'] = datetimes.dt.year
        features['month'] = datetimes.dt.month
        features['day'] = datetimes.dt.day
        features['day_of_week'] = datetimes.dt.dayofweek
        features['day_of_year'] = datetimes.dt.dayofyear
        features['week_of_year'] = datetimes.dt.isocalendar().week
        features['quarter'] = datetimes.dt.quarter

        # Time features
        features['hour'] = datetimes.dt.hour
        features['minute'] = datetimes.dt.minute
        features['second'] = datetimes.dt.second

        # Boolean features
        features['is_weekend'] = (datetimes.dt.dayofweek >= 5).astype(int)
        features['is_month_start'] = datetimes.dt.is_month_start.astype(int)
        features['is_month_end'] = datetimes.dt.is_month_end.astype(int)
        features['is_quarter_start'] = datetimes.dt.is_quarter_start.astype(int)
        features['is_quarter_end'] = datetimes.dt.is_quarter_end.astype(int)
        features['is_year_start'] = datetimes.dt.is_year_start.astype(int)
        features['is_year_end'] = datetimes.dt.is_year_end.astype(int)

        # Cyclical encoding
        if self.cyclical_encoding:
            # Month (1-12)
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

            # Day of week (0-6)
            features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)

            # Hour (0-23)
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)

        return features


class StructuredFeatureExtractor:
    """
    Extract features from structured/tabular data

    Features:
    - Aggregation features (sum, mean, std, etc.)
    - Ratio features
    - Difference features
    - Count features
    - Null features
    - Interaction features
    """

    def __init__(self):
        """Initialize structured feature extractor"""
        pass

    def extract_aggregations(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        agg_cols: List[str],
        agg_funcs: List[str] = ['mean', 'std', 'min', 'max', 'sum']
    ) -> pd.DataFrame:
        """
        Extract aggregation features

        Args:
            df: Input DataFrame
            group_cols: Columns to group by
            agg_cols: Columns to aggregate
            agg_funcs: Aggregation functions

        Returns:
            DataFrame with aggregation features
        """
        agg_dict = {col: agg_funcs for col in agg_cols}
        agg_df = df.groupby(group_cols).agg(agg_dict)

        # Flatten column names
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        agg_df = agg_df.reset_index()

        return agg_df

    def extract_ratios(self, df: pd.DataFrame, col_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Extract ratio features

        Args:
            df: Input DataFrame
            col_pairs: List of (numerator, denominator) column pairs

        Returns:
            DataFrame with ratio features
        """
        ratio_features = pd.DataFrame(index=df.index)

        for col1, col2 in col_pairs:
            feature_name = f'{col1}_div_{col2}'
            ratio_features[feature_name] = df[col1] / (df[col2] + 1e-8)  # Avoid division by zero

        return ratio_features

    def extract_differences(self, df: pd.DataFrame, col_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Extract difference features

        Args:
            df: Input DataFrame
            col_pairs: List of (col1, col2) column pairs

        Returns:
            DataFrame with difference features
        """
        diff_features = pd.DataFrame(index=df.index)

        for col1, col2 in col_pairs:
            feature_name = f'{col1}_minus_{col2}'
            diff_features[feature_name] = df[col1] - df[col2]

        return diff_features

    def extract_null_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract null/missing value features

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with null features
        """
        null_features = pd.DataFrame(index=df.index)

        # Null count per row
        null_features['null_count'] = df.isnull().sum(axis=1)

        # Null indicator per column
        for col in df.columns:
            null_features[f'{col}_is_null'] = df[col].isnull().astype(int)

        return null_features

    def extract_count_features(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str]
    ) -> pd.DataFrame:
        """
        Extract count features for categorical variables

        Args:
            df: Input DataFrame
            categorical_cols: Categorical columns

        Returns:
            DataFrame with count features
        """
        count_features = pd.DataFrame(index=df.index)

        for col in categorical_cols:
            # Count of each category
            value_counts = df[col].value_counts()
            count_features[f'{col}_count'] = df[col].map(value_counts)

            # Frequency (normalized count)
            freq = value_counts / len(df)
            count_features[f'{col}_freq'] = df[col].map(freq)

        return count_features


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance using various methods

    Methods:
    - Random Forest importance
    - Permutation importance
    - SHAP values
    - Correlation with target
    - Mutual information
    """

    def __init__(self):
        """Initialize feature importance analyzer"""
        self.importance_scores = {}

    def analyze_random_forest(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_estimators: int = 100
    ) -> pd.DataFrame:
        """
        Analyze importance using Random Forest

        Args:
            X: Features
            y: Target
            n_estimators: Number of trees

        Returns:
            DataFrame with feature importances
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        # Determine task type
        if len(np.unique(y)) < 20:
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

        model.fit(X, y)

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.importance_scores['random_forest'] = importance_df

        return importance_df

    def analyze_permutation(
        self,
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        n_repeats: int = 10
    ) -> pd.DataFrame:
        """
        Analyze importance using permutation importance

        Args:
            model: Trained model
            X: Features
            y: Target
            n_repeats: Number of permutations

        Returns:
            DataFrame with feature importances
        """
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': result.importances_mean,
            'std': result.importances_std
        }).sort_values('importance', ascending=False)

        self.importance_scores['permutation'] = importance_df

        return importance_df

    def analyze_correlation(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> pd.DataFrame:
        """
        Analyze correlation with target

        Args:
            X: Features
            y: Target

        Returns:
            DataFrame with correlations
        """
        correlations = []

        for col in X.columns:
            corr = np.corrcoef(X[col], y)[0, 1]
            correlations.append({
                'feature': col,
                'correlation': abs(corr),
                'correlation_raw': corr
            })

        correlation_df = pd.DataFrame(correlations).sort_values(
            'correlation', ascending=False
        )

        self.importance_scores['correlation'] = correlation_df

        return correlation_df

    def analyze_mutual_info(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        task: str = 'classification'
    ) -> pd.DataFrame:
        """
        Analyze mutual information

        Args:
            X: Features
            y: Target
            task: 'classification' or 'regression'

        Returns:
            DataFrame with mutual information scores
        """
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        if task == 'classification':
            mi_scores = mutual_info_classif(X, y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=42)

        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)

        self.importance_scores['mutual_info'] = mi_df

        return mi_df

    def plot_importance(
        self,
        method: str = 'random_forest',
        top_k: int = 20,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """Plot feature importance"""
        import matplotlib.pyplot as plt

        if method not in self.importance_scores:
            raise ValueError(f"No importance scores for method: {method}")

        df = self.importance_scores[method].head(top_k)

        plt.figure(figsize=figsize)
        plt.barh(range(len(df)), df.iloc[:, 1])
        plt.yticks(range(len(df)), df['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_k} Features - {method}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def get_top_features(
        self,
        method: str = 'random_forest',
        top_k: int = 10
    ) -> List[str]:
        """Get list of top k features"""
        if method not in self.importance_scores:
            raise ValueError(f"No importance scores for method: {method}")

        return self.importance_scores[method].head(top_k)['feature'].tolist()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Feature Extraction Test")
    print("=" * 70)

    # Test Text Feature Extraction
    print("\n1. Text Feature Extraction")
    print("-" * 70)

    texts = [
        "This is a sample text with multiple sentences. It has various features!",
        "Another example with different characteristics and structure?",
        "Short text.",
        "This text contains #hashtags and @mentions for testing purposes!"
    ]

    text_extractor = TextFeatureExtractor()
    text_features = text_extractor.extract(texts)

    print("Text features extracted:")
    print(text_features[['char_count', 'word_count', 'sentence_count', 'uppercase_count']].head())

    # Test Time Series Feature Extraction
    print("\n2. Time Series Feature Extraction")
    print("-" * 70)

    # Generate sample time series
    time_series = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.1

    ts_extractor = TimeSeriesFeatureExtractor()
    ts_features = ts_extractor.extract(time_series)

    print("Time series features extracted:")
    for key, value in list(ts_features.items())[:10]:
        print(f"  {key}: {value:.4f}")

    # Test DateTime Feature Extraction
    print("\n3. DateTime Feature Extraction")
    print("-" * 70)

    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    datetime_extractor = DateTimeFeatureExtractor(cyclical_encoding=True)
    datetime_features = datetime_extractor.extract(pd.Series(dates))

    print("DateTime features extracted:")
    print(datetime_features[['year', 'month', 'day', 'day_of_week', 'is_weekend']].head())

    # Test Structured Feature Extraction
    print("\n4. Structured Feature Extraction")
    print("-" * 70)

    # Sample data
    df = pd.DataFrame({
        'category': ['A', 'B', 'A', 'C', 'B', 'A'],
        'value1': [10, 20, 15, 25, 30, 12],
        'value2': [5, 10, 8, 12, 15, 6]
    })

    struct_extractor = StructuredFeatureExtractor()

    # Extract ratio features
    ratio_features = struct_extractor.extract_ratios(df, [('value1', 'value2')])
    print("\nRatio features:")
    print(ratio_features.head())

    # Extract count features
    count_features = struct_extractor.extract_count_features(df, ['category'])
    print("\nCount features:")
    print(count_features.head())

    # Test Feature Importance Analysis
    print("\n5. Feature Importance Analysis")
    print("-" * 70)

    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])

    importance_analyzer = FeatureImportanceAnalyzer()

    # Random Forest importance
    rf_importance = importance_analyzer.analyze_random_forest(X_df, y)
    print("\nTop 10 features by Random Forest importance:")
    print(rf_importance.head(10))

    # Correlation analysis
    corr_importance = importance_analyzer.analyze_correlation(X_df, y)
    print("\nTop 10 features by correlation:")
    print(corr_importance.head(10))

    print("\n" + "=" * 70)
    print("Feature extraction tested successfully!")
