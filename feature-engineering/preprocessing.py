"""
Feature Preprocessing
Scaling, normalization, encoding, and transformation utilities
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    Normalizer, PowerTransformer, QuantileTransformer,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    Binarizer, KBinsDiscretizer, PolynomialFeatures
)
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Union, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureScaler:
    """
    Comprehensive feature scaling utilities

    Supports multiple scaling methods:
    - StandardScaler: (X - mean) / std
    - MinMaxScaler: (X - min) / (max - min)
    - RobustScaler: Robust to outliers, uses median and IQR
    - MaxAbsScaler: Scale by maximum absolute value
    - Normalizer: Scale samples individually to unit norm

    Usage:
        scaler = FeatureScaler(method='standard')
        X_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    """

    def __init__(
        self,
        method: str = 'standard',
        feature_range: Tuple[float, float] = (0, 1),
        **kwargs
    ):
        """
        Initialize feature scaler

        Args:
            method: Scaling method
                - 'standard': StandardScaler (z-score normalization)
                - 'minmax': MinMaxScaler
                - 'robust': RobustScaler (uses median and IQR)
                - 'maxabs': MaxAbsScaler
                - 'normalizer': Normalizer (L1, L2, or max norm)
            feature_range: Range for MinMaxScaler
            **kwargs: Additional arguments for the scaler
        """
        self.method = method
        self.feature_range = feature_range

        if method == 'standard':
            self.scaler = StandardScaler(**kwargs)
        elif method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=feature_range, **kwargs)
        elif method == 'robust':
            self.scaler = RobustScaler(**kwargs)
        elif method == 'maxabs':
            self.scaler = MaxAbsScaler(**kwargs)
        elif method == 'normalizer':
            # Normalizer doesn't need fit, works on samples
            norm = kwargs.get('norm', 'l2')
            self.scaler = Normalizer(norm=norm)
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        self.fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        """Fit scaler to data"""
        self.scaler.fit(X)
        self.fitted = True
        logger.info(f"FeatureScaler ({self.method}) fitted")
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data"""
        if not self.fitted and self.method != 'normalizer':
            raise ValueError("Scaler not fitted. Call fit() first.")
        return self.scaler.transform(X)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform data"""
        if self.method == 'normalizer':
            # Normalizer doesn't need fit
            return self.scaler.transform(X)
        return self.scaler.fit_transform(X)

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Inverse transform (not available for Normalizer)"""
        if self.method == 'normalizer':
            raise ValueError("Normalizer does not support inverse transform")
        return self.scaler.inverse_transform(X)


class FeatureTransformer:
    """
    Advanced feature transformations

    Supports:
    - PowerTransformer: Box-Cox and Yeo-Johnson transformations
    - QuantileTransformer: Transform to uniform or normal distribution
    - Log transformation
    - Square root transformation
    """

    def __init__(
        self,
        method: str = 'yeo-johnson',
        n_quantiles: int = 1000,
        output_distribution: str = 'uniform'
    ):
        """
        Initialize feature transformer

        Args:
            method: Transformation method
                - 'yeo-johnson': Yeo-Johnson power transform (works with negative values)
                - 'box-cox': Box-Cox power transform (only positive values)
                - 'quantile': Quantile transformer
                - 'log': Log transformation (log1p)
                - 'sqrt': Square root transformation
            n_quantiles: Number of quantiles for QuantileTransformer
            output_distribution: Output distribution for QuantileTransformer ('uniform' or 'normal')
        """
        self.method = method

        if method == 'yeo-johnson':
            self.transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        elif method == 'box-cox':
            self.transformer = PowerTransformer(method='box-cox', standardize=True)
        elif method == 'quantile':
            self.transformer = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution=output_distribution,
                random_state=42
            )
        elif method in ['log', 'sqrt']:
            self.transformer = None  # Custom implementation
        else:
            raise ValueError(f"Unknown transformation method: {method}")

        self.fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        """Fit transformer to data"""
        if self.transformer is not None:
            self.transformer.fit(X)
        self.fitted = True
        logger.info(f"FeatureTransformer ({self.method}) fitted")
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data"""
        if self.method == 'log':
            return np.log1p(np.abs(X)) * np.sign(X)
        elif self.method == 'sqrt':
            return np.sqrt(np.abs(X)) * np.sign(X)
        else:
            if not self.fitted:
                raise ValueError("Transformer not fitted. Call fit() first.")
            return self.transformer.transform(X)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform data"""
        if self.method in ['log', 'sqrt']:
            return self.transform(X)
        return self.transformer.fit_transform(X)

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Inverse transform"""
        if self.method == 'log':
            return np.expm1(np.abs(X)) * np.sign(X)
        elif self.method == 'sqrt':
            return np.square(X)
        else:
            return self.transformer.inverse_transform(X)


class CategoricalEncoder:
    """
    Categorical variable encoding

    Supports:
    - Label Encoding: Convert categories to integers
    - One-Hot Encoding: Create binary columns for each category
    - Ordinal Encoding: Encode ordinal categories with order
    - Target Encoding: Encode based on target variable (for supervised learning)
    """

    def __init__(
        self,
        method: str = 'onehot',
        handle_unknown: str = 'ignore',
        drop: Optional[str] = None
    ):
        """
        Initialize categorical encoder

        Args:
            method: Encoding method
                - 'label': LabelEncoder
                - 'onehot': OneHotEncoder
                - 'ordinal': OrdinalEncoder
                - 'target': Target encoding
            handle_unknown: How to handle unknown categories ('error' or 'ignore')
            drop: Whether to drop one category to avoid multicollinearity ('first', 'if_binary', or None)
        """
        self.method = method
        self.handle_unknown = handle_unknown
        self.drop = drop

        if method == 'label':
            self.encoder = LabelEncoder()
        elif method == 'onehot':
            self.encoder = OneHotEncoder(
                handle_unknown=handle_unknown,
                drop=drop,
                sparse_output=False
            )
        elif method == 'ordinal':
            self.encoder = OrdinalEncoder(handle_unknown=handle_unknown)
        elif method == 'target':
            self.encoder = None  # Custom implementation
            self.target_map = {}
        else:
            raise ValueError(f"Unknown encoding method: {method}")

        self.fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
        """
        Fit encoder to data

        Args:
            X: Features
            y: Target variable (required for target encoding)
        """
        if self.method == 'target':
            if y is None:
                raise ValueError("Target encoding requires target variable y")

            # Calculate mean target for each category
            if isinstance(X, pd.DataFrame):
                for col in X.columns:
                    self.target_map[col] = X[col].groupby(X[col]).agg({col: lambda x: y[x.index].mean()})
            else:
                # Assume single column
                df = pd.DataFrame({'X': X.flatten(), 'y': y})
                self.target_map = df.groupby('X')['y'].mean().to_dict()
        else:
            self.encoder.fit(X)

        self.fitted = True
        logger.info(f"CategoricalEncoder ({self.method}) fitted")
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data"""
        if not self.fitted:
            raise ValueError("Encoder not fitted. Call fit() first.")

        if self.method == 'target':
            if isinstance(X, pd.DataFrame):
                result = X.copy()
                for col in X.columns:
                    result[col] = X[col].map(self.target_map[col])
                return result.values
            else:
                return np.array([self.target_map.get(x, 0) for x in X.flatten()])
        else:
            return self.encoder.transform(X)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform data"""
        if self.method == 'target':
            self.fit(X, y)
            return self.transform(X)
        return self.encoder.fit_transform(X)

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Inverse transform (not available for target encoding)"""
        if self.method == 'target':
            raise ValueError("Target encoding does not support inverse transform")
        return self.encoder.inverse_transform(X)

    def get_feature_names(self) -> List[str]:
        """Get feature names after encoding"""
        if self.method == 'onehot':
            return self.encoder.get_feature_names_out().tolist()
        else:
            return []


class BinningTransformer:
    """
    Discretize continuous features into bins

    Useful for:
    - Creating categorical features from continuous ones
    - Capturing non-linear relationships
    - Reducing impact of outliers
    """

    def __init__(
        self,
        n_bins: int = 5,
        strategy: str = 'quantile',
        encode: str = 'ordinal'
    ):
        """
        Initialize binning transformer

        Args:
            n_bins: Number of bins
            strategy: Binning strategy
                - 'uniform': Equal width bins
                - 'quantile': Equal frequency bins
                - 'kmeans': K-means clustering
            encode: Encoding method
                - 'ordinal': Encode bins as integers
                - 'onehot': One-hot encode bins
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode

        self.binner = KBinsDiscretizer(
            n_bins=n_bins,
            encode=encode,
            strategy=strategy
        )

        self.fitted = False

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        """Fit binner to data"""
        self.binner.fit(X)
        self.fitted = True
        logger.info(f"BinningTransformer fitted with {self.n_bins} bins ({self.strategy})")
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data"""
        if not self.fitted:
            raise ValueError("Binner not fitted. Call fit() first.")
        return self.binner.transform(X)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform data"""
        return self.binner.fit_transform(X)


class FeatureInteractionGenerator:
    """
    Generate interaction features between variables

    Creates:
    - Polynomial features (x^2, x^3, etc.)
    - Interaction terms (x1 * x2, x1 * x2 * x3, etc.)
    - Ratio features (x1 / x2)
    - Difference features (x1 - x2)
    """

    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
        include_ratios: bool = False,
        include_differences: bool = False
    ):
        """
        Initialize feature interaction generator

        Args:
            degree: Polynomial degree
            interaction_only: Only include interaction terms (no x^2, x^3, etc.)
            include_bias: Include bias column (all 1s)
            include_ratios: Include ratio features (x1/x2)
            include_differences: Include difference features (x1-x2)
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.include_ratios = include_ratios
        self.include_differences = include_differences

        self.poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias
        )

        self.fitted = False
        self.n_features_in_ = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        """Fit transformer"""
        self.poly.fit(X)
        self.n_features_in_ = X.shape[1]
        self.fitted = True
        logger.info(f"FeatureInteractionGenerator fitted (degree={self.degree})")
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data"""
        if not self.fitted:
            raise ValueError("Generator not fitted. Call fit() first.")

        # Polynomial features
        X_poly = self.poly.transform(X)

        # Additional features
        additional_features = []

        if self.include_ratios:
            # Create ratio features
            for i in range(self.n_features_in_):
                for j in range(i + 1, self.n_features_in_):
                    # Avoid division by zero
                    ratio = np.divide(
                        X[:, i],
                        X[:, j],
                        out=np.zeros_like(X[:, i]),
                        where=X[:, j] != 0
                    )
                    additional_features.append(ratio.reshape(-1, 1))

        if self.include_differences:
            # Create difference features
            for i in range(self.n_features_in_):
                for j in range(i + 1, self.n_features_in_):
                    diff = X[:, i] - X[:, j]
                    additional_features.append(diff.reshape(-1, 1))

        if additional_features:
            additional_features = np.hstack(additional_features)
            X_poly = np.hstack([X_poly, additional_features])

        return X_poly

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform data"""
        self.fit(X)
        return self.transform(X)

    def get_feature_names(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get feature names after transformation"""
        if input_features is None:
            input_features = [f'x{i}' for i in range(self.n_features_in_)]

        feature_names = self.poly.get_feature_names_out(input_features).tolist()

        # Add ratio and difference feature names
        if self.include_ratios:
            for i in range(self.n_features_in_):
                for j in range(i + 1, self.n_features_in_):
                    feature_names.append(f'{input_features[i]}/{input_features[j]}')

        if self.include_differences:
            for i in range(self.n_features_in_):
                for j in range(i + 1, self.n_features_in_):
                    feature_names.append(f'{input_features[i]}-{input_features[j]}')

        return feature_names


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Feature Preprocessing Test")
    print("=" * 70)

    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 3) * 10 + 50
    y = np.random.randint(0, 2, 100)

    print(f"Original data shape: {X.shape}")
    print(f"Data range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"Data mean: {X.mean():.2f}, std: {X.std():.2f}")

    # Test StandardScaler
    print("\n1. Standard Scaling")
    print("-" * 70)
    scaler = FeatureScaler(method='standard')
    X_scaled = scaler.fit_transform(X)
    print(f"Scaled range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    print(f"Scaled mean: {X_scaled.mean():.2f}, std: {X_scaled.std():.2f}")

    # Test MinMaxScaler
    print("\n2. MinMax Scaling")
    print("-" * 70)
    scaler = FeatureScaler(method='minmax', feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    print(f"Scaled range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

    # Test PowerTransformer
    print("\n3. Power Transformation (Yeo-Johnson)")
    print("-" * 70)
    transformer = FeatureTransformer(method='yeo-johnson')
    X_transformed = transformer.fit_transform(X)
    print(f"Transformed mean: {X_transformed.mean():.2f}, std: {X_transformed.std():.2f}")

    # Test Categorical Encoding
    print("\n4. Categorical Encoding")
    print("-" * 70)
    X_cat = np.random.choice(['A', 'B', 'C'], size=(100, 1))

    # Label encoding
    encoder = CategoricalEncoder(method='label')
    X_label = encoder.fit_transform(X_cat)
    print(f"Label encoded shape: {X_label.shape}")
    print(f"Unique values: {np.unique(X_label)}")

    # One-hot encoding
    encoder = CategoricalEncoder(method='onehot')
    X_onehot = encoder.fit_transform(X_cat)
    print(f"One-hot encoded shape: {X_onehot.shape}")
    print(f"Feature names: {encoder.get_feature_names()}")

    # Test Binning
    print("\n5. Binning/Discretization")
    print("-" * 70)
    binner = BinningTransformer(n_bins=5, strategy='quantile', encode='ordinal')
    X_binned = binner.fit_transform(X[:, 0:1])
    print(f"Original values (first 5): {X[:5, 0]}")
    print(f"Binned values (first 5): {X_binned[:5, 0]}")
    print(f"Unique bins: {np.unique(X_binned)}")

    # Test Feature Interactions
    print("\n6. Feature Interactions (Polynomial)")
    print("-" * 70)
    interaction_gen = FeatureInteractionGenerator(
        degree=2,
        include_ratios=True,
        include_differences=True
    )
    X_interactions = interaction_gen.fit_transform(X)
    print(f"Original features: {X.shape[1]}")
    print(f"Features after interactions: {X_interactions.shape[1]}")

    feature_names = interaction_gen.get_feature_names(['feat1', 'feat2', 'feat3'])
    print(f"First 10 feature names: {feature_names[:10]}")

    print("\n" + "=" * 70)
    print("Feature preprocessing tested successfully!")
