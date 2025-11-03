"""
Complete Feature Engineering Pipeline
End-to-end feature engineering combining extraction, preprocessing, selection, and reduction
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Any
import logging
from pathlib import Path
import pickle

from preprocessing import (
    FeatureScaler, FeatureTransformer, CategoricalEncoder,
    BinningTransformer, FeatureInteractionGenerator
)
from selection import FeatureSelector, DimensionalityReducer
from extraction import (
    TextFeatureExtractor, TimeSeriesFeatureExtractor,
    DateTimeFeatureExtractor, StructuredFeatureExtractor,
    FeatureImportanceAnalyzer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Complete Feature Engineering Pipeline

    Workflow:
    1. Feature Extraction (text, datetime, structured)
    2. Feature Preprocessing (scaling, encoding, transformation)
    3. Feature Generation (interactions, polynomials)
    4. Feature Selection (statistical, model-based)
    5. Dimensionality Reduction (optional)

    Usage:
        pipeline = FeatureEngineeringPipeline()
        X_processed = pipeline.fit_transform(X_train, y_train)
        X_test_processed = pipeline.transform(X_test)
    """

    def __init__(
        self,
        # Extraction
        extract_text_features: bool = False,
        extract_datetime_features: bool = True,
        text_columns: Optional[List[str]] = None,
        datetime_columns: Optional[List[str]] = None,

        # Preprocessing
        scaling_method: str = 'standard',
        encoding_method: str = 'onehot',
        handle_missing: str = 'mean',  # 'mean', 'median', 'drop'

        # Feature generation
        create_interactions: bool = False,
        interaction_degree: int = 2,

        # Feature selection
        select_features: bool = True,
        selection_method: str = 'mutual_info',
        n_features_to_select: int = 50,

        # Dimensionality reduction
        reduce_dimensions: bool = False,
        reduction_method: str = 'pca',
        n_components: int = 10,

        # General
        verbose: bool = True
    ):
        """
        Initialize feature engineering pipeline

        Args:
            extract_text_features: Extract features from text columns
            extract_datetime_features: Extract features from datetime columns
            text_columns: Names of text columns
            datetime_columns: Names of datetime columns
            scaling_method: Scaling method
            encoding_method: Categorical encoding method
            handle_missing: How to handle missing values
            create_interactions: Create interaction features
            interaction_degree: Degree for polynomial interactions
            select_features: Perform feature selection
            selection_method: Feature selection method
            n_features_to_select: Number of features to select
            reduce_dimensions: Perform dimensionality reduction
            reduction_method: Dimensionality reduction method
            n_components: Number of components for reduction
            verbose: Print progress
        """
        # Configuration
        self.extract_text_features = extract_text_features
        self.extract_datetime_features = extract_datetime_features
        self.text_columns = text_columns or []
        self.datetime_columns = datetime_columns or []

        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        self.handle_missing = handle_missing

        self.create_interactions = create_interactions
        self.interaction_degree = interaction_degree

        self.select_features = select_features
        self.selection_method = selection_method
        self.n_features_to_select = n_features_to_select

        self.reduce_dimensions = reduce_dimensions
        self.reduction_method = reduction_method
        self.n_components = n_components

        self.verbose = verbose

        # Components (initialized during fit)
        self.text_extractor = None
        self.datetime_extractor = None
        self.scaler = None
        self.categorical_encoders = {}
        self.interaction_generator = None
        self.feature_selector = None
        self.dimensionality_reducer = None

        # Metadata
        self.numeric_columns = []
        self.categorical_columns = []
        self.feature_names_in_ = []
        self.feature_names_out_ = []
        self.fitted = False

    def _log(self, message: str):
        """Log message if verbose"""
        if self.verbose:
            logger.info(message)

    def _identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify numeric and categorical columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove text and datetime columns from categorical
        categorical_cols = [
            col for col in categorical_cols
            if col not in self.text_columns and col not in self.datetime_columns
        ]

        return numeric_cols, categorical_cols

    def _handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle missing values"""
        df = df.copy()

        if self.handle_missing == 'drop':
            df = df.dropna()
        elif self.handle_missing == 'mean':
            if fit:
                self.fill_values = df[self.numeric_columns].mean()
            df[self.numeric_columns] = df[self.numeric_columns].fillna(self.fill_values)
        elif self.handle_missing == 'median':
            if fit:
                self.fill_values = df[self.numeric_columns].median()
            df[self.numeric_columns] = df[self.numeric_columns].fillna(self.fill_values)

        # Fill categorical with mode or 'missing'
        for col in self.categorical_columns:
            if fit:
                if col not in hasattr(self, 'categorical_fill_values'):
                    self.categorical_fill_values = {}
                self.categorical_fill_values[col] = df[col].mode()[0] if len(df[col].mode()) > 0 else 'missing'
            df[col] = df[col].fillna(self.categorical_fill_values.get(col, 'missing'))

        return df

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """
        Fit the feature engineering pipeline

        Args:
            X: Input features
            y: Target variable (optional, needed for some steps)
        """
        self._log("=" * 70)
        self._log("Fitting Feature Engineering Pipeline")
        self._log("=" * 70)

        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = X.columns.tolist()

        # Identify column types
        self.numeric_columns, self.categorical_columns = self._identify_column_types(X)

        self._log(f"Numeric columns: {len(self.numeric_columns)}")
        self._log(f"Categorical columns: {len(self.categorical_columns)}")
        self._log(f"Text columns: {len(self.text_columns)}")
        self._log(f"DateTime columns: {len(self.datetime_columns)}")

        # Handle missing values
        X = self._handle_missing_values(X, fit=True)

        # Extract features
        extracted_features = []

        # Text features
        if self.extract_text_features and self.text_columns:
            self._log("\nExtracting text features...")
            self.text_extractor = TextFeatureExtractor()
            for col in self.text_columns:
                if col in X.columns:
                    text_feats = self.text_extractor.extract(X[col].tolist())
                    text_feats.columns = [f'{col}_{c}' for c in text_feats.columns]
                    extracted_features.append(text_feats)
                    # Remove original text column
                    X = X.drop(columns=[col])

        # DateTime features
        if self.extract_datetime_features and self.datetime_columns:
            self._log("Extracting datetime features...")
            self.datetime_extractor = DateTimeFeatureExtractor(cyclical_encoding=True)
            for col in self.datetime_columns:
                if col in X.columns:
                    dt_feats = self.datetime_extractor.extract(X[col])
                    dt_feats.columns = [f'{col}_{c}' for c in dt_feats.columns]
                    extracted_features.append(dt_feats)
                    # Remove original datetime column
                    X = X.drop(columns=[col])

        # Combine extracted features
        if extracted_features:
            X = pd.concat([X] + extracted_features, axis=1)
            # Update column types
            self.numeric_columns, self.categorical_columns = self._identify_column_types(X)

        # Encode categorical features
        if self.categorical_columns:
            self._log(f"\nEncoding categorical features ({self.encoding_method})...")
            for col in self.categorical_columns:
                encoder = CategoricalEncoder(method=self.encoding_method, handle_unknown='ignore')
                encoder.fit(X[[col]], y)
                self.categorical_encoders[col] = encoder

            # Apply encoding
            encoded_features = []
            for col in self.categorical_columns:
                encoded = self.categorical_encoders[col].transform(X[[col]])
                if self.encoding_method == 'onehot':
                    # Get feature names for one-hot
                    feature_names = self.categorical_encoders[col].get_feature_names()
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
                else:
                    encoded_df = pd.DataFrame(encoded, columns=[f'{col}_encoded'], index=X.index)
                encoded_features.append(encoded_df)

            # Remove original categorical columns
            X = X.drop(columns=self.categorical_columns)

            # Add encoded features
            if encoded_features:
                X = pd.concat([X] + encoded_features, axis=1)

        # Create interactions
        if self.create_interactions:
            self._log(f"\nCreating interaction features (degree={self.interaction_degree})...")
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) > 0:
                self.interaction_generator = FeatureInteractionGenerator(
                    degree=self.interaction_degree,
                    interaction_only=False,
                    include_ratios=False,
                    include_differences=False
                )

                X_numeric = X[numeric_cols].values
                X_interactions = self.interaction_generator.fit_transform(X_numeric)

                # Get feature names
                feature_names = self.interaction_generator.get_feature_names(numeric_cols)

                # Replace numeric columns with interactions
                X = X.drop(columns=numeric_cols)
                X_interactions_df = pd.DataFrame(X_interactions, columns=feature_names, index=X.index)
                X = pd.concat([X, X_interactions_df], axis=1)

        # Scale numeric features
        self._log(f"\nScaling features ({self.scaling_method})...")
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) > 0:
            self.scaler = FeatureScaler(method=self.scaling_method)
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])

        # Feature selection
        if self.select_features and y is not None:
            self._log(f"\nSelecting features ({self.selection_method}, k={self.n_features_to_select})...")
            self.feature_selector = FeatureSelector(
                method=self.selection_method,
                k=min(self.n_features_to_select, X.shape[1])
            )

            X_selected = self.feature_selector.fit_transform(X.values, y)
            selected_features = self.feature_selector.get_selected_features()

            # Update X with selected features
            if isinstance(selected_features[0], str):
                X = X[selected_features]
            else:
                X = X.iloc[:, selected_features]

        # Dimensionality reduction
        if self.reduce_dimensions:
            self._log(f"\nReducing dimensions ({self.reduction_method}, n={self.n_components})...")
            self.dimensionality_reducer = DimensionalityReducer(
                method=self.reduction_method,
                n_components=min(self.n_components, X.shape[1])
            )

            X_reduced = self.dimensionality_reducer.fit_transform(X.values, y)

            # Create new column names
            component_names = [f'component_{i}' for i in range(X_reduced.shape[1])]
            X = pd.DataFrame(X_reduced, columns=component_names, index=X.index)

        self.feature_names_out_ = X.columns.tolist()
        self.fitted = True

        self._log("=" * 70)
        self._log("Pipeline Fitting Complete")
        self._log(f"Input features: {len(self.feature_names_in_)}")
        self._log(f"Output features: {len(self.feature_names_out_)}")
        self._log("=" * 70)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted pipeline

        Args:
            X: Input features

        Returns:
            Transformed features
        """
        if not self.fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        # Handle missing values
        X = self._handle_missing_values(X, fit=False)

        # Extract features
        extracted_features = []

        # Text features
        if self.extract_text_features and self.text_columns and self.text_extractor:
            for col in self.text_columns:
                if col in X.columns:
                    text_feats = self.text_extractor.extract(X[col].tolist())
                    text_feats.columns = [f'{col}_{c}' for c in text_feats.columns]
                    extracted_features.append(text_feats)
                    X = X.drop(columns=[col])

        # DateTime features
        if self.extract_datetime_features and self.datetime_columns and self.datetime_extractor:
            for col in self.datetime_columns:
                if col in X.columns:
                    dt_feats = self.datetime_extractor.extract(X[col])
                    dt_feats.columns = [f'{col}_{c}' for c in dt_feats.columns]
                    extracted_features.append(dt_feats)
                    X = X.drop(columns=[col])

        # Combine extracted features
        if extracted_features:
            X = pd.concat([X] + extracted_features, axis=1)

        # Encode categorical features
        if self.categorical_encoders:
            encoded_features = []
            for col in self.categorical_columns:
                if col in X.columns and col in self.categorical_encoders:
                    encoded = self.categorical_encoders[col].transform(X[[col]])
                    if self.encoding_method == 'onehot':
                        feature_names = self.categorical_encoders[col].get_feature_names()
                        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
                    else:
                        encoded_df = pd.DataFrame(encoded, columns=[f'{col}_encoded'], index=X.index)
                    encoded_features.append(encoded_df)

            # Remove original categorical columns
            X = X.drop(columns=[col for col in self.categorical_columns if col in X.columns])

            # Add encoded features
            if encoded_features:
                X = pd.concat([X] + encoded_features, axis=1)

        # Create interactions
        if self.create_interactions and self.interaction_generator:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) > 0:
                X_numeric = X[numeric_cols].values
                X_interactions = self.interaction_generator.transform(X_numeric)

                feature_names = self.interaction_generator.get_feature_names(numeric_cols)

                X = X.drop(columns=numeric_cols)
                X_interactions_df = pd.DataFrame(X_interactions, columns=feature_names, index=X.index)
                X = pd.concat([X, X_interactions_df], axis=1)

        # Scale
        if self.scaler:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                X[numeric_cols] = self.scaler.transform(X[numeric_cols])

        # Select features
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X.values)
            selected_features = self.feature_selector.get_selected_features()

            if isinstance(selected_features[0], str):
                X = X[selected_features]
            else:
                X = X.iloc[:, selected_features]

        # Reduce dimensions
        if self.dimensionality_reducer:
            X_reduced = self.dimensionality_reducer.transform(X.values)
            component_names = [f'component_{i}' for i in range(X_reduced.shape[1])]
            X = pd.DataFrame(X_reduced, columns=component_names, index=X.index)

        return X

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)

    def save(self, path: str):
        """Save pipeline to file"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        self._log(f"Pipeline saved to {path}")

    @staticmethod
    def load(path: str):
        """Load pipeline from file"""
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)
        logger.info(f"Pipeline loaded from {path}")
        return pipeline

    def get_feature_names(self) -> List[str]:
        """Get output feature names"""
        return self.feature_names_out_


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Complete Feature Engineering Pipeline Test")
    print("=" * 70)

    # Create sample data
    np.random.seed(42)

    df = pd.DataFrame({
        'numeric1': np.random.randn(1000),
        'numeric2': np.random.randn(1000) * 10 + 50,
        'numeric3': np.random.randint(0, 100, 1000),
        'category1': np.random.choice(['A', 'B', 'C'], 1000),
        'category2': np.random.choice(['X', 'Y'], 1000),
        'date': pd.date_range('2023-01-01', periods=1000, freq='D'),
    })

    # Create target
    y = (df['numeric1'] + df['numeric2'] / 50 + (df['category1'] == 'A').astype(int) * 5 > 0).astype(int)

    print(f"\nOriginal data shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")

    # Create pipeline
    pipeline = FeatureEngineeringPipeline(
        extract_datetime_features=True,
        datetime_columns=['date'],
        scaling_method='standard',
        encoding_method='onehot',
        create_interactions=True,
        interaction_degree=2,
        select_features=True,
        selection_method='mutual_info',
        n_features_to_select=20,
        reduce_dimensions=False,
        verbose=True
    )

    # Fit and transform
    X_train = df.iloc[:800]
    y_train = y.iloc[:800]
    X_test = df.iloc[800:]

    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    X_test_transformed = pipeline.transform(X_test)

    print(f"\nTransformed training data shape: {X_train_transformed.shape}")
    print(f"Transformed test data shape: {X_test_transformed.shape}")
    print(f"\nOutput features: {pipeline.get_feature_names()[:10]}...")

    # Save and load pipeline
    pipeline.save('feature_pipeline.pkl')
    loaded_pipeline = FeatureEngineeringPipeline.load('feature_pipeline.pkl')

    print("\nPipeline saved and loaded successfully!")

    print("\n" + "=" * 70)
    print("Feature engineering pipeline tested successfully!")
