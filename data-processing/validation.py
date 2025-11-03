"""
Data Validation Module

Comprehensive data validation utilities:
- Schema validation
- Type checking
- Range validation
- Pattern matching
- Consistency checks
- Data quality metrics
- Validation reports

Author: ML Framework Team
"""

import numpy as np
import re
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from collections import Counter


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

class DataValidator:
    """
    Comprehensive data validator with schema support.
    """

    def __init__(self, schema: Optional[Dict] = None):
        """
        Initialize validator.

        Parameters:
        -----------
        schema : dict, optional
            Validation schema defining expected structure and constraints.

            Example schema:
            {
                'feature1': {
                    'type': 'numeric',
                    'min': 0,
                    'max': 100,
                    'nullable': False
                },
                'feature2': {
                    'type': 'categorical',
                    'categories': ['A', 'B', 'C'],
                    'nullable': True
                }
            }
        """
        self.schema = schema
        self.validation_errors = []

    def validate(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate data against schema.

        Parameters:
        -----------
        X : np.ndarray
            Data to validate (n_samples, n_features).
        feature_names : list of str, optional
            Feature names.

        Returns:
        --------
        is_valid : bool
            True if data passes all validations.
        errors : list of str
            List of validation errors.
        """
        self.validation_errors = []

        if self.schema is None:
            self.validation_errors.append("No schema defined")
            return False, self.validation_errors

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Check number of features
        expected_features = len(self.schema)
        actual_features = X.shape[1]

        if expected_features != actual_features:
            self.validation_errors.append(
                f"Expected {expected_features} features, got {actual_features}"
            )
            return False, self.validation_errors

        # Validate each feature
        for i, feature_name in enumerate(feature_names):
            if feature_name not in self.schema:
                self.validation_errors.append(f"Unknown feature: {feature_name}")
                continue

            feature_schema = self.schema[feature_name]
            feature_data = X[:, i]

            self._validate_feature(feature_name, feature_data, feature_schema)

        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors

    def _validate_feature(
        self,
        feature_name: str,
        data: np.ndarray,
        schema: Dict
    ):
        """Validate a single feature."""

        # Check nullable
        if not schema.get('nullable', False):
            if np.any(np.isnan(data)):
                self.validation_errors.append(
                    f"{feature_name}: Contains null values but nullable=False"
                )

        # Non-null data for remaining checks
        valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            return

        # Type checking
        feature_type = schema.get('type')

        if feature_type == 'numeric':
            # Range checks
            if 'min' in schema:
                if np.any(valid_data < schema['min']):
                    self.validation_errors.append(
                        f"{feature_name}: Contains values below minimum {schema['min']}"
                    )

            if 'max' in schema:
                if np.any(valid_data > schema['max']):
                    self.validation_errors.append(
                        f"{feature_name}: Contains values above maximum {schema['max']}"
                    )

        elif feature_type == 'categorical':
            # Category checks
            if 'categories' in schema:
                valid_categories = set(schema['categories'])
                actual_categories = set(valid_data)

                invalid_categories = actual_categories - valid_categories
                if invalid_categories:
                    self.validation_errors.append(
                        f"{feature_name}: Contains invalid categories: {invalid_categories}"
                    )

        elif feature_type == 'binary':
            # Binary checks
            unique_values = np.unique(valid_data)
            if len(unique_values) > 2:
                self.validation_errors.append(
                    f"{feature_name}: Expected binary feature, got {len(unique_values)} unique values"
                )

        # Custom validator
        if 'validator' in schema:
            custom_validator = schema['validator']
            try:
                is_valid, error_msg = custom_validator(valid_data)
                if not is_valid:
                    self.validation_errors.append(f"{feature_name}: {error_msg}")
            except Exception as e:
                self.validation_errors.append(
                    f"{feature_name}: Custom validator failed: {str(e)}"
                )


# ============================================================================
# TYPE VALIDATION
# ============================================================================

class TypeValidator:
    """
    Validate data types.
    """

    @staticmethod
    def is_numeric(X: np.ndarray) -> np.ndarray:
        """
        Check if features are numeric.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        is_numeric : np.ndarray
            Boolean array indicating numeric features.
        """
        is_numeric = np.zeros(X.shape[1], dtype=bool)

        for i in range(X.shape[1]):
            try:
                # Try to convert to float
                _ = X[:, i].astype(float)
                is_numeric[i] = True
            except (ValueError, TypeError):
                is_numeric[i] = False

        return is_numeric

    @staticmethod
    def is_integer(X: np.ndarray) -> np.ndarray:
        """
        Check if features are integers.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        is_integer : np.ndarray
            Boolean array indicating integer features.
        """
        is_integer = np.zeros(X.shape[1], dtype=bool)

        for i in range(X.shape[1]):
            col = X[:, i]
            # Remove NaN
            valid_col = col[~np.isnan(col)]

            if len(valid_col) == 0:
                continue

            # Check if all values are integers
            is_integer[i] = np.all(valid_col == valid_col.astype(int))

        return is_integer

    @staticmethod
    def is_categorical(
        X: np.ndarray,
        threshold: int = 10
    ) -> np.ndarray:
        """
        Check if features are categorical (low cardinality).

        Parameters:
        -----------
        X : np.ndarray
            Data.
        threshold : int
            Maximum number of unique values for categorical.

        Returns:
        --------
        is_categorical : np.ndarray
            Boolean array indicating categorical features.
        """
        is_categorical = np.zeros(X.shape[1], dtype=bool)

        for i in range(X.shape[1]):
            col = X[:, i]
            valid_col = col[~np.isnan(col)]

            n_unique = len(np.unique(valid_col))
            is_categorical[i] = n_unique <= threshold

        return is_categorical


# ============================================================================
# RANGE VALIDATION
# ============================================================================

class RangeValidator:
    """
    Validate value ranges.
    """

    @staticmethod
    def check_range(
        X: np.ndarray,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if all values are within specified range.

        Parameters:
        -----------
        X : np.ndarray
            Data.
        min_val : float, optional
            Minimum allowed value.
        max_val : float, optional
            Maximum allowed value.

        Returns:
        --------
        is_valid : bool
            True if all values are within range.
        errors : list of str
            List of validation errors.
        """
        errors = []

        for i in range(X.shape[1]):
            col = X[:, i]
            valid_col = col[~np.isnan(col)]

            if len(valid_col) == 0:
                continue

            if min_val is not None and np.any(valid_col < min_val):
                errors.append(
                    f"Feature {i}: Contains values below minimum {min_val}"
                )

            if max_val is not None and np.any(valid_col > max_val):
                errors.append(
                    f"Feature {i}: Contains values above maximum {max_val}"
                )

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def check_positive(X: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Check if all values are positive.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        is_valid : bool
        errors : list of str
        """
        return RangeValidator.check_range(X, min_val=0)

    @staticmethod
    def check_normalized(
        X: np.ndarray,
        tolerance: float = 1e-6
    ) -> Tuple[bool, List[str]]:
        """
        Check if values are normalized to [0, 1].

        Parameters:
        -----------
        X : np.ndarray
            Data.
        tolerance : float
            Tolerance for boundary checking.

        Returns:
        --------
        is_valid : bool
        errors : list of str
        """
        return RangeValidator.check_range(
            X,
            min_val=-tolerance,
            max_val=1 + tolerance
        )


# ============================================================================
# CONSISTENCY CHECKS
# ============================================================================

class ConsistencyValidator:
    """
    Check data consistency.
    """

    @staticmethod
    def check_duplicates(X: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Check for duplicate rows.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        n_duplicates : int
            Number of duplicate rows.
        duplicate_indices : np.ndarray
            Indices of duplicate rows.
        """
        # Find duplicate rows
        unique_rows, indices, counts = np.unique(
            X,
            axis=0,
            return_index=True,
            return_counts=True
        )

        duplicate_mask = counts > 1
        n_duplicates = np.sum(counts[duplicate_mask] - 1)

        # Find indices of duplicate rows
        duplicate_indices = []
        for i, count in enumerate(counts):
            if count > 1:
                # Find all occurrences
                row = unique_rows[i]
                matches = np.where(np.all(X == row, axis=1))[0]
                duplicate_indices.extend(matches[1:].tolist())  # Skip first occurrence

        return n_duplicates, np.array(duplicate_indices)

    @staticmethod
    def check_constant_features(
        X: np.ndarray,
        threshold: float = 0.0
    ) -> np.ndarray:
        """
        Check for constant or near-constant features.

        Parameters:
        -----------
        X : np.ndarray
            Data.
        threshold : float
            Variance threshold. Features with variance <= threshold are flagged.

        Returns:
        --------
        constant_features : np.ndarray
            Indices of constant features.
        """
        variances = np.var(X, axis=0)
        constant_features = np.where(variances <= threshold)[0]

        return constant_features

    @staticmethod
    def check_correlation(
        X: np.ndarray,
        threshold: float = 0.95
    ) -> List[Tuple[int, int, float]]:
        """
        Check for highly correlated feature pairs.

        Parameters:
        -----------
        X : np.ndarray
            Data.
        threshold : float
            Correlation threshold.

        Returns:
        --------
        high_correlations : list of tuples
            List of (feature_i, feature_j, correlation) tuples.
        """
        n_features = X.shape[1]

        # Compute correlation matrix
        corr_matrix = np.corrcoef(X, rowvar=False)

        high_correlations = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = corr_matrix[i, j]
                if abs(corr) >= threshold:
                    high_correlations.append((i, j, corr))

        return high_correlations


# ============================================================================
# DATA QUALITY METRICS
# ============================================================================

class DataQualityMetrics:
    """
    Compute data quality metrics.
    """

    @staticmethod
    def completeness(X: np.ndarray) -> Dict[str, float]:
        """
        Measure data completeness (non-missing rate).

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        metrics : dict
            Completeness metrics.
        """
        n_total = X.size
        n_missing = np.isnan(X).sum()
        n_complete = n_total - n_missing

        completeness_overall = n_complete / n_total

        # Per-feature completeness
        completeness_per_feature = []
        for i in range(X.shape[1]):
            col_complete = np.sum(~np.isnan(X[:, i])) / X.shape[0]
            completeness_per_feature.append(col_complete)

        return {
            'overall': completeness_overall,
            'per_feature': completeness_per_feature,
            'n_missing': int(n_missing),
            'n_complete': int(n_complete)
        }

    @staticmethod
    def uniqueness(X: np.ndarray) -> Dict[str, float]:
        """
        Measure data uniqueness (unique row ratio).

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        metrics : dict
            Uniqueness metrics.
        """
        n_total = X.shape[0]
        n_unique = len(np.unique(X, axis=0))

        uniqueness = n_unique / n_total

        return {
            'uniqueness': uniqueness,
            'n_total': int(n_total),
            'n_unique': int(n_unique),
            'n_duplicates': int(n_total - n_unique)
        }

    @staticmethod
    def validity(
        X: np.ndarray,
        validator: DataValidator
    ) -> Dict[str, Any]:
        """
        Measure data validity based on validator.

        Parameters:
        -----------
        X : np.ndarray
            Data.
        validator : DataValidator
            Validator with schema.

        Returns:
        --------
        metrics : dict
            Validity metrics.
        """
        is_valid, errors = validator.validate(X)

        return {
            'is_valid': is_valid,
            'n_errors': len(errors),
            'errors': errors
        }

    @staticmethod
    def consistency(X: np.ndarray) -> Dict[str, Any]:
        """
        Measure data consistency.

        Parameters:
        -----------
        X : np.ndarray
            Data.

        Returns:
        --------
        metrics : dict
            Consistency metrics.
        """
        # Duplicates
        n_duplicates, _ = ConsistencyValidator.check_duplicates(X)

        # Constant features
        constant_features = ConsistencyValidator.check_constant_features(X)

        # High correlations
        high_correlations = ConsistencyValidator.check_correlation(X)

        return {
            'n_duplicates': int(n_duplicates),
            'n_constant_features': len(constant_features),
            'constant_features': constant_features.tolist(),
            'n_high_correlations': len(high_correlations),
            'high_correlations': high_correlations
        }


# ============================================================================
# VALIDATION REPORT
# ============================================================================

class ValidationReport:
    """
    Generate comprehensive validation report.
    """

    @staticmethod
    def generate(
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        validator: Optional[DataValidator] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.

        Parameters:
        -----------
        X : np.ndarray
            Data.
        feature_names : list of str, optional
            Feature names.
        validator : DataValidator, optional
            Validator with schema.

        Returns:
        --------
        report : dict
            Comprehensive validation report.
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        report = {
            'summary': {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'feature_names': feature_names
            },
            'completeness': DataQualityMetrics.completeness(X),
            'uniqueness': DataQualityMetrics.uniqueness(X),
            'consistency': DataQualityMetrics.consistency(X)
        }

        # Type information
        type_validator = TypeValidator()
        report['types'] = {
            'numeric': type_validator.is_numeric(X).tolist(),
            'integer': type_validator.is_integer(X).tolist(),
            'categorical': type_validator.is_categorical(X).tolist()
        }

        # Schema validation
        if validator is not None:
            report['validity'] = DataQualityMetrics.validity(X, validator)

        return report

    @staticmethod
    def print_report(report: Dict[str, Any]):
        """
        Print validation report.

        Parameters:
        -----------
        report : dict
            Validation report.
        """
        print("=" * 70)
        print("DATA VALIDATION REPORT")
        print("=" * 70)

        # Summary
        print("\nSummary:")
        print(f"  Samples:  {report['summary']['n_samples']}")
        print(f"  Features: {report['summary']['n_features']}")

        # Completeness
        print("\nCompleteness:")
        print(f"  Overall: {report['completeness']['overall']:.2%}")
        print(f"  Missing values: {report['completeness']['n_missing']}")

        # Uniqueness
        print("\nUniqueness:")
        print(f"  Unique rows: {report['uniqueness']['uniqueness']:.2%}")
        print(f"  Duplicate rows: {report['uniqueness']['n_duplicates']}")

        # Consistency
        print("\nConsistency:")
        print(f"  Constant features: {report['consistency']['n_constant_features']}")
        if report['consistency']['n_constant_features'] > 0:
            print(f"    Indices: {report['consistency']['constant_features']}")
        print(f"  High correlations: {report['consistency']['n_high_correlations']}")

        # Validity
        if 'validity' in report:
            print("\nValidity:")
            print(f"  Valid: {report['validity']['is_valid']}")
            if report['validity']['n_errors'] > 0:
                print(f"  Errors: {report['validity']['n_errors']}")
                for error in report['validity']['errors']:
                    print(f"    - {error}")

        print("\n" + "=" * 70)


# ============================================================================
# EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DATA VALIDATION EXAMPLES")
    print("=" * 70)

    # Create sample dataset
    np.random.seed(42)
    X = np.random.randn(100, 5)

    # Introduce some data quality issues
    # Missing values
    X[np.random.rand(100, 5) < 0.1] = np.nan

    # Duplicates
    X[10:15] = X[0:5]

    # Constant feature
    X[:, 3] = 5.0

    # Highly correlated features
    X[:, 4] = X[:, 0] * 0.95 + np.random.randn(100) * 0.1

    feature_names = ['age', 'income', 'score', 'constant_feature', 'correlated_feature']

    # Example 1: Schema Validation
    print("\n1. Schema Validation")
    print("-" * 70)

    schema = {
        'age': {
            'type': 'numeric',
            'min': 0,
            'max': 120,
            'nullable': False
        },
        'income': {
            'type': 'numeric',
            'min': 0,
            'nullable': False
        },
        'score': {
            'type': 'numeric',
            'min': -3,
            'max': 3,
            'nullable': True
        },
        'constant_feature': {
            'type': 'numeric',
            'nullable': False
        },
        'correlated_feature': {
            'type': 'numeric',
            'nullable': False
        }
    }

    validator = DataValidator(schema)
    is_valid, errors = validator.validate(X, feature_names)

    print(f"Validation result: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")

    # Example 2: Type Validation
    print("\n2. Type Validation")
    print("-" * 70)

    type_validator = TypeValidator()
    is_numeric = type_validator.is_numeric(X)
    is_integer = type_validator.is_integer(X)
    is_categorical = type_validator.is_categorical(X)

    print(f"Numeric features: {[feature_names[i] for i, v in enumerate(is_numeric) if v]}")
    print(f"Integer features: {[feature_names[i] for i, v in enumerate(is_integer) if v]}")
    print(f"Categorical features: {[feature_names[i] for i, v in enumerate(is_categorical) if v]}")

    # Example 3: Range Validation
    print("\n3. Range Validation")
    print("-" * 70)

    range_validator = RangeValidator()
    is_valid_range, range_errors = range_validator.check_range(X, min_val=-5, max_val=10)

    print(f"Range validation: {'PASS' if is_valid_range else 'FAIL'}")
    if range_errors:
        for error in range_errors:
            print(f"  - {error}")

    # Example 4: Consistency Checks
    print("\n4. Consistency Checks")
    print("-" * 70)

    consistency_validator = ConsistencyValidator()

    # Duplicates
    n_duplicates, duplicate_indices = consistency_validator.check_duplicates(X)
    print(f"Duplicates: {n_duplicates} rows")

    # Constant features
    constant_features = consistency_validator.check_constant_features(X)
    print(f"Constant features: {[feature_names[i] for i in constant_features]}")

    # Correlations
    high_correlations = consistency_validator.check_correlation(X, threshold=0.9)
    print(f"High correlations ({len(high_correlations)}):")
    for i, j, corr in high_correlations:
        print(f"  {feature_names[i]} <-> {feature_names[j]}: {corr:.3f}")

    # Example 5: Data Quality Metrics
    print("\n5. Data Quality Metrics")
    print("-" * 70)

    completeness = DataQualityMetrics.completeness(X)
    uniqueness = DataQualityMetrics.uniqueness(X)

    print(f"Completeness: {completeness['overall']:.2%}")
    print(f"Uniqueness: {uniqueness['uniqueness']:.2%}")

    # Example 6: Full Validation Report
    print("\n6. Full Validation Report")
    print("-" * 70)

    report = ValidationReport.generate(X, feature_names, validator)
    ValidationReport.print_report(report)

    print("\n" + "=" * 70)
    print("All validation examples completed!")
    print("=" * 70)
