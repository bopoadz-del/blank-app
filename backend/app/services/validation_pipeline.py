"""
5-Stage Validation Pipeline Service.
Implements syntactic, dimensional, physical, empirical, and safety validation.
"""
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func
import uuid
import re
import logging

from ..models.ethical_layer import (
    FormulaValidationResult,
    ValidationStatus,
    ValidationStage,
    KnowledgeSource,
    CredibilityTier
)
from ..models.database import Formula, FormulaExecution
from ..models.corrections import Correction, CorrectionStatus

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """
    Implements the 5-stage ethical validation pipeline for formulas.
    """

    def __init__(self, db: Session):
        self.db = db

    def validate_formula(
        self,
        formula: Formula,
        context: Optional[Dict[str, Any]] = None
    ) -> FormulaValidationResult:
        """
        Run complete 5-stage validation pipeline.

        Args:
            formula: Formula to validate
            context: Optional context (domain, environment, etc.)

        Returns:
            FormulaValidationResult with all stage results
        """
        validation_run_id = f"val_{uuid.uuid4().hex[:12]}"
        context = context or {}

        logger.info(f"Starting validation for formula {formula.id}: {validation_run_id}")

        result = FormulaValidationResult(
            formula_id=formula.id,
            validation_run_id=validation_run_id,
            validation_context=context,
            validation_timestamp=datetime.utcnow()
        )

        start_time = datetime.utcnow()

        # Stage 1: Syntactic Validation
        result.syntactic_passed, result.syntactic_errors = self._stage1_syntactic(formula)
        if not result.syntactic_passed:
            result.final_status = ValidationStatus.REJECTED
            result.validation_duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.db.add(result)
            self.db.commit()
            return result

        # Stage 2: Dimensional Analysis
        result.dimensional_passed, result.dimensional_errors, result.unit_consistency = \
            self._stage2_dimensional(formula, context)
        if not result.dimensional_passed:
            result.final_status = ValidationStatus.REJECTED
            result.validation_duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.db.add(result)
            self.db.commit()
            return result

        # Stage 3: Physical Constraints
        result.physical_passed, result.physical_violations, result.physics_score = \
            self._stage3_physical(formula, context)
        if not result.physical_passed:
            result.final_status = ValidationStatus.REJECTED
            result.validation_duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.db.add(result)
            self.db.commit()
            return result

        # Stage 4: Empirical Validation
        result.empirical_passed, result.historical_accuracy, result.historical_test_count = \
            self._stage4_empirical(formula, context)
        if result.historical_accuracy and result.historical_accuracy < 0.7:
            result.final_status = ValidationStatus.MANUAL_REVIEW
            result.validation_duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.db.add(result)
            self.db.commit()
            return result

        # Stage 5: Operational Safety
        result.safety_passed, result.safety_score, result.safety_issues = \
            self._stage5_safety(formula, context)
        if result.safety_score and result.safety_score < 0.8:
            result.final_status = ValidationStatus.ELEVATED_REVIEW
            result.validation_duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.db.add(result)
            self.db.commit()
            return result

        # All stages passed - assign tier
        result.final_status = ValidationStatus.APPROVED
        result.assigned_tier = self._assign_tier(formula, result)

        result.validation_duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)

        logger.info(f"Validation complete: {validation_run_id} - {result.final_status.value}")

        return result

    def _stage1_syntactic(self, formula: Formula) -> Tuple[bool, Optional[List[str]]]:
        """
        Stage 1: Syntactic Validation.
        Checks formula syntax, structure, and basic validity.
        """
        errors = []

        # Check formula_id format
        if not formula.formula_id or len(formula.formula_id) == 0:
            errors.append("Missing formula_id")

        # Check name
        if not formula.name or len(formula.name.strip()) == 0:
            errors.append("Missing formula name")

        # Check input schema
        if formula.input_schema:
            if not isinstance(formula.input_schema, dict):
                errors.append("Invalid input_schema format (must be dict/JSON)")
        else:
            errors.append("Missing input_schema")

        # Check output schema
        if formula.output_schema:
            if not isinstance(formula.output_schema, dict):
                errors.append("Invalid output_schema format (must be dict/JSON)")

        # Check for dangerous patterns (code injection, etc.)
        dangerous_patterns = [
            r'__import__',
            r'eval\s*\(',
            r'exec\s*\(',
            r'compile\s*\(',
            r'open\s*\(',
            r'file\s*\(',
        ]

        formula_text = str(formula.metadata or {})
        for pattern in dangerous_patterns:
            if re.search(pattern, formula_text, re.IGNORECASE):
                errors.append(f"Potentially dangerous pattern detected: {pattern}")

        passed = len(errors) == 0
        return passed, errors if errors else None

    def _stage2_dimensional(
        self,
        formula: Formula,
        context: Dict
    ) -> Tuple[bool, Optional[List[str]], Optional[float]]:
        """
        Stage 2: Dimensional Analysis.
        Verifies unit consistency and dimensional correctness.
        """
        errors = []
        unit_consistency = 1.0  # Perfect consistency

        # Check input schema for units
        if formula.input_schema:
            for field_name, field_spec in formula.input_schema.items():
                if isinstance(field_spec, dict):
                    if 'unit' not in field_spec and 'type' in field_spec:
                        # Numeric fields should have units
                        if field_spec['type'] in ['number', 'float', 'integer']:
                            errors.append(f"Missing unit specification for numeric field: {field_name}")
                            unit_consistency -= 0.1

        # Check output schema for units
        if formula.output_schema:
            for field_name, field_spec in formula.output_schema.items():
                if isinstance(field_spec, dict):
                    if 'unit' not in field_spec and 'type' in field_spec:
                        if field_spec['type'] in ['number', 'float', 'integer']:
                            errors.append(f"Missing unit specification for output field: {field_name}")
                            unit_consistency -= 0.1

        # TODO: Implement actual unit algebra checking (e.g., pint library)
        # For now, simplified check

        unit_consistency = max(0.0, unit_consistency)
        passed = len(errors) == 0 and unit_consistency >= 0.7

        return passed, errors if errors else None, unit_consistency

    def _stage3_physical(
        self,
        formula: Formula,
        context: Dict
    ) -> Tuple[bool, Optional[List[str]], Optional[float]]:
        """
        Stage 3: Physical Constraints.
        Checks for violations of physical laws and reasonable bounds.
        """
        violations = []
        physics_score = 1.0

        # Check metadata for physical constraints
        metadata = formula.metadata or {}

        # Check for reasonable output bounds
        if formula.output_schema:
            for field_name, field_spec in formula.output_schema.items():
                if isinstance(field_spec, dict):
                    # Check for negative values where they shouldn't exist
                    if 'min' in field_spec:
                        min_val = field_spec['min']
                        field_type = field_spec.get('type', '')

                        # Physical quantities that can't be negative
                        if any(keyword in field_name.lower() for keyword in ['mass', 'volume', 'area', 'length', 'count']):
                            if min_val < 0:
                                violations.append(f"Physical quantity {field_name} has negative minimum: {min_val}")
                                physics_score -= 0.2

        # Check for conservation laws (if applicable)
        # TODO: Implement domain-specific physics checks
        # - Energy conservation
        # - Mass balance
        # - Momentum conservation

        physics_score = max(0.0, physics_score)
        passed = len(violations) == 0 and physics_score >= 0.6

        return passed, violations if violations else None, physics_score

    def _stage4_empirical(
        self,
        formula: Formula,
        context: Dict
    ) -> Tuple[bool, Optional[float], int]:
        """
        Stage 4: Empirical Validation.
        Validates against historical execution data.
        """
        # Get historical executions for this formula
        executions = self.db.query(FormulaExecution).filter(
            FormulaExecution.formula_id == formula.id,
            FormulaExecution.status == "completed"
        ).limit(100).all()

        if not executions:
            # No historical data - can't validate empirically
            return True, None, 0

        # Calculate success rate based on corrections
        total_count = len(executions)
        corrected_count = 0

        for execution in executions:
            corrections = self.db.query(Correction).filter(
                Correction.execution_id == execution.id,
                Correction.status.in_([CorrectionStatus.APPROVED, CorrectionStatus.APPLIED])
            ).count()

            if corrections > 0:
                corrected_count += 1

        # Accuracy = executions WITHOUT corrections / total
        accuracy = (total_count - corrected_count) / total_count if total_count > 0 else 0.0

        passed = accuracy >= 0.7 if total_count >= 10 else True  # Require 70% if we have data

        return passed, accuracy, total_count

    def _stage5_safety(
        self,
        formula: Formula,
        context: Dict
    ) -> Tuple[bool, Optional[float], Optional[List[str]]]:
        """
        Stage 5: Operational Safety.
        Assesses safety for production use.
        """
        issues = []
        safety_score = 1.0

        # Check for safety-critical domain
        domain = context.get('domain', '')
        is_safety_critical = any(keyword in domain.lower() for keyword in [
            'structural', 'safety', 'life', 'critical', 'emergency'
        ])

        # Safety-critical formulas must have higher standards
        if is_safety_critical:
            # Must have knowledge source
            if not formula.metadata or 'source_id' not in formula.metadata:
                issues.append("Safety-critical formula missing verified knowledge source")
                safety_score -= 0.3

            # Must have been tested multiple times
            execution_count = self.db.query(FormulaExecution).filter(
                FormulaExecution.formula_id == formula.id
            ).count()

            if execution_count < 20:
                issues.append(f"Safety-critical formula has insufficient test executions: {execution_count}/20")
                safety_score -= 0.2

        # Check for error handling
        metadata = formula.metadata or {}
        if 'error_handling' not in metadata:
            issues.append("Missing error handling specification")
            safety_score -= 0.1

        # Check for validation bounds
        if not formula.output_schema or 'validation' not in str(formula.output_schema):
            issues.append("Missing output validation bounds")
            safety_score -= 0.1

        safety_score = max(0.0, safety_score)
        passed = safety_score >= 0.8

        return passed, safety_score, issues if issues else None

    def _assign_tier(self, formula: Formula, validation_result: FormulaValidationResult) -> int:
        """
        Assign credibility tier based on validation results.
        """
        # Default to conservative tier
        tier = 2

        # Check source credibility if available
        if validation_result.knowledge_source_id:
            source = self.db.query(KnowledgeSource).filter(
                KnowledgeSource.id == validation_result.knowledge_source_id
            ).first()

            if source:
                tier = source.credibility_tier.value
                validation_result.source_credibility_at_validation = source.credibility_score

        # Adjust based on validation scores
        if validation_result.historical_accuracy:
            if validation_result.historical_accuracy >= 0.95 and validation_result.historical_test_count >= 100:
                tier = min(tier, 1)  # Promote to Tier 1 if excellent history
            elif validation_result.historical_accuracy < 0.7:
                tier = max(tier, 3)  # Demote if poor history

        if validation_result.safety_score:
            if validation_result.safety_score < 0.8:
                tier = max(tier, 3)  # Demote if safety concerns

        return tier


class CredibilityLearning:
    """
    Updates source credibility based on real-world outcomes.
    """

    @staticmethod
    def update_source_credibility(
        db: Session,
        source_id: int,
        outcome_success: bool,
        domain: str,
        context: Dict
    ):
        """
        Update knowledge source credibility based on outcome.

        Args:
            db: Database session
            source_id: Knowledge source ID
            outcome_success: Whether the outcome was successful
            domain: Domain context (e.g., "structural", "cost_estimation")
            context: Additional context dictionary
        """
        source = db.query(KnowledgeSource).filter(
            KnowledgeSource.id == source_id
        ).first()

        if not source:
            logger.warning(f"Source {source_id} not found for credibility update")
            return

        # Update using the model's built-in method
        source.update_credibility(outcome_success, domain, context)

        db.commit()

        logger.info(f"Updated source {source.source_name} credibility: {source.credibility_score:.3f} (Tier {source.credibility_tier.value})")
