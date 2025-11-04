"""
Corrections and Audit Trail Models.

This is the foundation of verifiable decisions - every correction made by operators
is stored as an immutable record linked to the original execution.
"""
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.models.database import Base


class CorrectionStatus(enum.Enum):
    """Status of a correction."""
    PENDING = "pending"  # Awaiting review
    APPROVED = "approved"  # Approved by admin
    REJECTED = "rejected"  # Rejected by admin
    APPLIED = "applied"  # Applied to retraining


class CorrectionType(enum.Enum):
    """Type of correction."""
    VALUE_CORRECTION = "value_correction"  # Correcting output values
    CLASSIFICATION_CORRECTION = "classification_correction"  # Correcting classifications
    DETECTION_CORRECTION = "detection_correction"  # Correcting object detections
    FORMULA_CORRECTION = "formula_correction"  # Correcting formula logic
    PARAMETER_CORRECTION = "parameter_correction"  # Correcting input parameters


class Correction(Base):
    """
    Immutable record of human corrections to AI outputs.

    This is the core of the verifiable audit trail. Every time an operator
    corrects an AI decision, it's logged here with full context.
    """
    __tablename__ = "corrections"

    id = Column(Integer, primary_key=True, index=True)

    # Link to original execution
    execution_id = Column(Integer, ForeignKey("formula_executions.id"), nullable=False, index=True)

    # Who made the correction
    corrected_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Correction details
    correction_type = Column(SQLEnum(CorrectionType), nullable=False)
    status = Column(SQLEnum(CorrectionStatus), default=CorrectionStatus.PENDING, nullable=False)

    # Original AI output (immutable)
    original_output = Column(JSON, nullable=False)

    # Corrected output by human (immutable)
    corrected_output = Column(JSON, nullable=False)

    # Explanation for the correction
    correction_reason = Column(Text, nullable=True)

    # Confidence level from operator (0-100)
    operator_confidence = Column(Integer, default=100)

    # Review information
    reviewed_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    review_notes = Column(Text, nullable=True)

    # Training metadata
    used_in_training = Column(Boolean, default=False)
    training_run_id = Column(String, nullable=True)  # MLflow run ID

    # Timestamps (immutable)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Metadata
    metadata = Column(JSON, default=dict)

    # Relationships
    execution = relationship("FormulaExecution", back_populates="corrections")
    corrected_by = relationship("User", foreign_keys=[corrected_by_user_id], back_populates="corrections_made")
    reviewed_by = relationship("User", foreign_keys=[reviewed_by_user_id], back_populates="corrections_reviewed")


class AuditLog(Base):
    """
    Comprehensive audit log for all system actions.

    Immutable record of every action in the system for compliance and traceability.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)

    # Who performed the action
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Nullable for system actions

    # What action was performed
    action = Column(String(100), nullable=False, index=True)  # e.g., "formula.execute", "correction.create"
    entity_type = Column(String(50), nullable=False)  # e.g., "formula", "correction", "user"
    entity_id = Column(Integer, nullable=True)

    # Action details
    description = Column(Text, nullable=False)

    # Before and after state
    before_state = Column(JSON, nullable=True)
    after_state = Column(JSON, nullable=True)

    # Request context
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(255), nullable=True)
    request_id = Column(String(50), nullable=True)

    # Result
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)

    # Timestamp (immutable)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Metadata
    metadata = Column(JSON, default=dict)

    # Relationships
    user = relationship("User", back_populates="audit_logs")


class FormulaCertification(Base):
    """
    Record of formula certification in the Tier system.

    Tracks the promotion of formulas from Tier 4 (experimental) to Tier 1 (certified).
    """
    __tablename__ = "formula_certifications"

    id = Column(Integer, primary_key=True, index=True)

    # Formula being certified
    formula_id = Column(Integer, ForeignKey("formulas.id"), nullable=False)

    # Version information
    from_version = Column(String(50), nullable=False)  # e.g., "v2.5-exp"
    to_version = Column(String(50), nullable=False)    # e.g., "v2.5.1-prod"

    # Tier progression
    from_tier = Column(Integer, nullable=False)  # 4 = experimental
    to_tier = Column(Integer, nullable=False)    # 1 = certified

    # Certification details
    certified_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    certification_notes = Column(Text, nullable=True)

    # Performance metrics at certification
    test_accuracy = Column(JSON, nullable=True)  # Test results
    validation_metrics = Column(JSON, nullable=True)

    # Review information
    review_period_start = Column(DateTime, nullable=True)
    review_period_end = Column(DateTime, nullable=True)
    executions_reviewed = Column(Integer, default=0)
    corrections_count = Column(Integer, default=0)

    # Timestamp
    certified_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Immutability lock
    is_locked = Column(Boolean, default=True)  # Once certified, cannot be changed

    # Relationships
    formula = relationship("Formula", back_populates="certifications")
    certified_by = relationship("User", back_populates="certifications_made")


# Update User model relationships (add to User model)
# corrections_made = relationship("Correction", foreign_keys="Correction.corrected_by_user_id", back_populates="corrected_by")
# corrections_reviewed = relationship("Correction", foreign_keys="Correction.reviewed_by_user_id", back_populates="reviewed_by")
# audit_logs = relationship("AuditLog", back_populates="user")
# certifications_made = relationship("FormulaCertification", back_populates="certified_by")

# Update FormulaExecution model
# corrections = relationship("Correction", back_populates="execution")

# Update Formula model
# certifications = relationship("FormulaCertification", back_populates="formula")
