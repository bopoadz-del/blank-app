"""
Ethical Layer Models - Credibility-Based Autonomy Framework.
Implements trust hierarchy, source credibility tracking, and validation pipeline.
"""
from datetime import datetime
from typing import Optional, Dict, List
from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, DateTime, Text, Enum as SQLEnum, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
import enum

from .database import Base


class SourceType(str, enum.Enum):
    """Type of knowledge source."""
    ISO_STANDARD = "iso_standard"
    ANSI_STANDARD = "ansi_standard"
    BS_STANDARD = "bs_standard"
    ASHRAE_STANDARD = "ashrae_standard"
    GOVERNMENT_REGULATION = "government_regulation"
    PEER_REVIEWED_PAPER = "peer_reviewed_paper"
    INDUSTRY_HANDBOOK = "industry_handbook"
    CONSULTANT_REPORT = "consultant_report"
    MANUFACTURER_SPEC = "manufacturer_spec"
    HISTORICAL_DATA = "historical_data"
    ML_DISCOVERED = "ml_discovered"
    AI_GENERATED = "ai_generated"
    THIRD_PARTY_CALC = "third_party_calc"
    NOVEL_APPROACH = "novel_approach"


class CredibilityTier(int, enum.Enum):
    """
    Credibility-based trust tier for autonomy decisions.
    Maps to FormulaTier but with ethical semantics.
    """
    TIER_1_FULLY_AUTOMATED = 1      # 95-100% trust, auto-deploy
    TIER_2_SUPERVISED = 2            # 70-94% trust, manager approval
    TIER_3_EXPERIMENTAL = 3          # 40-69% trust, expert committee
    TIER_4_RESEARCH = 4              # <40% trust, sandbox only


class ValidationStage(str, enum.Enum):
    """5-stage validation pipeline stages."""
    SYNTACTIC = "syntactic"           # Stage 1: Syntax check
    DIMENSIONAL = "dimensional"       # Stage 2: Unit analysis
    PHYSICAL = "physical"             # Stage 3: Physics constraints
    EMPIRICAL = "empirical"           # Stage 4: Historical validation
    SAFETY = "safety"                 # Stage 5: Operational safety


class ValidationStatus(str, enum.Enum):
    """Validation result status."""
    APPROVED = "approved"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"
    ELEVATED_REVIEW = "elevated_review"
    SANDBOX_ONLY = "sandbox_only"


class KnowledgeSource(Base):
    """
    Tracks credibility of knowledge sources.
    Sources gain/lose trust based on real-world outcomes.
    """
    __tablename__ = "knowledge_sources"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(String(255), unique=True, nullable=False, index=True)
    source_name = Column(String(500), nullable=False)
    source_type = Column(SQLEnum(SourceType), nullable=False, index=True)

    # Credibility metrics
    credibility_score = Column(Float, default=0.5, nullable=False)  # 0.0 - 1.0
    credibility_tier = Column(SQLEnum(CredibilityTier), default=CredibilityTier.TIER_4_RESEARCH, nullable=False)

    # Usage statistics
    usage_count = Column(Integer, default=0, nullable=False)
    success_count = Column(Integer, default=0, nullable=False)
    failure_count = Column(Integer, default=0, nullable=False)

    # Context tracking
    successful_domains = Column(JSON, default=list, nullable=False)  # List of domains where source worked
    failure_contexts = Column(JSON, default=list, nullable=False)     # List of failure scenarios

    # Source metadata
    publication_date = Column(DateTime, nullable=True)
    version = Column(String(100), nullable=True)
    issuing_authority = Column(String(500), nullable=True)
    geographic_scope = Column(JSON, nullable=True)  # ["US", "UK", "Global"]
    domain_tags = Column(JSON, default=list, nullable=False)  # ["structural", "MEP", "cost"]

    # Verification
    is_verified = Column(Boolean, default=False, nullable=False)
    verified_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    verified_at = Column(DateTime, nullable=True)

    # Last credibility update
    last_outcome_date = Column(DateTime, nullable=True)
    last_credibility_update = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Metadata
    notes = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    validation_results = relationship("FormulaValidationResult", back_populates="knowledge_source")
    verified_by = relationship("User", foreign_keys=[verified_by_user_id])

    # Indexes
    __table_args__ = (
        Index('idx_source_credibility', 'credibility_score', 'source_type'),
        Index('idx_source_tier', 'credibility_tier'),
    )

    def update_credibility(self, outcome_success: bool, domain: str, context: Dict):
        """Update credibility score based on outcome."""
        if outcome_success:
            self.success_count += 1
            if domain not in self.successful_domains:
                self.successful_domains.append(domain)
        else:
            self.failure_count += 1
            self.failure_contexts.append({
                'domain': domain,
                'context': context,
                'timestamp': datetime.utcnow().isoformat()
            })

        self.usage_count += 1

        # Calculate new credibility score
        if self.usage_count > 0:
            base_score = self.success_count / self.usage_count

            # Apply domain-specific weights
            if domain == "safety_critical":
                base_score *= 0.9  # Conservative for safety
            elif domain == "cost_estimation":
                base_score *= 1.1  # Allow more flexibility

            self.credibility_score = max(0.0, min(1.0, base_score))

        # Update tier based on score and usage
        self._update_tier()

        self.last_outcome_date = datetime.utcnow()
        self.last_credibility_update = datetime.utcnow()

    def _update_tier(self):
        """Update credibility tier based on score and usage."""
        score = self.credibility_score

        if score >= 0.95 and self.usage_count >= 100:
            self.credibility_tier = CredibilityTier.TIER_1_FULLY_AUTOMATED
        elif score >= 0.70:
            self.credibility_tier = CredibilityTier.TIER_2_SUPERVISED
        elif score >= 0.40:
            self.credibility_tier = CredibilityTier.TIER_3_EXPERIMENTAL
        else:
            self.credibility_tier = CredibilityTier.TIER_4_RESEARCH


class FormulaValidationResult(Base):
    """
    Tracks results of 5-stage validation pipeline.
    Each formula goes through syntactic, dimensional, physical, empirical, and safety checks.
    """
    __tablename__ = "formula_validation_results"

    id = Column(Integer, primary_key=True, index=True)
    formula_id = Column(Integer, ForeignKey("formulas.id", ondelete="CASCADE"), nullable=False, index=True)
    validation_run_id = Column(String(255), unique=True, nullable=False, index=True)

    # Overall validation
    final_status = Column(SQLEnum(ValidationStatus), nullable=False, index=True)
    assigned_tier = Column(Integer, nullable=True)  # Final tier assignment

    # Stage 1: Syntactic Validation
    syntactic_passed = Column(Boolean, default=False, nullable=False)
    syntactic_errors = Column(JSON, nullable=True)

    # Stage 2: Dimensional Analysis
    dimensional_passed = Column(Boolean, default=False, nullable=False)
    dimensional_errors = Column(JSON, nullable=True)
    unit_consistency = Column(Float, nullable=True)  # 0.0 - 1.0

    # Stage 3: Physical Constraints
    physical_passed = Column(Boolean, default=False, nullable=False)
    physical_violations = Column(JSON, nullable=True)
    physics_score = Column(Float, nullable=True)  # 0.0 - 1.0

    # Stage 4: Empirical Validation
    empirical_passed = Column(Boolean, default=False, nullable=False)
    historical_accuracy = Column(Float, nullable=True)  # 0.0 - 1.0
    historical_test_count = Column(Integer, default=0, nullable=False)

    # Stage 5: Operational Safety
    safety_passed = Column(Boolean, default=False, nullable=False)
    safety_score = Column(Float, nullable=True)  # 0.0 - 1.0
    safety_issues = Column(JSON, nullable=True)

    # Source credibility
    knowledge_source_id = Column(Integer, ForeignKey("knowledge_sources.id"), nullable=True)
    source_credibility_at_validation = Column(Float, nullable=True)

    # Context
    validation_context = Column(JSON, nullable=True)  # Domain, environment, etc.
    validator_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Metadata
    validation_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    validation_duration_ms = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)

    # Relationships
    formula = relationship("Formula", foreign_keys=[formula_id])
    knowledge_source = relationship("KnowledgeSource", back_populates="validation_results")
    validator = relationship("User", foreign_keys=[validator_user_id])

    # Indexes
    __table_args__ = (
        Index('idx_validation_formula_status', 'formula_id', 'final_status'),
        Index('idx_validation_timestamp', 'validation_timestamp'),
    )


class EthicalOverride(Base):
    """
    Context-aware ethical overrides.
    Adjusts credibility based on environmental, temporal, or contractor factors.
    """
    __tablename__ = "ethical_overrides"

    id = Column(Integer, primary_key=True, index=True)
    override_id = Column(String(255), unique=True, nullable=False, index=True)

    # Override type
    override_category = Column(String(100), nullable=False, index=True)  # environmental, temporal, contractor
    override_name = Column(String(255), nullable=False)

    # Conditions
    trigger_conditions = Column(JSON, nullable=False)  # e.g., {"weather": "extreme", "seismic_zone": true}

    # Actions
    credibility_adjustment = Column(Float, default=0.0, nullable=False)  # -0.2 to +0.2
    safety_margin_multiplier = Column(Float, default=1.0, nullable=False)  # e.g., 1.2 for 20% increase
    required_min_tier = Column(Integer, nullable=True)  # Force minimum tier
    additional_checks = Column(JSON, nullable=True)  # Extra validation requirements

    # Scope
    applicable_domains = Column(JSON, default=list, nullable=False)  # ["structural", "MEP"]
    applicable_deployments = Column(JSON, default=list, nullable=False)  # ["diriyah", "all"]

    # Activation
    is_active = Column(Boolean, default=True, nullable=False)
    priority = Column(Integer, default=100, nullable=False)  # Higher = applied first

    # Metadata
    description = Column(Text, nullable=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    created_by = relationship("User", foreign_keys=[created_by_user_id])

    # Indexes
    __table_args__ = (
        Index('idx_override_category_active', 'override_category', 'is_active'),
    )


class EthicalConfiguration(Base):
    """
    Deployment-specific ethical configuration.
    Each deployment (Diriyah, MSSDPPG, etc.) can have different ethical settings.
    """
    __tablename__ = "ethical_configurations"

    id = Column(Integer, primary_key=True, index=True)
    deployment_name = Column(String(255), unique=True, nullable=False, index=True)

    # Default settings
    default_tier = Column(Integer, default=2, nullable=False)  # Conservative start
    auto_promotion_enabled = Column(Boolean, default=True, nullable=False)
    safety_margin = Column(Float, default=1.2, nullable=False)  # 20% safety factor

    # Domain-specific minimum tiers
    domain_min_tiers = Column(JSON, default=dict, nullable=False)
    # e.g., {"structural": 1, "scheduling": 2, "cost": 2, "logistics": 3}

    # Red lines (never override)
    red_lines = Column(JSON, default=list, nullable=False)
    # e.g., ["human_safety_tier_1_only", "environmental_compliance_gov_only"]

    # Transparency settings
    audit_frequency = Column(String(50), default="weekly", nullable=False)  # daily, weekly, monthly
    require_two_factor_override = Column(Boolean, default=True, nullable=False)
    log_retention_days = Column(Integer, default=2555, nullable=False)  # 7 years = 2555 days

    # Privacy & security
    cross_deployment_learning = Column(Boolean, default=False, nullable=False)
    encrypted_formulas = Column(Boolean, default=True, nullable=False)
    allow_external_api = Column(Boolean, default=False, nullable=False)

    # Metadata
    configured_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    configured_by = relationship("User", foreign_keys=[configured_by_user_id])


class EthicalAuditLog(Base):
    """
    Specialized audit log for ethical decisions.
    Tracks why decisions were made and which ethical rules were applied.
    """
    __tablename__ = "ethical_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    audit_id = Column(String(255), unique=True, nullable=False, index=True)

    # Decision tracking
    decision_type = Column(String(100), nullable=False, index=True)  # formula_approval, tier_assignment, override_applied
    formula_id = Column(Integer, ForeignKey("formulas.id"), nullable=True)
    execution_id = Column(Integer, ForeignKey("formula_executions.id"), nullable=True)

    # Ethical reasoning
    credibility_tier_assigned = Column(Integer, nullable=True)
    source_credibility_scores = Column(JSON, nullable=True)  # All sources with scores
    overrides_applied = Column(JSON, default=list, nullable=False)  # Which overrides triggered
    red_lines_checked = Column(JSON, default=list, nullable=False)  # Which red lines were validated

    # Human involvement
    required_human_approval = Column(Boolean, default=False, nullable=False)
    approved_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    approval_timestamp = Column(DateTime, nullable=True)

    # Context
    deployment_name = Column(String(255), nullable=True)
    domain = Column(String(100), nullable=True)
    context_factors = Column(JSON, nullable=True)  # Environmental, temporal, contractor

    # Transparency
    decision_explanation = Column(Text, nullable=False)  # Why this decision was made
    sources_cited = Column(JSON, default=list, nullable=False)  # Which sources were used

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    formula = relationship("Formula", foreign_keys=[formula_id])
    approved_by = relationship("User", foreign_keys=[approved_by_user_id])

    # Indexes
    __table_args__ = (
        Index('idx_ethical_audit_decision', 'decision_type', 'created_at'),
        Index('idx_ethical_audit_formula', 'formula_id'),
    )
