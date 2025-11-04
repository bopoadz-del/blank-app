"""
Safety Layer Models - Harm Prevention System.
Implements prohibited content detection, safety monitoring, and emergency protocols.
NON-NEGOTIABLE: Cannot be disabled or overridden.
"""
from datetime import datetime
from typing import Optional, Dict, List
from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, DateTime, Text, Enum as SQLEnum, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
import enum

from .database import Base


class ProhibitedCategory(str, enum.Enum):
    """Categories of prohibited content - CANNOT BE DISABLED."""
    CHILD_EXPLOITATION = "child_exploitation"
    ILLEGAL_DRUGS = "illegal_drugs"
    WEAPONS_MANUFACTURING = "weapons_manufacturing"
    EXPLOSIVES = "explosives"
    BIOWEAPONS = "bioweapons"
    GAMBLING_MANIPULATION = "gambling_manipulation"
    ILLEGAL_ALCOHOL = "illegal_alcohol"
    HUMAN_TRAFFICKING = "human_trafficking"
    TERRORISM = "terrorism"
    VIOLENCE_INCITEMENT = "violence_incitement"
    FRAUD = "fraud"
    IDENTITY_THEFT = "identity_theft"


class SafetyAction(str, enum.Enum):
    """Actions taken when prohibited content detected."""
    IMMEDIATE_BLOCK = "immediate_block"              # Block instantly
    BLOCK_AND_REPORT = "block_and_report"            # Block + alert authorities
    BLOCK_AND_INVESTIGATE = "block_and_investigate"  # Block + internal investigation
    RESTRICT = "restrict"                            # Limit access
    WARN = "warn"                                    # Warning only
    LOG_ONLY = "log_only"                            # Monitor only


class AlertLevel(str, enum.Enum):
    """Alert escalation levels."""
    CRITICAL = "critical"            # Law enforcement + immediate action
    HIGH = "high"                    # Security team + management
    MEDIUM = "medium"                # Compliance officer
    LOW = "low"                      # Logging only
    INFO = "info"                    # Informational


class SafetyIncident(Base):
    """
    Tracks safety incidents and blocked requests.
    CRITICAL: Immutable audit trail for legal compliance.
    """
    __tablename__ = "safety_incidents"

    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(String(255), unique=True, nullable=False, index=True)

    # Classification
    prohibited_category = Column(SQLEnum(ProhibitedCategory), nullable=True, index=True)  # Nullable for non-category incidents
    safety_action = Column(SQLEnum(SafetyAction), nullable=False)
    alert_level = Column(SQLEnum(AlertLevel), nullable=False, index=True)

    # Detection details
    detected_at_stage = Column(String(50), nullable=False)  # input_screening, context_analysis, etc.
    detection_confidence = Column(Float, nullable=False)    # 0.0 - 1.0
    matched_keywords = Column(JSON, default=list, nullable=False)
    matched_patterns = Column(JSON, default=list, nullable=False)

    # Request information
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    formula_id = Column(Integer, ForeignKey("formulas.id"), nullable=True)
    request_content = Column(Text, nullable=False)  # Full request for investigation
    request_context = Column(JSON, nullable=True)   # IP, location, device, etc.

    # Response
    action_taken = Column(Text, nullable=False)
    user_message = Column(Text, nullable=True)      # Message shown to user
    blocked_successfully = Column(Boolean, default=True, nullable=False)

    # Escalation
    reported_to = Column(JSON, default=list, nullable=False)  # ["law_enforcement", "security_team"]
    escalation_timestamp = Column(DateTime, nullable=True)
    investigation_status = Column(String(50), default="pending", nullable=False)

    # Session tracking
    session_id = Column(String(255), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    geographic_location = Column(String(100), nullable=True)

    # Metadata
    incident_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    resolution_notes = Column(Text, nullable=True)

    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    resolver = relationship("User", foreign_keys=[resolved_by_user_id])

    # Indexes
    __table_args__ = (
        Index('idx_safety_incident_category_time', 'prohibited_category', 'incident_timestamp'),
        Index('idx_safety_incident_alert', 'alert_level', 'investigation_status'),
        Index('idx_safety_incident_user', 'user_id', 'incident_timestamp'),
    )


class SafetyPattern(Base):
    """
    Behavioral patterns indicating potential harm.
    Used for proactive detection of coordinated harmful activity.
    """
    __tablename__ = "safety_patterns"

    id = Column(Integer, primary_key=True, index=True)
    pattern_id = Column(String(255), unique=True, nullable=False, index=True)

    # Pattern definition
    pattern_name = Column(String(255), nullable=False)
    pattern_category = Column(SQLEnum(ProhibitedCategory), nullable=False)
    pattern_description = Column(Text, nullable=False)

    # Detection criteria
    detection_rules = Column(JSON, nullable=False)  # Conditions that trigger detection
    time_window_minutes = Column(Integer, default=60, nullable=False)
    threshold_count = Column(Integer, default=3, nullable=False)

    # Severity
    risk_score = Column(Float, default=0.5, nullable=False)  # 0.0 - 1.0
    action_if_matched = Column(SQLEnum(SafetyAction), nullable=False)
    alert_level_if_matched = Column(SQLEnum(AlertLevel), nullable=False)

    # Pattern performance
    detection_count = Column(Integer, default=0, nullable=False)
    false_positive_count = Column(Integer, default=0, nullable=False)
    true_positive_count = Column(Integer, default=0, nullable=False)

    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    last_triggered = Column(DateTime, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Relationships
    created_by = relationship("User", foreign_keys=[created_by_user_id])

    # Indexes
    __table_args__ = (
        Index('idx_safety_pattern_category', 'pattern_category', 'is_active'),
    )


class UserSafetyScore(Base):
    """
    Tracks user safety scores and behavioral patterns.
    Used for risk assessment and proactive intervention.
    """
    __tablename__ = "user_safety_scores"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)

    # Safety metrics
    safety_score = Column(Float, default=1.0, nullable=False)  # 1.0 = safe, 0.0 = high risk
    risk_level = Column(String(50), default="low", nullable=False)  # low, medium, high, critical

    # Activity tracking
    total_requests = Column(Integer, default=0, nullable=False)
    blocked_requests = Column(Integer, default=0, nullable=False)
    warnings_issued = Column(Integer, default=0, nullable=False)
    safety_incidents = Column(Integer, default=0, nullable=False)

    # Pattern flags
    flagged_patterns = Column(JSON, default=list, nullable=False)  # List of matched concerning patterns
    suspicious_activity = Column(Boolean, default=False, nullable=False)

    # Status
    account_status = Column(String(50), default="active", nullable=False)  # active, restricted, suspended, banned
    restriction_reason = Column(Text, nullable=True)
    restricted_at = Column(DateTime, nullable=True)
    restricted_until = Column(DateTime, nullable=True)

    # Verification
    identity_verified = Column(Boolean, default=False, nullable=False)
    age_verified = Column(Boolean, default=False, nullable=False)
    professional_credentials_verified = Column(Boolean, default=False, nullable=False)

    # Last incident
    last_incident_at = Column(DateTime, nullable=True)
    last_incident_category = Column(String(100), nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="safety_score")

    def update_after_incident(self, incident: SafetyIncident):
        """Update safety score after an incident."""
        self.safety_incidents += 1
        self.blocked_requests += 1
        self.last_incident_at = incident.incident_timestamp
        self.last_incident_category = incident.prohibited_category.value

        # Decrease safety score based on severity
        severity_penalties = {
            ProhibitedCategory.CHILD_EXPLOITATION: -1.0,  # Immediate ban
            ProhibitedCategory.TERRORISM: -1.0,
            ProhibitedCategory.WEAPONS_MANUFACTURING: -0.5,
            ProhibitedCategory.ILLEGAL_DRUGS: -0.3,
            ProhibitedCategory.GAMBLING_MANIPULATION: -0.2,
        }

        penalty = severity_penalties.get(incident.prohibited_category, -0.1)
        self.safety_score = max(0.0, self.safety_score + penalty)

        # Update risk level
        if self.safety_score <= 0.0:
            self.risk_level = "critical"
            self.account_status = "banned"
        elif self.safety_score < 0.3:
            self.risk_level = "high"
            self.account_status = "suspended"
        elif self.safety_score < 0.6:
            self.risk_level = "medium"
            self.account_status = "restricted"
        else:
            self.risk_level = "low"


class SafetyConfiguration(Base):
    """
    Deployment-specific safety configuration.
    CANNOT DISABLE: Core protections are hardcoded and non-overridable.
    """
    __tablename__ = "safety_configurations"

    id = Column(Integer, primary_key=True, index=True)
    deployment_name = Column(String(255), unique=True, nullable=False, index=True)

    # Strict mode (cannot be disabled)
    strict_mode = Column(Boolean, default=True, nullable=False)  # ALWAYS TRUE
    compliance_framework = Column(String(100), nullable=True)  # e.g., "KSA_REGULATIONS"

    # Additional regional blocks (beyond core protections)
    additional_blocks = Column(JSON, default=dict, nullable=False)
    # e.g., {"alcohol": "TOTAL_BLOCK", "gambling": "TOTAL_BLOCK"} for KSA

    # Reporting configuration
    reporting_channels = Column(JSON, default=dict, nullable=False)
    # e.g., {"child_safety": ["local_authorities", "platform_admin"]}

    # Audit requirements
    audit_frequency = Column(String(50), default="continuous", nullable=False)
    audit_retention_days = Column(Integer, default=2555, nullable=False)  # 7 years
    encryption_standard = Column(String(50), default="AES-256", nullable=False)

    # Access control
    multi_factor_required = Column(Boolean, default=True, nullable=False)
    professional_verification_required = Column(Boolean, default=False, nullable=False)

    # Geographic restrictions
    allowed_regions = Column(JSON, nullable=True)  # List of allowed country codes
    blocked_regions = Column(JSON, default=list, nullable=False)  # Sanctioned regions

    # Metadata
    configured_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    configured_by = relationship("User", foreign_keys=[configured_by_user_id])


class EmergencyProtocol(Base):
    """
    Emergency response protocols for critical safety incidents.
    Defines immediate actions for each prohibited category.
    """
    __tablename__ = "emergency_protocols"

    id = Column(Integer, primary_key=True, index=True)
    protocol_id = Column(String(255), unique=True, nullable=False, index=True)

    # Protocol definition
    prohibited_category = Column(SQLEnum(ProhibitedCategory), unique=True, nullable=False, index=True)
    protocol_name = Column(String(255), nullable=False)
    protocol_description = Column(Text, nullable=False)

    # Immediate actions (executed automatically)
    immediate_actions = Column(JSON, nullable=False)
    # e.g., ["terminate_session", "preserve_logs", "alert_authorities"]

    # Notification recipients
    notify_law_enforcement = Column(Boolean, default=False, nullable=False)
    notify_security_team = Column(Boolean, default=True, nullable=False)
    notify_management = Column(Boolean, default=False, nullable=False)
    notification_recipients = Column(JSON, default=list, nullable=False)

    # User actions
    account_action = Column(String(50), nullable=False)  # lockdown, suspend, ban, monitor
    data_preservation = Column(Boolean, default=True, nullable=False)

    # Metadata
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Indexes
    __table_args__ = (
        Index('idx_emergency_protocol_category', 'prohibited_category', 'is_active'),
    )


# Add safety_score relationship to User model (if not already present)
# This will be added to auth.py via a migration or direct edit
