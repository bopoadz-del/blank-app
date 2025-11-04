"""
Authentication models and schemas for JWT-based auth.
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Enum as SQLEnum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum

from app.models.database import Base


class UserRole(enum.Enum):
    """User roles with specific permissions."""
    OPERATOR = "operator"  # Can run formulas and submit corrections
    ADMIN = "admin"        # Can certify models and manage system
    AUDITOR = "auditor"    # Read-only access to audit trail
    SYSTEM = "system"      # Internal system operations


class User(Base):
    """User model for authentication with RBAC."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.OPERATOR, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships for corrections system
    corrections_made = relationship("Correction", foreign_keys="Correction.corrected_by_user_id", back_populates="corrected_by")
    corrections_reviewed = relationship("Correction", foreign_keys="Correction.reviewed_by_user_id", back_populates="reviewed_by")
    audit_logs = relationship("AuditLog", back_populates="user")
    certifications_made = relationship("FormulaCertification", back_populates="certified_by")


class RefreshToken(Base):
    """Refresh token storage."""
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    token = Column(String, unique=True, index=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
