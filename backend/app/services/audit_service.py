"""
Audit logging service for tracking all system actions.

Every action (document upload, formula execution, human correction) must be
saved as an immutable record for verifiable decisions.
"""
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from app.models.corrections import AuditLog
from app.models.auth import User

logger = logging.getLogger(__name__)


class AuditService:
    """Service for creating and managing audit logs."""

    @staticmethod
    def log_action(
        db: Session,
        action: str,
        entity_type: str,
        description: str,
        user_id: Optional[int] = None,
        entity_id: Optional[int] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        """
        Create an immutable audit log entry.

        Args:
            db: Database session
            action: Action performed (e.g., "formula.execute", "correction.create")
            entity_type: Type of entity (e.g., "formula", "correction", "user")
            description: Human-readable description
            user_id: ID of user who performed action (None for system actions)
            entity_id: ID of the entity affected
            before_state: State before action
            after_state: State after action
            ip_address: IP address of requester
            user_agent: User agent string
            request_id: Unique request ID
            success: Whether action succeeded
            error_message: Error message if failed
            metadata: Additional metadata
        """
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            description=description,
            before_state=before_state or {},
            after_state=after_state or {},
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )

        db.add(audit_log)
        db.commit()
        db.refresh(audit_log)

        logger.info(f"Audit log created: {action} by user {user_id} on {entity_type} {entity_id}")

        return audit_log

    @staticmethod
    def log_formula_execution(
        db: Session,
        user_id: int,
        formula_id: int,
        execution_id: int,
        input_values: Dict[str, Any],
        output_values: Dict[str, Any],
        success: bool = True,
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """Log formula execution."""
        return AuditService.log_action(
            db=db,
            action="formula.execute",
            entity_type="formula_execution",
            entity_id=execution_id,
            description=f"Formula {formula_id} executed",
            user_id=user_id,
            before_state={"inputs": input_values},
            after_state={"outputs": output_values},
            success=success,
            error_message=error_message,
            ip_address=ip_address,
            request_id=request_id,
            metadata={
                "formula_id": formula_id,
                "execution_id": execution_id
            }
        )

    @staticmethod
    def log_correction_created(
        db: Session,
        user_id: int,
        correction_id: int,
        execution_id: int,
        original_output: Dict[str, Any],
        corrected_output: Dict[str, Any],
        correction_reason: Optional[str] = None,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """Log correction creation."""
        return AuditService.log_action(
            db=db,
            action="correction.create",
            entity_type="correction",
            entity_id=correction_id,
            description=f"Correction created for execution {execution_id}",
            user_id=user_id,
            before_state={"original": original_output},
            after_state={"corrected": corrected_output},
            success=True,
            ip_address=ip_address,
            request_id=request_id,
            metadata={
                "execution_id": execution_id,
                "correction_id": correction_id,
                "reason": correction_reason
            }
        )

    @staticmethod
    def log_correction_reviewed(
        db: Session,
        user_id: int,
        correction_id: int,
        status: str,
        review_notes: Optional[str] = None,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """Log correction review."""
        return AuditService.log_action(
            db=db,
            action="correction.review",
            entity_type="correction",
            entity_id=correction_id,
            description=f"Correction {correction_id} reviewed: {status}",
            user_id=user_id,
            after_state={"status": status, "notes": review_notes},
            success=True,
            ip_address=ip_address,
            request_id=request_id,
            metadata={
                "correction_id": correction_id,
                "status": status
            }
        )

    @staticmethod
    def log_formula_certified(
        db: Session,
        user_id: int,
        formula_id: int,
        certification_id: int,
        from_tier: int,
        to_tier: int,
        from_version: str,
        to_version: str,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """Log formula certification."""
        return AuditService.log_action(
            db=db,
            action="formula.certify",
            entity_type="formula_certification",
            entity_id=certification_id,
            description=f"Formula {formula_id} certified from Tier {from_tier} to Tier {to_tier}",
            user_id=user_id,
            before_state={"tier": from_tier, "version": from_version},
            after_state={"tier": to_tier, "version": to_version},
            success=True,
            ip_address=ip_address,
            request_id=request_id,
            metadata={
                "formula_id": formula_id,
                "certification_id": certification_id,
                "from_tier": from_tier,
                "to_tier": to_tier
            }
        )

    @staticmethod
    def log_document_uploaded(
        db: Session,
        user_id: int,
        document_id: int,
        filename: str,
        file_size: int,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """Log document upload."""
        return AuditService.log_action(
            db=db,
            action="document.upload",
            entity_type="document",
            entity_id=document_id,
            description=f"Document uploaded: {filename}",
            user_id=user_id,
            after_state={"filename": filename, "size": file_size},
            success=True,
            ip_address=ip_address,
            request_id=request_id,
            metadata={
                "document_id": document_id,
                "filename": filename,
                "file_size": file_size
            }
        )

    @staticmethod
    def log_user_action(
        db: Session,
        user_id: int,
        action: str,
        target_user_id: Optional[int] = None,
        description: Optional[str] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """Log user management actions."""
        return AuditService.log_action(
            db=db,
            action=f"user.{action}",
            entity_type="user",
            entity_id=target_user_id,
            description=description or f"User {action}",
            user_id=user_id,
            before_state=before_state,
            after_state=after_state,
            success=True,
            ip_address=ip_address,
            request_id=request_id,
            metadata={"target_user_id": target_user_id}
        )

    @staticmethod
    def get_audit_logs(
        db: Session,
        action: Optional[str] = None,
        entity_type: Optional[str] = None,
        user_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ):
        """Query audit logs with filters."""
        query = db.query(AuditLog)

        if action:
            query = query.filter(AuditLog.action == action)
        if entity_type:
            query = query.filter(AuditLog.entity_type == entity_type)
        if user_id:
            query = query.filter(AuditLog.user_id == user_id)
        if start_date:
            query = query.filter(AuditLog.created_at >= start_date)
        if end_date:
            query = query.filter(AuditLog.created_at <= end_date)

        return query.order_by(AuditLog.created_at.desc()).limit(limit).all()


# Create global instance
audit_service = AuditService()
