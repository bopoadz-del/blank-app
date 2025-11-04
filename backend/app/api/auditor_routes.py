"""
Auditor Dashboard API routes.

Read-only access to audit trail, corrections, and execution logs.
Auditors can see a perfect, verifiable trail of all interactions.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.rbac import get_current_auditor, rbac
from app.models.auth import User
from app.models.corrections import AuditLog, Correction, CorrectionStatus
from app.models.database import FormulaExecution, Formula
from app.services.audit_service import audit_service

router = APIRouter()


# Pydantic schemas
class AuditLogResponse(BaseModel):
    id: int
    user_id: Optional[int]
    action: str
    entity_type: str
    entity_id: Optional[int]
    description: str
    before_state: dict
    after_state: dict
    ip_address: Optional[str]
    user_agent: Optional[str]
    request_id: Optional[str]
    success: bool
    error_message: Optional[str]
    created_at: str
    metadata: dict

    class Config:
        from_attributes = True


class ExecutionTrailResponse(BaseModel):
    """Complete trail of an execution with corrections."""
    execution: dict
    corrections: List[dict]
    audit_logs: List[dict]


class DashboardStats(BaseModel):
    """Statistics for auditor dashboard."""
    total_executions: int
    total_corrections: int
    pending_corrections: int
    approved_corrections: int
    rejected_corrections: int
    recent_audit_logs: int
    tier_1_formulas: int
    tier_4_formulas: int


@router.get("/audit-logs", response_model=List[AuditLogResponse])
async def get_audit_logs(
    action: Optional[str] = None,
    entity_type: Optional[str] = None,
    user_id: Optional[int] = None,
    days: int = 7,
    limit: int = 100,
    current_user: User = Depends(get_current_auditor),
    db: Session = Depends(get_db)
):
    """
    Get audit logs (Auditor/Admin only).

    Provides complete read-only access to all system actions.
    """
    rbac.check_audit_log_access(current_user)

    start_date = datetime.utcnow() - timedelta(days=days)

    logs = audit_service.get_audit_logs(
        db=db,
        action=action,
        entity_type=entity_type,
        user_id=user_id,
        start_date=start_date,
        limit=limit
    )

    return [
        AuditLogResponse(
            id=log.id,
            user_id=log.user_id,
            action=log.action,
            entity_type=log.entity_type,
            entity_id=log.entity_id,
            description=log.description,
            before_state=log.before_state or {},
            after_state=log.after_state or {},
            ip_address=log.ip_address,
            user_agent=log.user_agent,
            request_id=log.request_id,
            success=log.success,
            error_message=log.error_message,
            created_at=log.created_at.isoformat(),
            metadata=log.metadata or {}
        ) for log in logs
    ]


@router.get("/execution-trail/{execution_id}", response_model=ExecutionTrailResponse)
async def get_execution_trail(
    execution_id: int,
    current_user: User = Depends(get_current_auditor),
    db: Session = Depends(get_db)
):
    """
    Get complete trail for an execution.

    Shows execution -> corrections -> audit logs in chronological order.
    This provides the complete verifiable chain of events.
    """
    rbac.check_audit_log_access(current_user)

    # Get execution
    execution = db.query(FormulaExecution).filter(
        FormulaExecution.id == execution_id
    ).first()

    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {execution_id} not found"
        )

    # Get corrections
    corrections = db.query(Correction).filter(
        Correction.execution_id == execution_id
    ).all()

    # Get audit logs
    audit_logs = db.query(AuditLog).filter(
        AuditLog.entity_id == execution_id,
        AuditLog.entity_type.in_(["formula_execution", "correction"])
    ).order_by(AuditLog.created_at).all()

    return ExecutionTrailResponse(
        execution={
            "id": execution.id,
            "execution_id": execution.execution_id,
            "formula_id": execution.formula_id,
            "input_values": execution.input_values,
            "output_values": execution.output_values,
            "status": execution.status.value,
            "execution_time": execution.execution_time,
            "executed_by": execution.executed_by,
            "execution_timestamp": execution.execution_timestamp.isoformat()
        },
        corrections=[
            {
                "id": c.id,
                "correction_type": c.correction_type.value,
                "status": c.status.value,
                "original_output": c.original_output,
                "corrected_output": c.corrected_output,
                "correction_reason": c.correction_reason,
                "operator_confidence": c.operator_confidence,
                "corrected_by_user_id": c.corrected_by_user_id,
                "reviewed_by_user_id": c.reviewed_by_user_id,
                "reviewed_at": c.reviewed_at.isoformat() if c.reviewed_at else None,
                "used_in_training": c.used_in_training,
                "training_run_id": c.training_run_id,
                "created_at": c.created_at.isoformat()
            } for c in corrections
        ],
        audit_logs=[
            {
                "id": log.id,
                "action": log.action,
                "description": log.description,
                "user_id": log.user_id,
                "before_state": log.before_state,
                "after_state": log.after_state,
                "success": log.success,
                "created_at": log.created_at.isoformat()
            } for log in audit_logs
        ]
    )


@router.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(
    days: int = 30,
    current_user: User = Depends(get_current_auditor),
    db: Session = Depends(get_db)
):
    """
    Get dashboard statistics (Auditor/Admin only).

    Provides high-level metrics for the auditor dashboard.
    """
    rbac.check_audit_log_access(current_user)

    start_date = datetime.utcnow() - timedelta(days=days)

    # Calculate stats
    total_executions = db.query(FormulaExecution).filter(
        FormulaExecution.execution_timestamp >= start_date
    ).count()

    total_corrections = db.query(Correction).filter(
        Correction.created_at >= start_date
    ).count()

    pending_corrections = db.query(Correction).filter(
        Correction.status == CorrectionStatus.PENDING
    ).count()

    approved_corrections = db.query(Correction).filter(
        Correction.status == CorrectionStatus.APPROVED,
        Correction.created_at >= start_date
    ).count()

    rejected_corrections = db.query(Correction).filter(
        Correction.status == CorrectionStatus.REJECTED,
        Correction.created_at >= start_date
    ).count()

    recent_audit_logs = db.query(AuditLog).filter(
        AuditLog.created_at >= start_date
    ).count()

    from app.models.database import FormulaTier
    tier_1_formulas = db.query(Formula).filter(
        Formula.tier == FormulaTier.TIER_1_CERTIFIED
    ).count()

    tier_4_formulas = db.query(Formula).filter(
        Formula.tier == FormulaTier.TIER_4_EXPERIMENTAL
    ).count()

    return DashboardStats(
        total_executions=total_executions,
        total_corrections=total_corrections,
        pending_corrections=pending_corrections,
        approved_corrections=approved_corrections,
        rejected_corrections=rejected_corrections,
        recent_audit_logs=recent_audit_logs,
        tier_1_formulas=tier_1_formulas,
        tier_4_formulas=tier_4_formulas
    )


@router.get("/corrections/timeline")
async def get_corrections_timeline(
    days: int = 30,
    current_user: User = Depends(get_current_auditor),
    db: Session = Depends(get_db)
):
    """
    Get corrections timeline (Auditor/Admin only).

    Shows trend of corrections over time.
    """
    rbac.check_audit_log_access(current_user)

    start_date = datetime.utcnow() - timedelta(days=days)

    corrections = db.query(Correction).filter(
        Correction.created_at >= start_date
    ).order_by(Correction.created_at).all()

    # Group by date
    timeline = {}
    for correction in corrections:
        date_key = correction.created_at.date().isoformat()
        if date_key not in timeline:
            timeline[date_key] = {
                "date": date_key,
                "total": 0,
                "pending": 0,
                "approved": 0,
                "rejected": 0
            }

        timeline[date_key]["total"] += 1
        if correction.status == CorrectionStatus.PENDING:
            timeline[date_key]["pending"] += 1
        elif correction.status == CorrectionStatus.APPROVED:
            timeline[date_key]["approved"] += 1
        elif correction.status == CorrectionStatus.REJECTED:
            timeline[date_key]["rejected"] += 1

    return {"timeline": list(timeline.values())}


@router.get("/formulas/tier-distribution")
async def get_formula_tier_distribution(
    current_user: User = Depends(get_current_auditor),
    db: Session = Depends(get_db)
):
    """
    Get formula tier distribution (Auditor/Admin only).

    Shows how many formulas are in each tier.
    """
    rbac.check_audit_log_access(current_user)

    from app.models.database import FormulaTier

    distribution = {}
    for tier in FormulaTier:
        count = db.query(Formula).filter(Formula.tier == tier).count()
        distribution[f"tier_{tier.value}"] = count

    return {"distribution": distribution}
