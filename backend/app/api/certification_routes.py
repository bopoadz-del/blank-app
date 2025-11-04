"""
Formula Certification API routes.

Admins certify formulas through the Tier system:
Tier 4 (experimental) -> Tier 3 -> Tier 2 -> Tier 1 (certified for production)
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from app.core.database import get_db
from app.core.rbac import get_current_admin, rbac
from app.models.auth import User
from app.models.corrections import FormulaCertification
from app.models.database import Formula, FormulaTier
from app.services.audit_service import audit_service

router = APIRouter()


# Pydantic schemas
class CertificationCreate(BaseModel):
    formula_id: int
    to_tier: int  # Target tier (1-4)
    certification_notes: Optional[str] = None
    test_accuracy: Optional[dict] = None
    validation_metrics: Optional[dict] = None
    review_period_days: Optional[int] = 7


class CertificationResponse(BaseModel):
    id: int
    formula_id: int
    from_version: str
    to_version: str
    from_tier: int
    to_tier: int
    certified_by_user_id: int
    certification_notes: Optional[str]
    test_accuracy: Optional[dict]
    validation_metrics: Optional[dict]
    review_period_start: Optional[str]
    review_period_end: Optional[str]
    executions_reviewed: int
    corrections_count: int
    certified_at: str
    is_locked: bool

    class Config:
        from_attributes = True


@router.post("/certifications", response_model=CertificationResponse, status_code=status.HTTP_201_CREATED)
async def certify_formula(
    cert_data: CertificationCreate,
    request: Request,
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Certify a formula (promote to higher tier) - Admin only.

    Rules:
    - Only admins can certify formulas
    - Can only promote one tier at a time (e.g., Tier 4 -> Tier 3)
    - Tier 1 certifications are locked and immutable
    - Must provide certification notes and metrics
    """
    rbac.check_formula_certification(current_user)

    # Get formula
    formula = db.query(Formula).filter(Formula.id == cert_data.formula_id).first()

    if not formula:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Formula {cert_data.formula_id} not found"
        )

    # Check if formula is locked
    if formula.is_locked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This formula is locked and cannot be modified"
        )

    # Validate tier promotion
    current_tier = formula.tier.value
    target_tier = cert_data.to_tier

    if target_tier >= current_tier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Can only promote to a lower tier number (current: Tier {current_tier})"
        )

    if target_tier < 1 or target_tier > 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tier must be between 1 and 4"
        )

    # Can only promote one tier at a time
    if current_tier - target_tier != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Can only promote one tier at a time (current: Tier {current_tier}, requested: Tier {target_tier})"
        )

    # Generate new version
    from_version = formula.version
    version_parts = from_version.split('.')
    if target_tier == 1:
        # Tier 1 gets a production version
        to_version = f"{version_parts[0]}.{int(version_parts[1]) + 1}.0-prod"
    else:
        to_version = f"{version_parts[0]}.{version_parts[1]}.{int(version_parts[2].split('-')[0]) + 1}-tier{target_tier}"

    # Calculate review period
    review_period_end = datetime.utcnow()
    review_period_start = None
    if cert_data.review_period_days:
        from datetime import timedelta
        review_period_start = review_period_end - timedelta(days=cert_data.review_period_days)

    # Count executions and corrections in review period
    executions_count = formula.total_executions
    corrections_count = db.query(Correction).join(
        FormulaExecution, Correction.execution_id == FormulaExecution.id
    ).filter(
        FormulaExecution.formula_id == formula.id
    ).count() if 'Correction' in dir() else 0

    # Create certification record
    certification = FormulaCertification(
        formula_id=formula.id,
        from_version=from_version,
        to_version=to_version,
        from_tier=current_tier,
        to_tier=target_tier,
        certified_by_user_id=current_user.id,
        certification_notes=cert_data.certification_notes,
        test_accuracy=cert_data.test_accuracy,
        validation_metrics=cert_data.validation_metrics,
        review_period_start=review_period_start,
        review_period_end=review_period_end,
        executions_reviewed=executions_count,
        corrections_count=corrections_count,
        is_locked=True
    )

    db.add(certification)

    # Update formula
    formula.tier = FormulaTier(target_tier)
    formula.version = to_version
    formula.tier_updated_at = datetime.utcnow()
    formula.tier_change_reason = f"Certified by {current_user.username}"

    # Lock formula if promoted to Tier 1
    if target_tier == 1:
        formula.is_locked = True

    db.commit()
    db.refresh(certification)

    # Audit log
    audit_service.log_formula_certified(
        db=db,
        user_id=current_user.id,
        formula_id=formula.id,
        certification_id=certification.id,
        from_tier=current_tier,
        to_tier=target_tier,
        from_version=from_version,
        to_version=to_version,
        ip_address=request.client.host if request.client else None,
        request_id=request.headers.get("X-Request-ID")
    )

    return CertificationResponse(
        id=certification.id,
        formula_id=certification.formula_id,
        from_version=certification.from_version,
        to_version=certification.to_version,
        from_tier=certification.from_tier,
        to_tier=certification.to_tier,
        certified_by_user_id=certification.certified_by_user_id,
        certification_notes=certification.certification_notes,
        test_accuracy=certification.test_accuracy,
        validation_metrics=certification.validation_metrics,
        review_period_start=certification.review_period_start.isoformat() if certification.review_period_start else None,
        review_period_end=certification.review_period_end.isoformat() if certification.review_period_end else None,
        executions_reviewed=certification.executions_reviewed,
        corrections_count=certification.corrections_count,
        certified_at=certification.certified_at.isoformat(),
        is_locked=certification.is_locked
    )


@router.get("/certifications", response_model=List[CertificationResponse])
async def list_certifications(
    formula_id: Optional[int] = None,
    tier: Optional[int] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """List formula certifications (Admin only)."""
    query = db.query(FormulaCertification)

    if formula_id:
        query = query.filter(FormulaCertification.formula_id == formula_id)

    if tier:
        query = query.filter(FormulaCertification.to_tier == tier)

    certifications = query.order_by(FormulaCertification.certified_at.desc()).limit(limit).all()

    return [
        CertificationResponse(
            id=c.id,
            formula_id=c.formula_id,
            from_version=c.from_version,
            to_version=c.to_version,
            from_tier=c.from_tier,
            to_tier=c.to_tier,
            certified_by_user_id=c.certified_by_user_id,
            certification_notes=c.certification_notes,
            test_accuracy=c.test_accuracy,
            validation_metrics=c.validation_metrics,
            review_period_start=c.review_period_start.isoformat() if c.review_period_start else None,
            review_period_end=c.review_period_end.isoformat() if c.review_period_end else None,
            executions_reviewed=c.executions_reviewed,
            corrections_count=c.corrections_count,
            certified_at=c.certified_at.isoformat(),
            is_locked=c.is_locked
        ) for c in certifications
    ]


@router.get("/formulas/{formula_id}/certification-history", response_model=List[CertificationResponse])
async def get_formula_certification_history(
    formula_id: int,
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Get certification history for a formula (Admin only)."""
    formula = db.query(Formula).filter(Formula.id == formula_id).first()

    if not formula:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Formula {formula_id} not found"
        )

    certifications = db.query(FormulaCertification).filter(
        FormulaCertification.formula_id == formula_id
    ).order_by(FormulaCertification.certified_at.desc()).all()

    return [
        CertificationResponse(
            id=c.id,
            formula_id=c.formula_id,
            from_version=c.from_version,
            to_version=c.to_version,
            from_tier=c.from_tier,
            to_tier=c.to_tier,
            certified_by_user_id=c.certified_by_user_id,
            certification_notes=c.certification_notes,
            test_accuracy=c.test_accuracy,
            validation_metrics=c.validation_metrics,
            review_period_start=c.review_period_start.isoformat() if c.review_period_start else None,
            review_period_end=c.review_period_end.isoformat() if c.review_period_end else None,
            executions_reviewed=c.executions_reviewed,
            corrections_count=c.corrections_count,
            certified_at=c.certified_at.isoformat(),
            is_locked=c.is_locked
        ) for c in certifications
    ]
