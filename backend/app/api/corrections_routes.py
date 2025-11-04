"""
Corrections API routes.

Operators submit corrections to AI outputs. Admins review and approve them.
These corrections become the foundation for the auto-retrain loop.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.rbac import get_current_operator, get_current_admin, rbac
from app.models.auth import User
from app.models.corrections import Correction, CorrectionStatus, CorrectionType
from app.models.database import FormulaExecution
from app.services.audit_service import audit_service

router = APIRouter()


# Pydantic schemas
class CorrectionCreate(BaseModel):
    execution_id: int
    correction_type: str
    corrected_output: dict
    correction_reason: Optional[str] = None
    operator_confidence: int = 100


class CorrectionReview(BaseModel):
    status: str  # "approved" or "rejected"
    review_notes: Optional[str] = None


class CorrectionResponse(BaseModel):
    id: int
    execution_id: int
    correction_type: str
    status: str
    original_output: dict
    corrected_output: dict
    correction_reason: Optional[str]
    operator_confidence: int
    corrected_by_user_id: int
    reviewed_by_user_id: Optional[int]
    reviewed_at: Optional[str]
    review_notes: Optional[str]
    used_in_training: bool
    training_run_id: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


@router.post("/corrections", response_model=CorrectionResponse, status_code=status.HTTP_201_CREATED)
async def create_correction(
    correction_data: CorrectionCreate,
    request: Request,
    current_user: User = Depends(get_current_operator),
    db: Session = Depends(get_db)
):
    """
    Create a new correction (Operator only).

    When an operator corrects an AI output, this creates an immutable record
    of the correction. This is the foundation of verifiable decisions.
    """
    # Check permissions
    rbac.check_correction_submission(current_user)

    # Verify execution exists
    execution = db.query(FormulaExecution).filter(
        FormulaExecution.id == correction_data.execution_id
    ).first()

    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {correction_data.execution_id} not found"
        )

    # Validate correction type
    try:
        correction_type = CorrectionType[correction_data.correction_type.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid correction type: {correction_data.correction_type}"
        )

    # Create correction
    correction = Correction(
        execution_id=correction_data.execution_id,
        corrected_by_user_id=current_user.id,
        correction_type=correction_type,
        status=CorrectionStatus.PENDING,
        original_output=execution.output_values or {},
        corrected_output=correction_data.corrected_output,
        correction_reason=correction_data.correction_reason,
        operator_confidence=correction_data.operator_confidence
    )

    db.add(correction)
    db.commit()
    db.refresh(correction)

    # Audit log
    audit_service.log_correction_created(
        db=db,
        user_id=current_user.id,
        correction_id=correction.id,
        execution_id=execution.id,
        original_output=execution.output_values or {},
        corrected_output=correction_data.corrected_output,
        correction_reason=correction_data.correction_reason,
        ip_address=request.client.host if request.client else None,
        request_id=request.headers.get("X-Request-ID")
    )

    return CorrectionResponse(
        id=correction.id,
        execution_id=correction.execution_id,
        correction_type=correction.correction_type.value,
        status=correction.status.value,
        original_output=correction.original_output,
        corrected_output=correction.corrected_output,
        correction_reason=correction.correction_reason,
        operator_confidence=correction.operator_confidence,
        corrected_by_user_id=correction.corrected_by_user_id,
        reviewed_by_user_id=correction.reviewed_by_user_id,
        reviewed_at=correction.reviewed_at.isoformat() if correction.reviewed_at else None,
        review_notes=correction.review_notes,
        used_in_training=correction.used_in_training,
        training_run_id=correction.training_run_id,
        created_at=correction.created_at.isoformat()
    )


@router.get("/corrections", response_model=List[CorrectionResponse])
async def list_corrections(
    status_filter: Optional[str] = None,
    correction_type: Optional[str] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List corrections.

    - Operators see their own corrections
    - Admins and Auditors see all corrections
    """
    query = db.query(Correction)

    # Filter by user role
    if current_user.role.value == "operator":
        query = query.filter(Correction.corrected_by_user_id == current_user.id)

    # Apply filters
    if status_filter:
        try:
            status_enum = CorrectionStatus[status_filter.upper()]
            query = query.filter(Correction.status == status_enum)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}"
            )

    if correction_type:
        try:
            type_enum = CorrectionType[correction_type.upper()]
            query = query.filter(Correction.correction_type == type_enum)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid correction type: {correction_type}"
            )

    corrections = query.order_by(Correction.created_at.desc()).limit(limit).all()

    return [
        CorrectionResponse(
            id=c.id,
            execution_id=c.execution_id,
            correction_type=c.correction_type.value,
            status=c.status.value,
            original_output=c.original_output,
            corrected_output=c.corrected_output,
            correction_reason=c.correction_reason,
            operator_confidence=c.operator_confidence,
            corrected_by_user_id=c.corrected_by_user_id,
            reviewed_by_user_id=c.reviewed_by_user_id,
            reviewed_at=c.reviewed_at.isoformat() if c.reviewed_at else None,
            review_notes=c.review_notes,
            used_in_training=c.used_in_training,
            training_run_id=c.training_run_id,
            created_at=c.created_at.isoformat()
        ) for c in corrections
    ]


@router.get("/corrections/{correction_id}", response_model=CorrectionResponse)
async def get_correction(
    correction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific correction."""
    correction = db.query(Correction).filter(Correction.id == correction_id).first()

    if not correction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Correction {correction_id} not found"
        )

    # Check permissions
    if current_user.role.value == "operator" and correction.corrected_by_user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own corrections"
        )

    return CorrectionResponse(
        id=correction.id,
        execution_id=correction.execution_id,
        correction_type=correction.correction_type.value,
        status=correction.status.value,
        original_output=correction.original_output,
        corrected_output=correction.corrected_output,
        correction_reason=correction.correction_reason,
        operator_confidence=correction.operator_confidence,
        corrected_by_user_id=correction.corrected_by_user_id,
        reviewed_by_user_id=correction.reviewed_by_user_id,
        reviewed_at=correction.reviewed_at.isoformat() if correction.reviewed_at else None,
        review_notes=correction.review_notes,
        used_in_training=correction.used_in_training,
        training_run_id=correction.training_run_id,
        created_at=correction.created_at.isoformat()
    )


@router.patch("/corrections/{correction_id}/review", response_model=CorrectionResponse)
async def review_correction(
    correction_id: int,
    review_data: CorrectionReview,
    request: Request,
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Review a correction (Admin only).

    Admins review corrections and approve/reject them. Approved corrections
    are used in the auto-retrain pipeline.
    """
    rbac.check_correction_review(current_user)

    correction = db.query(Correction).filter(Correction.id == correction_id).first()

    if not correction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Correction {correction_id} not found"
        )

    # Validate status
    if review_data.status not in ["approved", "rejected"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Status must be 'approved' or 'rejected'"
        )

    # Update correction
    correction.status = CorrectionStatus[review_data.status.upper()]
    correction.reviewed_by_user_id = current_user.id
    correction.reviewed_at = datetime.utcnow()
    correction.review_notes = review_data.review_notes

    db.commit()
    db.refresh(correction)

    # Audit log
    audit_service.log_correction_reviewed(
        db=db,
        user_id=current_user.id,
        correction_id=correction.id,
        status=review_data.status,
        review_notes=review_data.review_notes,
        ip_address=request.client.host if request.client else None,
        request_id=request.headers.get("X-Request-ID")
    )

    return CorrectionResponse(
        id=correction.id,
        execution_id=correction.execution_id,
        correction_type=correction.correction_type.value,
        status=correction.status.value,
        original_output=correction.original_output,
        corrected_output=correction.corrected_output,
        correction_reason=correction.correction_reason,
        operator_confidence=correction.operator_confidence,
        corrected_by_user_id=correction.corrected_by_user_id,
        reviewed_by_user_id=correction.reviewed_by_user_id,
        reviewed_at=correction.reviewed_at.isoformat() if correction.reviewed_at else None,
        review_notes=correction.review_notes,
        used_in_training=correction.used_in_training,
        training_run_id=correction.training_run_id,
        created_at=correction.created_at.isoformat()
    )


@router.get("/corrections/pending/count")
async def get_pending_corrections_count(
    current_user: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """Get count of pending corrections (Admin only)."""
    count = db.query(Correction).filter(
        Correction.status == CorrectionStatus.PENDING
    ).count()

    return {"count": count}
