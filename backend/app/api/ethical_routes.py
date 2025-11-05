"""
Ethical Layer API routes.
Manage knowledge sources, validation results, overrides, and audit logs.
"""
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ..core.database import get_db
from ..models.ethical_layer import (
    KnowledgeSource,
    SourceType,
    CredibilityTier,
    FormulaValidationResult,
    ValidationStatus,
    EthicalOverride,
    EthicalConfiguration,
    EthicalAuditLog
)
from ..models.database import Formula
from ..services.validation_pipeline import ValidationPipeline, CredibilityLearning
from ..services.ethical_safeguards import EthicalSafeguards, STANDARD_OVERRIDES
from ..core.rbac import get_current_admin, get_current_operator


router = APIRouter(prefix="/api/v1/ethical", tags=["Ethical Layer"])


# Pydantic Schemas
class KnowledgeSourceCreate(BaseModel):
    source_id: str
    source_name: str
    source_type: str
    publication_date: Optional[datetime] = None
    version: Optional[str] = None
    issuing_authority: Optional[str] = None
    geographic_scope: Optional[List[str]] = None
    domain_tags: Optional[List[str]] = None
    initial_credibility: float = Field(default=0.5, ge=0.0, le=1.0)


class KnowledgeSourceResponse(BaseModel):
    id: int
    source_id: str
    source_name: str
    source_type: str
    credibility_score: float
    credibility_tier: int
    usage_count: int
    success_count: int
    failure_count: int
    is_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ValidationTrigger(BaseModel):
    formula_id: int
    context: Optional[dict] = None


class ValidationResultResponse(BaseModel):
    id: int
    formula_id: int
    validation_run_id: str
    final_status: str
    assigned_tier: Optional[int]
    syntactic_passed: bool
    dimensional_passed: bool
    physical_passed: bool
    empirical_passed: bool
    safety_passed: bool
    historical_accuracy: Optional[float]
    safety_score: Optional[float]
    validation_timestamp: datetime

    class Config:
        from_attributes = True


class EthicalOverrideCreate(BaseModel):
    override_category: str
    override_name: str
    trigger_conditions: dict
    credibility_adjustment: float = Field(default=0.0, ge=-0.5, le=0.5)
    safety_margin_multiplier: float = Field(default=1.0, ge=0.5, le=2.0)
    applicable_domains: Optional[List[str]] = []
    applicable_deployments: Optional[List[str]] = []
    description: Optional[str] = None


class EthicalOverrideResponse(BaseModel):
    id: int
    override_id: str
    override_category: str
    override_name: str
    trigger_conditions: dict
    credibility_adjustment: float
    safety_margin_multiplier: float
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class EthicalConfigResponse(BaseModel):
    id: int
    deployment_name: str
    default_tier: int
    auto_promotion_enabled: bool
    safety_margin: float
    domain_min_tiers: dict
    red_lines: List[str]
    audit_frequency: str

    class Config:
        from_attributes = True


class EthicalAuditLogResponse(BaseModel):
    id: int
    audit_id: str
    decision_type: str
    formula_id: Optional[int]
    credibility_tier_assigned: Optional[int]
    overrides_applied: List[str]
    red_lines_checked: List[str]
    decision_explanation: str
    created_at: datetime

    class Config:
        from_attributes = True


# Routes

@router.post("/sources", response_model=KnowledgeSourceResponse, status_code=status.HTTP_201_CREATED)
async def create_knowledge_source(
    source_data: KnowledgeSourceCreate,
    current_user = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Register a new knowledge source (admin only).
    """
    # Check if source already exists
    existing = db.query(KnowledgeSource).filter(
        KnowledgeSource.source_id == source_data.source_id
    ).first()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Source with ID {source_data.source_id} already exists"
        )

    # Create source
    source = KnowledgeSource(
        source_id=source_data.source_id,
        source_name=source_data.source_name,
        source_type=SourceType(source_data.source_type),
        credibility_score=source_data.initial_credibility,
        publication_date=source_data.publication_date,
        version=source_data.version,
        issuing_authority=source_data.issuing_authority,
        geographic_scope=source_data.geographic_scope or [],
        domain_tags=source_data.domain_tags or []
    )

    # Assign initial tier based on source type
    if source.source_type in [
        SourceType.ISO_STANDARD,
        SourceType.ANSI_STANDARD,
        SourceType.GOVERNMENT_REGULATION
    ]:
        source.credibility_tier = CredibilityTier.TIER_1_FULLY_AUTOMATED
        source.credibility_score = 0.98
    elif source.source_type in [
        SourceType.CONSULTANT_REPORT,
        SourceType.MANUFACTURER_SPEC,
        SourceType.HISTORICAL_DATA
    ]:
        source.credibility_tier = CredibilityTier.TIER_2_SUPERVISED
        source.credibility_score = 0.80
    else:
        source.credibility_tier = CredibilityTier.TIER_3_EXPERIMENTAL
        source.credibility_score = 0.50

    db.add(source)
    db.commit()
    db.refresh(source)

    return source


@router.get("/sources", response_model=List[KnowledgeSourceResponse])
async def list_knowledge_sources(
    source_type: Optional[str] = None,
    min_credibility: float = 0.0,
    limit: int = 100,
    current_user = Depends(get_current_operator),
    db: Session = Depends(get_db)
):
    """
    List all knowledge sources with optional filters.
    """
    query = db.query(KnowledgeSource)

    if source_type:
        query = query.filter(KnowledgeSource.source_type == SourceType(source_type))

    if min_credibility > 0:
        query = query.filter(KnowledgeSource.credibility_score >= min_credibility)

    sources = query.order_by(KnowledgeSource.credibility_score.desc()).limit(limit).all()

    return sources


@router.get("/sources/{source_id}", response_model=KnowledgeSourceResponse)
async def get_knowledge_source(
    source_id: str,
    current_user = Depends(get_current_operator),
    db: Session = Depends(get_db)
):
    """
    Get details of a specific knowledge source.
    """
    source = db.query(KnowledgeSource).filter(
        KnowledgeSource.source_id == source_id
    ).first()

    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source {source_id} not found"
        )

    return source


@router.post("/sources/{source_id}/verify")
async def verify_knowledge_source(
    source_id: str,
    current_user = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Verify a knowledge source (admin only).
    """
    source = db.query(KnowledgeSource).filter(
        KnowledgeSource.source_id == source_id
    ).first()

    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source {source_id} not found"
        )

    source.is_verified = True
    source.verified_by_user_id = current_user.id
    source.verified_at = datetime.utcnow()

    db.commit()

    return {"success": True, "message": f"Source {source_id} verified"}


@router.post("/validate", response_model=ValidationResultResponse)
async def validate_formula(
    validation_data: ValidationTrigger,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Run 5-stage validation pipeline on a formula (admin only).
    """
    formula = db.query(Formula).filter(Formula.id == validation_data.formula_id).first()

    if not formula:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Formula {validation_data.formula_id} not found"
        )

    # Run validation
    pipeline = ValidationPipeline(db)
    result = pipeline.validate_formula(
        formula=formula,
        context=validation_data.context or {}
    )

    return result


@router.get("/validation-results", response_model=List[ValidationResultResponse])
async def list_validation_results(
    formula_id: Optional[int] = None,
    status_filter: Optional[str] = None,
    limit: int = 100,
    current_user = Depends(get_current_operator),
    db: Session = Depends(get_db)
):
    """
    List validation results with optional filters.
    """
    query = db.query(FormulaValidationResult)

    if formula_id:
        query = query.filter(FormulaValidationResult.formula_id == formula_id)

    if status_filter:
        query = query.filter(FormulaValidationResult.final_status == ValidationStatus(status_filter))

    results = query.order_by(FormulaValidationResult.validation_timestamp.desc()).limit(limit).all()

    return results


@router.post("/overrides", response_model=EthicalOverrideResponse, status_code=status.HTTP_201_CREATED)
async def create_ethical_override(
    override_data: EthicalOverrideCreate,
    current_user = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Create a context-aware ethical override (admin only).
    """
    import uuid

    override = EthicalOverride(
        override_id=f"override_{uuid.uuid4().hex[:8]}",
        override_category=override_data.override_category,
        override_name=override_data.override_name,
        trigger_conditions=override_data.trigger_conditions,
        credibility_adjustment=override_data.credibility_adjustment,
        safety_margin_multiplier=override_data.safety_margin_multiplier,
        applicable_domains=override_data.applicable_domains or [],
        applicable_deployments=override_data.applicable_deployments or [],
        description=override_data.description,
        created_by_user_id=current_user.id
    )

    db.add(override)
    db.commit()
    db.refresh(override)

    return override


@router.get("/overrides", response_model=List[EthicalOverrideResponse])
async def list_ethical_overrides(
    category: Optional[str] = None,
    active_only: bool = True,
    current_user = Depends(get_current_operator),
    db: Session = Depends(get_db)
):
    """
    List ethical overrides.
    """
    query = db.query(EthicalOverride)

    if category:
        query = query.filter(EthicalOverride.override_category == category)

    if active_only:
        query = query.filter(EthicalOverride.is_active == True)

    overrides = query.order_by(EthicalOverride.priority.desc()).all()

    return overrides


@router.post("/overrides/initialize-standard")
async def initialize_standard_overrides(
    current_user = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Initialize standard predefined overrides (admin only).
    """
    created_count = 0

    for override_data in STANDARD_OVERRIDES:
        # Check if already exists
        existing = db.query(EthicalOverride).filter(
            EthicalOverride.override_id == override_data['override_id']
        ).first()

        if existing:
            continue

        override = EthicalOverride(
            **override_data,
            created_by_user_id=current_user.id
        )

        db.add(override)
        created_count += 1

    db.commit()

    return {
        "success": True,
        "created_count": created_count,
        "message": f"Initialized {created_count} standard overrides"
    }


@router.get("/config/{deployment_name}", response_model=EthicalConfigResponse)
async def get_ethical_configuration(
    deployment_name: str,
    current_user = Depends(get_current_operator),
    db: Session = Depends(get_db)
):
    """
    Get ethical configuration for a deployment.
    """
    safeguards = EthicalSafeguards(db, deployment_name)
    return safeguards.config


@router.put("/config/{deployment_name}")
async def update_ethical_configuration(
    deployment_name: str,
    default_tier: Optional[int] = None,
    auto_promotion: Optional[bool] = None,
    safety_margin: Optional[float] = None,
    domain_min_tiers: Optional[dict] = None,
    current_user = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Update ethical configuration for a deployment (admin only).
    """
    config = db.query(EthicalConfiguration).filter(
        EthicalConfiguration.deployment_name == deployment_name
    ).first()

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration for deployment '{deployment_name}' not found"
        )

    if default_tier is not None:
        config.default_tier = default_tier

    if auto_promotion is not None:
        config.auto_promotion_enabled = auto_promotion

    if safety_margin is not None:
        config.safety_margin = safety_margin

    if domain_min_tiers is not None:
        config.domain_min_tiers = domain_min_tiers

    config.configured_by_user_id = current_user.id
    config.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(config)

    return config


@router.get("/audit-logs", response_model=List[EthicalAuditLogResponse])
async def list_ethical_audit_logs(
    decision_type: Optional[str] = None,
    formula_id: Optional[int] = None,
    days: int = 30,
    limit: int = 100,
    current_user = Depends(get_current_operator),
    db: Session = Depends(get_db)
):
    """
    List ethical audit logs.
    """
    start_date = datetime.utcnow() - timedelta(days=days)

    query = db.query(EthicalAuditLog).filter(
        EthicalAuditLog.created_at >= start_date
    )

    if decision_type:
        query = query.filter(EthicalAuditLog.decision_type == decision_type)

    if formula_id:
        query = query.filter(EthicalAuditLog.formula_id == formula_id)

    logs = query.order_by(EthicalAuditLog.created_at.desc()).limit(limit).all()

    return logs


@router.get("/audit-logs/{audit_id}", response_model=EthicalAuditLogResponse)
async def get_ethical_audit_log(
    audit_id: str,
    current_user = Depends(get_current_operator),
    db: Session = Depends(get_db)
):
    """
    Get details of a specific ethical audit log.
    """
    log = db.query(EthicalAuditLog).filter(
        EthicalAuditLog.audit_id == audit_id
    ).first()

    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit log {audit_id} not found"
        )

    return log


@router.post("/check-execution")
async def check_formula_execution(
    formula_id: int,
    context: dict,
    current_user = Depends(get_current_operator),
    db: Session = Depends(get_db)
):
    """
    Check if a formula can be auto-executed or requires approval.
    Returns red line violations, override adjustments, and decision.
    """
    formula = db.query(Formula).filter(Formula.id == formula_id).first()

    if not formula:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Formula {formula_id} not found"
        )

    deployment_name = context.get('deployment', 'default')
    safeguards = EthicalSafeguards(db, deployment_name)

    # Check red lines
    passes_red_lines, violations = safeguards.check_red_lines(formula, context)

    # Apply overrides
    modifications = safeguards.apply_context_overrides(formula, context)

    # Check if can auto-execute
    can_auto, reason = safeguards.can_auto_execute(formula, context)

    return {
        "can_auto_execute": can_auto,
        "reason": reason,
        "passes_red_lines": passes_red_lines,
        "red_line_violations": violations,
        "credibility_modifications": modifications,
        "formula_tier": formula.tier.value if hasattr(formula.tier, 'value') else formula.tier
    }
