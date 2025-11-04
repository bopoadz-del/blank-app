"""
Safety Layer API Routes.
Provides endpoints for safety incident management, monitoring, and emergency protocols.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from ..database import get_db
from ..models.safety_layer import (
    SafetyIncident,
    SafetyPattern,
    UserSafetyScore,
    SafetyConfiguration,
    EmergencyProtocol,
    ProhibitedCategory,
    SafetyAction,
    AlertLevel
)
from ..models.auth import User
from ..services.safety_pipeline import SafetyPipeline
from .auth_routes import get_current_user

router = APIRouter(prefix="/api/v1/safety", tags=["safety"])


# ============================================================================
# Request/Response Models
# ============================================================================

class SafetyCheckRequest(BaseModel):
    """Request to check content safety."""
    content: str
    context: Optional[dict] = {}
    formula_id: Optional[int] = None


class SafetyCheckResponse(BaseModel):
    """Response from safety check."""
    is_safe: bool
    incident_id: Optional[str] = None
    alert_level: Optional[str] = None
    user_message: Optional[str] = None
    detected_category: Optional[str] = None


class SafetyIncidentResponse(BaseModel):
    """Safety incident details."""
    incident_id: str
    prohibited_category: Optional[str]
    safety_action: str
    alert_level: str
    detected_at_stage: str
    detection_confidence: float
    matched_keywords: List[str]
    user_id: Optional[int]
    incident_timestamp: datetime
    investigation_status: str
    resolved_at: Optional[datetime]

    class Config:
        from_attributes = True


class UserSafetyScoreResponse(BaseModel):
    """User safety score details."""
    user_id: int
    safety_score: float
    risk_level: str
    total_requests: int
    blocked_requests: int
    warnings_issued: int
    safety_incidents: int
    account_status: str
    last_incident_at: Optional[datetime]

    class Config:
        from_attributes = True


class SafetyPatternCreate(BaseModel):
    """Create new safety pattern."""
    pattern_name: str
    pattern_category: str
    pattern_description: str
    detection_rules: dict
    time_window_minutes: int = 60
    threshold_count: int = 3
    risk_score: float = 0.5
    action_if_matched: str = "block_and_investigate"
    alert_level_if_matched: str = "high"


class SafetyPatternResponse(BaseModel):
    """Safety pattern details."""
    pattern_id: str
    pattern_name: str
    pattern_category: str
    detection_count: int
    false_positive_count: int
    true_positive_count: int
    is_active: bool
    last_triggered: Optional[datetime]

    class Config:
        from_attributes = True


class EmergencyProtocolResponse(BaseModel):
    """Emergency protocol details."""
    protocol_id: str
    prohibited_category: str
    protocol_name: str
    immediate_actions: List[str]
    notify_law_enforcement: bool
    notify_security_team: bool
    account_action: str
    is_active: bool

    class Config:
        from_attributes = True


class SafetyConfigurationResponse(BaseModel):
    """Safety configuration details."""
    deployment_name: str
    strict_mode: bool
    compliance_framework: Optional[str]
    additional_blocks: dict
    reporting_channels: dict
    multi_factor_required: bool

    class Config:
        from_attributes = True


# ============================================================================
# Safety Check Endpoints
# ============================================================================

@router.post("/check", response_model=SafetyCheckResponse)
async def check_content_safety(
    check_request: SafetyCheckRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Check if content passes safety screening.
    Runs through multi-stage safety pipeline.
    """
    pipeline = SafetyPipeline(db, deployment_name="default")

    context = check_request.context or {}
    context['formula_id'] = check_request.formula_id

    is_safe, incident = pipeline.execute_pipeline(
        request_content=check_request.content,
        user=current_user,
        context=context
    )

    if not is_safe and incident:
        return SafetyCheckResponse(
            is_safe=False,
            incident_id=incident.incident_id,
            alert_level=incident.alert_level.value,
            user_message=incident.user_message,
            detected_category=incident.prohibited_category.value if incident.prohibited_category else None
        )

    return SafetyCheckResponse(is_safe=True)


# ============================================================================
# Incident Management
# ============================================================================

@router.get("/incidents", response_model=List[SafetyIncidentResponse])
async def list_safety_incidents(
    skip: int = 0,
    limit: int = 50,
    alert_level: Optional[str] = None,
    category: Optional[str] = None,
    investigation_status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List safety incidents.
    Admin only: View all incidents.
    Users: View their own incidents only.
    """
    query = db.query(SafetyIncident)

    # Non-admin users can only see their own incidents
    if current_user.role not in ["admin", "auditor"]:
        query = query.filter(SafetyIncident.user_id == current_user.id)

    # Filters
    if alert_level:
        try:
            query = query.filter(SafetyIncident.alert_level == AlertLevel(alert_level))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid alert level: {alert_level}")

    if category:
        try:
            query = query.filter(SafetyIncident.prohibited_category == ProhibitedCategory(category))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

    if investigation_status:
        query = query.filter(SafetyIncident.investigation_status == investigation_status)

    incidents = query.order_by(SafetyIncident.incident_timestamp.desc()).offset(skip).limit(limit).all()

    return [
        SafetyIncidentResponse(
            incident_id=i.incident_id,
            prohibited_category=i.prohibited_category.value if i.prohibited_category else None,
            safety_action=i.safety_action.value,
            alert_level=i.alert_level.value,
            detected_at_stage=i.detected_at_stage,
            detection_confidence=i.detection_confidence,
            matched_keywords=i.matched_keywords,
            user_id=i.user_id,
            incident_timestamp=i.incident_timestamp,
            investigation_status=i.investigation_status,
            resolved_at=i.resolved_at
        )
        for i in incidents
    ]


@router.get("/incidents/{incident_id}", response_model=SafetyIncidentResponse)
async def get_safety_incident(
    incident_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get specific safety incident details."""
    incident = db.query(SafetyIncident).filter(
        SafetyIncident.incident_id == incident_id
    ).first()

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    # Permission check
    if current_user.role not in ["admin", "auditor"] and incident.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view this incident")

    return SafetyIncidentResponse(
        incident_id=incident.incident_id,
        prohibited_category=incident.prohibited_category.value if incident.prohibited_category else None,
        safety_action=incident.safety_action.value,
        alert_level=incident.alert_level.value,
        detected_at_stage=incident.detected_at_stage,
        detection_confidence=incident.detection_confidence,
        matched_keywords=incident.matched_keywords,
        user_id=incident.user_id,
        incident_timestamp=incident.incident_timestamp,
        investigation_status=incident.investigation_status,
        resolved_at=incident.resolved_at
    )


@router.put("/incidents/{incident_id}/resolve")
async def resolve_safety_incident(
    incident_id: str,
    resolution_notes: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Resolve a safety incident.
    Admin only.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    incident = db.query(SafetyIncident).filter(
        SafetyIncident.incident_id == incident_id
    ).first()

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    incident.investigation_status = "resolved"
    incident.resolved_at = datetime.utcnow()
    incident.resolved_by_user_id = current_user.id
    incident.resolution_notes = resolution_notes

    db.commit()

    return {"status": "resolved", "incident_id": incident_id}


# ============================================================================
# User Safety Scores
# ============================================================================

@router.get("/users/{user_id}/safety-score", response_model=UserSafetyScoreResponse)
async def get_user_safety_score(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get user safety score.
    Users can view their own score.
    Admin/auditor can view any user's score.
    """
    if current_user.role not in ["admin", "auditor"] and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    safety_score = db.query(UserSafetyScore).filter(
        UserSafetyScore.user_id == user_id
    ).first()

    if not safety_score:
        # Return default safe score
        return UserSafetyScoreResponse(
            user_id=user_id,
            safety_score=1.0,
            risk_level="low",
            total_requests=0,
            blocked_requests=0,
            warnings_issued=0,
            safety_incidents=0,
            account_status="active",
            last_incident_at=None
        )

    return UserSafetyScoreResponse(
        user_id=safety_score.user_id,
        safety_score=safety_score.safety_score,
        risk_level=safety_score.risk_level,
        total_requests=safety_score.total_requests,
        blocked_requests=safety_score.blocked_requests,
        warnings_issued=safety_score.warnings_issued,
        safety_incidents=safety_score.safety_incidents,
        account_status=safety_score.account_status,
        last_incident_at=safety_score.last_incident_at
    )


@router.get("/users/high-risk", response_model=List[UserSafetyScoreResponse])
async def list_high_risk_users(
    min_risk_level: str = "high",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List high-risk users.
    Admin/auditor only.
    """
    if current_user.role not in ["admin", "auditor"]:
        raise HTTPException(status_code=403, detail="Admin or auditor access required")

    query = db.query(UserSafetyScore)

    if min_risk_level == "critical":
        query = query.filter(UserSafetyScore.risk_level == "critical")
    elif min_risk_level == "high":
        query = query.filter(UserSafetyScore.risk_level.in_(["high", "critical"]))
    elif min_risk_level == "medium":
        query = query.filter(UserSafetyScore.risk_level.in_(["medium", "high", "critical"]))

    users = query.order_by(UserSafetyScore.safety_score.asc()).all()

    return [
        UserSafetyScoreResponse(
            user_id=u.user_id,
            safety_score=u.safety_score,
            risk_level=u.risk_level,
            total_requests=u.total_requests,
            blocked_requests=u.blocked_requests,
            warnings_issued=u.warnings_issued,
            safety_incidents=u.safety_incidents,
            account_status=u.account_status,
            last_incident_at=u.last_incident_at
        )
        for u in users
    ]


# ============================================================================
# Safety Patterns
# ============================================================================

@router.get("/patterns", response_model=List[SafetyPatternResponse])
async def list_safety_patterns(
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List safety patterns.
    Admin only.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    query = db.query(SafetyPattern)

    if is_active is not None:
        query = query.filter(SafetyPattern.is_active == is_active)

    patterns = query.order_by(SafetyPattern.detection_count.desc()).all()

    return [
        SafetyPatternResponse(
            pattern_id=p.pattern_id,
            pattern_name=p.pattern_name,
            pattern_category=p.pattern_category.value,
            detection_count=p.detection_count,
            false_positive_count=p.false_positive_count,
            true_positive_count=p.true_positive_count,
            is_active=p.is_active,
            last_triggered=p.last_triggered
        )
        for p in patterns
    ]


@router.post("/patterns", response_model=SafetyPatternResponse)
async def create_safety_pattern(
    pattern_data: SafetyPatternCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create new safety pattern.
    Admin only.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    import uuid

    pattern = SafetyPattern(
        pattern_id=f"pattern_{uuid.uuid4().hex[:12]}",
        pattern_name=pattern_data.pattern_name,
        pattern_category=ProhibitedCategory(pattern_data.pattern_category),
        pattern_description=pattern_data.pattern_description,
        detection_rules=pattern_data.detection_rules,
        time_window_minutes=pattern_data.time_window_minutes,
        threshold_count=pattern_data.threshold_count,
        risk_score=pattern_data.risk_score,
        action_if_matched=SafetyAction(pattern_data.action_if_matched),
        alert_level_if_matched=AlertLevel(pattern_data.alert_level_if_matched),
        is_active=True,
        created_by_user_id=current_user.id
    )

    db.add(pattern)
    db.commit()
    db.refresh(pattern)

    return SafetyPatternResponse(
        pattern_id=pattern.pattern_id,
        pattern_name=pattern.pattern_name,
        pattern_category=pattern.pattern_category.value,
        detection_count=pattern.detection_count,
        false_positive_count=pattern.false_positive_count,
        true_positive_count=pattern.true_positive_count,
        is_active=pattern.is_active,
        last_triggered=pattern.last_triggered
    )


# ============================================================================
# Emergency Protocols
# ============================================================================

@router.get("/protocols", response_model=List[EmergencyProtocolResponse])
async def list_emergency_protocols(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List emergency protocols.
    Admin only.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    protocols = db.query(EmergencyProtocol).filter(
        EmergencyProtocol.is_active == True
    ).all()

    return [
        EmergencyProtocolResponse(
            protocol_id=p.protocol_id,
            prohibited_category=p.prohibited_category.value,
            protocol_name=p.protocol_name,
            immediate_actions=p.immediate_actions,
            notify_law_enforcement=p.notify_law_enforcement,
            notify_security_team=p.notify_security_team,
            account_action=p.account_action,
            is_active=p.is_active
        )
        for p in protocols
    ]


# ============================================================================
# Safety Configuration
# ============================================================================

@router.get("/config", response_model=SafetyConfigurationResponse)
async def get_safety_configuration(
    deployment_name: str = "default",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get safety configuration for deployment.
    Admin only.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    config = db.query(SafetyConfiguration).filter(
        SafetyConfiguration.deployment_name == deployment_name
    ).first()

    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")

    return SafetyConfigurationResponse(
        deployment_name=config.deployment_name,
        strict_mode=config.strict_mode,
        compliance_framework=config.compliance_framework,
        additional_blocks=config.additional_blocks or {},
        reporting_channels=config.reporting_channels or {},
        multi_factor_required=config.multi_factor_required
    )


@router.put("/config/{deployment_name}/additional-blocks")
async def update_additional_blocks(
    deployment_name: str,
    additional_blocks: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update additional regional blocks.
    Admin only.
    Example: {"alcohol": "TOTAL_BLOCK", "gambling": "TOTAL_BLOCK"} for KSA
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    config = db.query(SafetyConfiguration).filter(
        SafetyConfiguration.deployment_name == deployment_name
    ).first()

    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")

    config.additional_blocks = additional_blocks
    config.updated_at = datetime.utcnow()

    db.commit()

    return {"status": "updated", "additional_blocks": additional_blocks}


# ============================================================================
# Statistics
# ============================================================================

@router.get("/stats/incidents")
async def get_incident_statistics(
    time_range_hours: int = 24,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get incident statistics.
    Admin/auditor only.
    """
    if current_user.role not in ["admin", "auditor"]:
        raise HTTPException(status_code=403, detail="Admin or auditor access required")

    time_threshold = datetime.utcnow() - timedelta(hours=time_range_hours)

    total_incidents = db.query(SafetyIncident).filter(
        SafetyIncident.incident_timestamp >= time_threshold
    ).count()

    critical_incidents = db.query(SafetyIncident).filter(
        SafetyIncident.incident_timestamp >= time_threshold,
        SafetyIncident.alert_level == AlertLevel.CRITICAL
    ).count()

    # Incidents by category
    from sqlalchemy import func
    category_counts = db.query(
        SafetyIncident.prohibited_category,
        func.count(SafetyIncident.id).label('count')
    ).filter(
        SafetyIncident.incident_timestamp >= time_threshold
    ).group_by(SafetyIncident.prohibited_category).all()

    return {
        "time_range_hours": time_range_hours,
        "total_incidents": total_incidents,
        "critical_incidents": critical_incidents,
        "incidents_by_category": {
            str(cat): count for cat, count in category_counts if cat is not None
        }
    }
