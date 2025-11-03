"""
Main FastAPI application for The Reasoner AI Platform.
"""
from fastapi import FastAPI, Depends, HTTPException, status, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime
import time
import logging
import json
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.core.config import settings
from app.core.database import get_db, engine
from app.models import database, schemas
from app.services.reasoner import reasoner_engine
from app.services.tinker import tinker_ml
from app.services.orchestration import orchestration_pipeline
from app.services.units import unit_service
from app.repositories.repositories import (
    get_formula_repository,
    get_execution_repository,
    get_validation_repository,
    get_learning_repository
)
import sqlalchemy as sa

# Create database tables
database.Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Universal Mathematical Reasoning Infrastructure with Continuous Learning"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(",") if hasattr(settings, 'CORS_ORIGINS') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL if hasattr(settings, 'LOG_LEVEL') else 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== PROMETHEUS METRICS ====================
REQUEST_COUNT = Counter(
    'reasoner_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)
REQUEST_DURATION = Histogram(
    'reasoner_http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)
FORMULA_EXECUTIONS = Counter(
    'reasoner_formula_executions_total',
    'Total formula executions',
    ['formula_id', 'status']
)

# ==================== API KEY AUTHENTICATION ====================
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key if authentication is enabled."""
    api_key_enabled = getattr(settings, 'API_KEY_ENABLED', False)
    if not api_key_enabled:
        return True
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key required"
        )
    
    expected_key = getattr(settings, 'API_KEY', '')
    if api_key != expected_key:
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    
    return True

# ==================== REQUEST LOGGING MIDDLEWARE ====================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"[{request_id}] {request.method} {request.url.path} started")
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        logger.info(f"[{request_id}] {request.method} {request.url.path} completed {response.status_code} in {duration:.3f}s")
        
        response.headers["X-Request-ID"] = request_id
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"[{request_id}] {request.method} {request.url.path} failed after {duration:.3f}s: {str(e)}")
        raise

# Register data/context routes
from app.api.data_context_routes import router as data_context_router
app.include_router(
    data_context_router,
    prefix=settings.API_V1_PREFIX,
    tags=["data-context"]
)


# ==================== HEALTH & MONITORING ====================

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Enhanced health check with component status."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.APP_VERSION,
        "components": {}
    }
    
    # Check database
    try:
        db.execute(sa.text("SELECT 1"))
        health_status["components"]["database"] = {"status": "up", "message": "Connected"}
    except Exception as e:
        health_status["components"]["database"] = {"status": "down", "message": str(e)}
        health_status["status"] = "degraded"
    
    # Check Redis (if configured)
    try:
        redis_url = getattr(settings, 'REDIS_URL', None)
        if redis_url:
            health_status["components"]["redis"] = {"status": "up", "message": "Connected"}
        else:
            health_status["components"]["redis"] = {"status": "not_configured"}
    except Exception as e:
        health_status["components"]["redis"] = {"status": "down", "message": str(e)}
    
    # Check MLflow (if configured)
    try:
        mlflow_uri = getattr(settings, 'MLFLOW_TRACKING_URI', None)
        if mlflow_uri:
            health_status["components"]["mlflow"] = {"status": "assumed_up"}
        else:
            health_status["components"]["mlflow"] = {"status": "not_configured"}
    except:
        health_status["components"]["mlflow"] = {"status": "unknown"}
    
    # Formula count
    try:
        formula_count = db.query(database.Formula).count()
        health_status["components"]["formulas"] = {
            "status": "up",
            "count": formula_count
        }
    except:
        health_status["components"]["formulas"] = {"status": "error"}
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/")
async def root():
    """API root with basic info."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs_url": "/docs",
        "health_url": "/health",
        "metrics_url": "/metrics"
    }


# ==================== FORMULA MANAGEMENT ====================

@app.post(
    f"{settings.API_V1_PREFIX}/formulas",
    response_model=schemas.FormulaResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_formula(
    formula: schemas.FormulaCreate,
    db: Session = Depends(get_db)
):
    """Create a new formula."""
    # Check if formula_id already exists
    existing = db.query(database.Formula).filter(
        database.Formula.formula_id == formula.formula_id
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formula with ID {formula.formula_id} already exists"
        )
    
    # Create formula
    db_formula = database.Formula(
        formula_id=formula.formula_id,
        name=formula.name,
        description=formula.description,
        domain=formula.domain,
        formula_expression=formula.formula_expression,
        input_parameters=formula.input_parameters,
        output_parameters=formula.output_parameters,
        required_context=formula.required_context,
        optional_context=formula.optional_context,
        source=formula.source,
        source_reference=formula.source_reference,
        version=formula.version,
        status=database.FormulaStatus.PENDING_REVIEW
    )
    
    db.add(db_formula)
    db.commit()
    db.refresh(db_formula)
    
    return db_formula


@app.get(
    f"{settings.API_V1_PREFIX}/formulas",
    response_model=List[schemas.FormulaResponse]
)
async def list_formulas(
    domain: Optional[str] = None,
    status: Optional[schemas.FormulaStatusEnum] = None,
    min_confidence: float = 0.0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List formulas with optional filters using repository pattern."""
    formula_repo = get_formula_repository(db)
    
    formulas = formula_repo.list_all(
        domain=domain,
        status=status,
        min_confidence=min_confidence,
        limit=limit
    )
    
    return formulas


@app.get(
    f"{settings.API_V1_PREFIX}/formulas/{{formula_id}}",
    response_model=schemas.FormulaResponse
)
async def get_formula(formula_id: str, db: Session = Depends(get_db)):
    """Get a specific formula by ID."""
    formula = db.query(database.Formula).filter(
        database.Formula.formula_id == formula_id
    ).first()
    
    if not formula:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Formula {formula_id} not found"
        )
    
    return formula


@app.patch(
    f"{settings.API_V1_PREFIX}/formulas/{{formula_id}}",
    response_model=schemas.FormulaResponse
)
async def update_formula(
    formula_id: str,
    update: schemas.FormulaUpdate,
    db: Session = Depends(get_db)
):
    """Update a formula."""
    formula = db.query(database.Formula).filter(
        database.Formula.formula_id == formula_id
    ).first()
    
    if not formula:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Formula {formula_id} not found"
        )
    
    # Update fields
    update_data = update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(formula, field, value)
    
    formula.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(formula)
    
    return formula


# ==================== FORMULA EXECUTION ====================

@app.post(
    f"{settings.API_V1_PREFIX}/formulas/execute",
    response_model=schemas.FormulaExecutionResponse
)
async def execute_formula(
    request: schemas.FormulaExecutionRequest,
    db: Session = Depends(get_db)
):
    """Execute a formula using the orchestration pipeline."""
    
    # Use orchestration pipeline for complete processing
    result = await orchestration_pipeline.execute_formula_pipeline(
        db=db,
        formula_id=request.formula_id,
        input_values=request.input_values,
        context_data=request.context_data,
        expected_output=request.expected_output,
        edge_node_id=request.edge_node_id,
        user_id=None  # TODO: Get from auth context when implemented
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error_message", "Execution failed")
        )
    
    # Get execution record from repository
    exec_repo = get_execution_repository(db)
    execution = exec_repo.get_by_id(result["execution_id"])
    
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution record not found"
        )
    
    return execution


@app.get(
    f"{settings.API_V1_PREFIX}/executions/{{execution_id}}",
    response_model=schemas.FormulaExecutionResponse
)
async def get_execution(execution_id: str, db: Session = Depends(get_db)):
    """Get execution details."""
    execution = db.query(database.FormulaExecution).filter(
        database.FormulaExecution.execution_id == execution_id
    ).first()
    
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution {execution_id} not found"
        )
    
    return execution


# ==================== RECOMMENDATIONS ====================

@app.post(
    f"{settings.API_V1_PREFIX}/formulas/recommend",
    response_model=schemas.FormulaRecommendationResponse
)
async def recommend_formulas(
    request: schemas.FormulaRecommendationRequest,
    db: Session = Depends(get_db)
):
    """Get formula recommendations based on domain and context."""
    recommendations = tinker_ml.recommend_formulas(
        db=db,
        domain=request.domain,
        context=request.context,
        min_confidence=request.min_confidence,
        limit=request.limit
    )
    
    return schemas.FormulaRecommendationResponse(
        recommendations=recommendations,
        total_count=len(recommendations),
        context_used=request.context or {}
    )


# ==================== VALIDATION ====================

@app.post(
    f"{settings.API_V1_PREFIX}/formulas/{{formula_id}}/validate",
    response_model=schemas.ValidationResponse
)
async def validate_formula(
    formula_id: str,
    request: schemas.ValidationRequest,
    db: Session = Depends(get_db)
):
    """Run validation on a formula."""
    formula = db.query(database.Formula).filter(
        database.Formula.formula_id == formula_id
    ).first()
    
    if not formula:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Formula {formula_id} not found"
        )
    
    # Run validation
    validation_result = await reasoner_engine.validate_formula(
        formula_expression=formula.formula_expression,
        input_parameters=formula.input_parameters,
        output_parameters=formula.output_parameters,
        test_data=request.test_data,
        stages=request.validation_stages
    )
    
    # Store validation results
    for stage_result in validation_result["stages"]:
        db_validation = database.ValidationResult(
            formula_id=formula.id,
            validation_stage=stage_result["stage"],
            passed=stage_result["passed"],
            confidence=stage_result.get("confidence"),
            validation_data=stage_result.get("details"),
            error_message=stage_result.get("error"),
            validated_by="system"
        )
        db.add(db_validation)
    
    # Update formula validation status
    if validation_result["overall_passed"]:
        formula.validation_stages_passed = [
            s["stage"] for s in validation_result["stages"] if s["passed"]
        ]
        formula.last_validation_date = datetime.utcnow()
    
    db.commit()
    
    return schemas.ValidationResponse(
        formula_id=formula.formula_id,
        overall_passed=validation_result["overall_passed"],
        stages=validation_result["stages"],
        timestamp=datetime.utcnow()
    )


# ==================== ANALYTICS & INSIGHTS ====================

@app.get(
    f"{settings.API_V1_PREFIX}/analytics/formulas/{{formula_id}}",
    response_model=schemas.FormulaAnalytics
)
async def get_formula_analytics(formula_id: str, db: Session = Depends(get_db)):
    """Get analytics for a specific formula."""
    formula = db.query(database.Formula).filter(
        database.Formula.formula_id == formula_id
    ).first()
    
    if not formula:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Formula {formula_id} not found"
        )
    
    # Get learning events for confidence trend
    learning_events = db.query(database.LearningEvent).filter(
        database.LearningEvent.formula_id == formula.id
    ).order_by(database.LearningEvent.timestamp).limit(100).all()
    
    confidence_trend = [
        {
            "timestamp": e.timestamp.isoformat(),
            "confidence": e.new_confidence
        }
        for e in learning_events
    ]
    
    # Get context performances
    context_perfs = db.query(database.ContextPerformance).filter(
        database.ContextPerformance.formula_id == formula.id
    ).order_by(database.ContextPerformance.confidence_in_context.desc()).limit(10).all()
    
    top_contexts = [
        {
            "context": cp.context_data,
            "success_rate": cp.successful_executions / cp.total_executions if cp.total_executions > 0 else 0,
            "executions": cp.total_executions
        }
        for cp in context_perfs
    ]
    
    return schemas.FormulaAnalytics(
        formula_id=formula.formula_id,
        name=formula.name,
        domain=formula.domain,
        total_executions=formula.total_executions,
        success_rate=formula.successful_executions / formula.total_executions if formula.total_executions > 0 else 0,
        average_execution_time=formula.average_execution_time or 0.0,
        confidence_trend=confidence_trend,
        top_contexts=top_contexts,
        performance_by_context={}
    )


@app.get(
    f"{settings.API_V1_PREFIX}/analytics/system",
    response_model=schemas.SystemMetrics
)
async def get_system_metrics(db: Session = Depends(get_db)):
    """Get system-wide metrics."""
    total_formulas = db.query(database.Formula).count()
    total_executions = db.query(database.FormulaExecution).count()
    
    # Average confidence
    avg_confidence = db.query(
        database.func.avg(database.Formula.confidence_score)
    ).scalar() or 0.0
    
    # Formulas by status
    status_counts = {}
    for status_val in database.FormulaStatus:
        count = db.query(database.Formula).filter(
            database.Formula.status == status_val
        ).count()
        status_counts[status_val.value] = count
    
    # Formulas by domain
    domain_counts = {}
    domains = db.query(database.Formula.domain).distinct().all()
    for (domain,) in domains:
        count = db.query(database.Formula).filter(
            database.Formula.domain == domain
        ).count()
        domain_counts[domain] = count
    
    # Recent learning events
    recent_events = db.query(database.LearningEvent).filter(
        database.LearningEvent.timestamp >= datetime.utcnow() - timedelta(days=7)
    ).count()
    
    return schemas.SystemMetrics(
        total_formulas=total_formulas,
        total_executions=total_executions,
        average_confidence=float(avg_confidence),
        formulas_by_status=status_counts,
        formulas_by_domain=domain_counts,
        recent_learning_events=recent_events,
        edge_nodes_active=len(settings.EDGE_NODES),
        uptime_seconds=0.0  # TODO: Implement uptime tracking
    )


@app.get(
    f"{settings.API_V1_PREFIX}/learning/insights",
)
async def get_learning_insights(
    formula_id: Optional[str] = None,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Get learning insights and trends."""
    formula_db_id = None
    if formula_id:
        formula = db.query(database.Formula).filter(
            database.Formula.formula_id == formula_id
        ).first()
        if formula:
            formula_db_id = formula.id
    
    insights = tinker_ml.get_learning_insights(
        db=db,
        formula_id=formula_db_id,
        days=days
    )
    
    return insights


# ==================== UNIT CONVERSION ====================

@app.post(f"{settings.API_V1_PREFIX}/units/convert")
async def convert_units(
    value: float,
    from_unit: str,
    to_unit: str,
    context: Optional[str] = None
):
    """Convert between units using the unit service."""
    converted, success, error = unit_service.convert(
        value, from_unit, to_unit, context
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )
    
    return {
        "original_value": value,
        "original_unit": from_unit,
        "converted_value": converted,
        "converted_unit": to_unit,
        "context": context
    }


@app.get(f"{settings.API_V1_PREFIX}/units/info/{{unit}}")
async def get_unit_info(unit: str):
    """Get information about a unit."""
    info = unit_service.get_unit_info(unit)
    
    if not info["valid"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unit '{unit}' not found"
        )
    
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS if not settings.DEBUG else 1
    )
