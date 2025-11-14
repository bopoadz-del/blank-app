"""Formula execution endpoints"""

from fastapi import APIRouter, Depends, Request
from typing import List
from sqlalchemy.orm import Session
import time

from app.schemas.formula import (
    FormulaExecuteRequest,
    FormulaExecuteResponse,
    FormulaInfo
)
from app.services.formula_service import formula_service
from app.core.rate_limit import rate_limiter
from app.database.session import get_db
from app.models.formula_execution import FormulaExecution
from app.utils.unit_converter import unit_converter
from app.utils.mlflow_tracker import mlflow_tracker

router = APIRouter()


@router.post("/execute", response_model=FormulaExecuteResponse)
async def execute_formula(
    request: Request,
    payload: FormulaExecuteRequest,
    db: Session = Depends(get_db)
):
    """
    Execute a formula with given input values

    Features:
    - No authentication required
    - Rate limited to 10 requests per minute
    - Saves execution to database
    - Tracks execution in MLflow
    - Supports unit conversion
    """
    # Check rate limit with a default identifier
    await rate_limiter.check_rate_limit(request, "public")

    start_time = time.time()
    result = None
    unit = None
    original_unit = None
    error_msg = None
    success = False
    mlflow_run_id = None

    try:
        # Execute formula
        result, unit = formula_service.execute(
            payload.formula_id,
            payload.input_values
        )
        original_unit = unit

        # Convert units if requested
        if payload.convert_to_unit and payload.convert_to_unit != unit:
            result, unit = unit_converter.convert(result, unit, payload.convert_to_unit)

        success = True

    except ValueError as e:
        error_msg = str(e)
        success = False

    execution_time = (time.time() - start_time) * 1000

    # Log to MLflow
    mlflow_run_id = mlflow_tracker.log_execution(
        formula_id=payload.formula_id,
        input_values=payload.input_values,
        result=result,
        unit=unit,
        execution_time_ms=execution_time,
        success=success,
        error=error_msg
    )

    # Save to database
    db_execution = FormulaExecution(
        formula_id=payload.formula_id,
        input_values=payload.input_values,
        result=result,
        unit=unit,
        success=success,
        error=error_msg,
        execution_time_ms=execution_time,
        api_key_hash="public",  # No authentication required
        mlflow_run_id=mlflow_run_id
    )
    db.add(db_execution)
    db.commit()
    db.refresh(db_execution)

    return FormulaExecuteResponse(
        success=success,
        formula_id=payload.formula_id,
        result=result,
        unit=unit,
        original_unit=original_unit if original_unit != unit else None,
        error=error_msg,
        execution_time_ms=execution_time,
        execution_id=db_execution.id,
        mlflow_run_id=mlflow_run_id
    )


@router.get("/list", response_model=List[FormulaInfo])
async def list_formulas():
    """
    List all available formulas

    No authentication required.
    """
    return formula_service.list_formulas()


@router.get("/{formula_id}", response_model=FormulaInfo)
async def get_formula_info(formula_id: str):
    """
    Get information about a specific formula

    No authentication required.
    """
    try:
        return formula_service.get_formula_info(formula_id)
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/history/recent", response_model=List[dict])
async def get_recent_executions(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get recent formula executions

    Args:
        limit: Maximum number of executions to return (default 10)

    Returns:
        List of recent executions
    """
    executions = db.query(FormulaExecution)\
        .order_by(FormulaExecution.created_at.desc())\
        .limit(limit)\
        .all()

    return [
        {
            "id": ex.id,
            "formula_id": ex.formula_id,
            "result": ex.result,
            "unit": ex.unit,
            "success": ex.success,
            "execution_time_ms": ex.execution_time_ms,
            "created_at": ex.created_at
        }
        for ex in executions
    ]
