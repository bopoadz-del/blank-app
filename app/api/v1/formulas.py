"""Formula execution endpoints"""

from fastapi import APIRouter, Depends, Request
from typing import List
import time

from app.schemas.formula import (
    FormulaExecuteRequest,
    FormulaExecuteResponse,
    FormulaInfo
)
from app.services.formula_service import formula_service
from app.core.security import verify_api_key
from app.core.rate_limit import rate_limiter

router = APIRouter()


@router.post("/execute", response_model=FormulaExecuteResponse)
async def execute_formula(
    request: Request,
    payload: FormulaExecuteRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Execute a formula with given input values

    Requires valid API key in X-API-Key header.
    Rate limited to 10 requests per minute.
    """
    # Check rate limit
    await rate_limiter.check_rate_limit(request, api_key)

    start_time = time.time()

    try:
        result, unit = formula_service.execute(
            payload.formula_id,
            payload.input_values
        )

        execution_time = (time.time() - start_time) * 1000

        return FormulaExecuteResponse(
            success=True,
            formula_id=payload.formula_id,
            result=result,
            unit=unit,
            error=None,
            execution_time_ms=execution_time
        )

    except ValueError as e:
        execution_time = (time.time() - start_time) * 1000

        return FormulaExecuteResponse(
            success=False,
            formula_id=payload.formula_id,
            result=None,
            unit=None,
            error=str(e),
            execution_time_ms=execution_time
        )


@router.get("/list", response_model=List[FormulaInfo])
async def list_formulas(api_key: str = Depends(verify_api_key)):
    """
    List all available formulas

    Requires valid API key in X-API-Key header.
    """
    return formula_service.list_formulas()


@router.get("/{formula_id}", response_model=FormulaInfo)
async def get_formula_info(
    formula_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get information about a specific formula

    Requires valid API key in X-API-Key header.
    """
    try:
        return formula_service.get_formula_info(formula_id)
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=str(e))
