"""Formula execution schemas"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime


class FormulaExecuteRequest(BaseModel):
    """Request schema for formula execution"""

    formula_id: str = Field(..., description="ID of the formula to execute")
    input_values: Dict[str, float] = Field(..., description="Input parameters for the formula")

    class Config:
        json_schema_extra = {
            "example": {
                "formula_id": "beam_deflection_simply_supported",
                "input_values": {
                    "w": 10.0,
                    "L": 5.0,
                    "E": 200.0,
                    "I": 0.0001
                }
            }
        }


class FormulaExecuteResponse(BaseModel):
    """Response schema for formula execution"""

    success: bool = Field(..., description="Execution status")
    formula_id: str = Field(..., description="ID of the executed formula")
    result: Optional[float] = Field(None, description="Calculation result")
    unit: Optional[str] = Field(None, description="Unit of the result")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Execution timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "formula_id": "beam_deflection_simply_supported",
                "result": 0.00065104,
                "unit": "m",
                "error": None,
                "execution_time_ms": 1.23,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class FormulaInfo(BaseModel):
    """Schema for formula information"""

    formula_id: str
    name: str
    description: str
    parameters: Dict[str, str]
    unit: str
    category: str

    class Config:
        json_schema_extra = {
            "example": {
                "formula_id": "beam_deflection_simply_supported",
                "name": "Simply Supported Beam Deflection",
                "description": "Calculate maximum deflection of a simply supported beam with uniform load",
                "parameters": {
                    "w": "Uniform load (N/m)",
                    "L": "Beam length (m)",
                    "E": "Young's modulus (GPa)",
                    "I": "Second moment of area (m^4)"
                },
                "unit": "m",
                "category": "Structural Engineering"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
