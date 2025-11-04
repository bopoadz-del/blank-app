"""
Orchestration Pipeline
Coordinates the flow: Input → Validation → Execution → Learning → Response
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib
import json
from loguru import logger
from sqlalchemy.orm import Session

from app.services.reasoner import reasoner_engine
from app.services.tinker import tinker_ml
from app.services.units import unit_service
from app.services.safety_pipeline import SafetyPipeline
from app.repositories.repositories import (
    get_formula_repository,
    get_execution_repository,
    get_validation_repository,
    get_learning_repository
)
from app.models.database import FormulaExecution, LearningEvent
from app.models.schemas import ExecutionStatus


class OrchestrationPipeline:
    """
    Coordinates the complete formula processing pipeline
    
    Flow:
    1. Context Detection & Formula Selection
    2. Input Validation & Unit Conversion
    3. Formula Execution
    4. Result Validation
    5. Confidence Update (Learning)
    6. Logging & Audit Trail
    """
    
    def __init__(self):
        self.reasoner = reasoner_engine
        self.tinker = tinker_ml
        self.units = unit_service
        
    async def execute_formula_pipeline(
        self,
        db: Session,
        formula_id: str,
        input_values: Dict[str, Any],
        context_data: Optional[Dict[str, Any]] = None,
        expected_output: Optional[Dict[str, Any]] = None,
        edge_node_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute complete formula processing pipeline
        
        Args:
            db: Database session
            formula_id: Formula identifier
            input_values: Input parameter values
            context_data: Optional context (climate, site conditions, etc.)
            expected_output: Optional expected values for validation
            edge_node_id: Optional edge node identifier
            user_id: Optional user identifier
            
        Returns:
            Dict with execution results and metadata
        """
        pipeline_start = datetime.utcnow()

        try:
            # ==================== STAGE 0: SAFETY CHECK ====================
            # CRITICAL: Non-negotiable safety screening BEFORE any processing
            from app.models.auth import User

            user = None
            if user_id:
                user = db.query(User).filter(User.id == user_id).first()

            # Prepare request content for safety check
            request_content = json.dumps({
                "formula_id": formula_id,
                "input_values": input_values,
                "context_data": context_data
            })

            # Execute safety pipeline
            safety_pipeline = SafetyPipeline(db, deployment_name="default")
            is_safe, safety_incident = safety_pipeline.execute_pipeline(
                request_content=request_content,
                user=user,
                context={
                    "formula_id": formula_id,
                    "edge_node_id": edge_node_id,
                    "ip_address": None,  # Would be populated from request context
                    "user_agent": None
                }
            )

            if not is_safe:
                logger.critical(f"SAFETY VIOLATION: Request blocked by safety layer. Incident: {safety_incident.incident_id}")
                return self._error_response(
                    "safety_violation",
                    safety_incident.user_message or "Request blocked for safety reasons"
                )

            logger.info("Safety check passed - proceeding with execution")

            # ==================== STAGE 1: LOAD FORMULA ====================
            formula_repo = get_formula_repository(db)
            formula = formula_repo.get_by_id(formula_id)
            
            if not formula:
                return self._error_response(
                    "formula_not_found",
                    f"Formula {formula_id} not found"
                )
            
            logger.info(f"Pipeline: Loaded formula {formula.name}")
            
            # ==================== STAGE 2: VALIDATE INPUTS ====================
            validation_result = await self._validate_inputs(
                formula,
                input_values,
                context_data
            )
            
            if not validation_result["valid"]:
                return self._error_response(
                    "validation_failed",
                    validation_result["errors"]
                )
            
            logger.info("Pipeline: Inputs validated")
            
            # ==================== STAGE 3: UNIT CONVERSIONS ====================
            converted_inputs = await self._convert_units(
                formula,
                input_values
            )
            
            logger.info("Pipeline: Units converted")
            
            # ==================== STAGE 4: EXECUTE FORMULA ====================
            execution_result = await self.reasoner.execute_formula(
                formula_expression=formula.formula_expression,
                input_values=converted_inputs,
                context=context_data
            )
            
            if not execution_result["success"]:
                return self._error_response(
                    "execution_failed",
                    execution_result.get("error", "Unknown error")
                )
            
            logger.info(f"Pipeline: Formula executed successfully")
            
            # ==================== STAGE 5: VALIDATE RESULT ====================
            result_validation = await self._validate_result(
                formula,
                execution_result["result"],
                expected_output
            )
            
            # ==================== STAGE 6: LOG EXECUTION ====================
            exec_repo = get_execution_repository(db)
            execution_record = await self._log_execution(
                exec_repo,
                formula,
                input_values,
                execution_result,
                context_data,
                expected_output,
                result_validation,
                edge_node_id,
                user_id
            )
            
            logger.info(f"Pipeline: Execution logged (ID: {execution_record.execution_id})")
            
            # ==================== STAGE 7: UPDATE CONFIDENCE ====================
            learning_result = await self._update_confidence(
                db,
                formula,
                execution_record,
                result_validation
            )
            
            logger.info(f"Pipeline: Confidence updated to {formula.confidence_score:.3f}")
            
            # ==================== STAGE 8: CONTEXT LEARNING ====================
            if context_data:
                await self._update_context_performance(
                    db,
                    formula,
                    context_data,
                    execution_result["success"]
                )
            
            # ==================== RETURN RESULTS ====================
            pipeline_duration = (datetime.utcnow() - pipeline_start).total_seconds()
            
            return {
                "success": True,
                "execution_id": execution_record.execution_id,
                "formula_id": formula.formula_id,
                "formula_name": formula.name,
                "result": execution_result["result"],
                "output_unit": formula.output_parameters.get("unit") if formula.output_parameters else None,
                "confidence_score": formula.confidence_score,
                "validation": result_validation,
                "execution_time_ms": execution_result.get("execution_time"),
                "pipeline_time_ms": pipeline_duration * 1000,
                "learning": {
                    "confidence_change": learning_result.get("confidence_change", 0),
                    "total_executions": formula.total_executions
                },
                "metadata": {
                    "edge_node_id": edge_node_id,
                    "user_id": user_id,
                    "context_hash": self._hash_context(context_data) if context_data else None
                }
            }
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            return self._error_response("pipeline_error", str(e))
    
    async def _validate_inputs(
        self,
        formula,
        input_values: Dict[str, Any],
        context_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate input parameters"""
        errors = []
        
        # Check required parameters
        required_params = formula.input_parameters.keys()
        missing = set(required_params) - set(input_values.keys())
        
        if missing:
            errors.append(f"Missing required parameters: {missing}")
        
        # Check parameter types and ranges
        for param, value in input_values.items():
            if param in formula.input_parameters:
                param_spec = formula.input_parameters[param]
                
                # Type check
                expected_type = param_spec.get("type", "float")
                if expected_type == "float" and not isinstance(value, (int, float)):
                    errors.append(f"Parameter {param}: expected number, got {type(value)}")
                
                # Range check
                if isinstance(value, (int, float)):
                    min_val = param_spec.get("min_value")
                    max_val = param_spec.get("max_value")
                    
                    if min_val is not None and value < min_val:
                        errors.append(f"Parameter {param}: value {value} below minimum {min_val}")
                    if max_val is not None and value > max_val:
                        errors.append(f"Parameter {param}: value {value} above maximum {max_val}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def _convert_units(
        self,
        formula,
        input_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert input units to formula's expected units"""
        converted = {}
        
        for param, value in input_values.items():
            # If value is dict with unit, convert it
            if isinstance(value, dict) and "value" in value and "unit" in value:
                param_spec = formula.input_parameters.get(param, {})
                target_unit = param_spec.get("unit")
                
                if target_unit:
                    converted_value, success, error = self.units.convert(
                        value["value"],
                        value["unit"],
                        target_unit
                    )
                    if success:
                        converted[param] = converted_value
                    else:
                        logger.warning(f"Unit conversion failed for {param}: {error}")
                        converted[param] = value["value"]
                else:
                    converted[param] = value["value"]
            else:
                # Direct value, no conversion needed
                converted[param] = value
        
        return converted
    
    async def _validate_result(
        self,
        formula,
        result: float,
        expected_output: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate execution result"""
        validation = {
            "passed": True,
            "warnings": [],
            "error_magnitude": None
        }
        
        # Check against output parameter constraints
        if formula.output_parameters:
            output_spec = list(formula.output_parameters.values())[0]
            min_val = output_spec.get("min_value")
            max_val = output_spec.get("max_value")
            
            if min_val is not None and result < min_val:
                validation["warnings"].append(f"Result {result} below minimum {min_val}")
                validation["passed"] = False
            if max_val is not None and result > max_val:
                validation["warnings"].append(f"Result {result} above maximum {max_val}")
                validation["passed"] = False
        
        # Compare with expected output if provided
        if expected_output:
            expected_value = list(expected_output.values())[0]
            if expected_value:
                error = abs(result - expected_value) / abs(expected_value) if expected_value != 0 else abs(result)
                validation["error_magnitude"] = error
                
                if error > 0.05:  # 5% threshold
                    validation["warnings"].append(f"Result differs from expected by {error*100:.1f}%")
        
        return validation
    
    async def _log_execution(
        self,
        exec_repo,
        formula,
        input_values,
        execution_result,
        context_data,
        expected_output,
        validation,
        edge_node_id,
        user_id
    ) -> FormulaExecution:
        """Log execution to database"""
        execution = FormulaExecution(
            formula_id=formula.id,
            input_values=input_values,
            output_values={"result": execution_result["result"]},
            context_data=context_data,
            status=ExecutionStatus.COMPLETED if execution_result["success"] else ExecutionStatus.FAILED,
            execution_time=execution_result.get("execution_time"),
            error_message=execution_result.get("error"),
            expected_output=expected_output,
            actual_vs_expected_error=validation.get("error_magnitude"),
            validation_passed=validation["passed"],
            edge_node_id=edge_node_id,
            executed_by=user_id
        )
        
        return exec_repo.create(execution)
    
    async def _update_confidence(
        self,
        db: Session,
        formula,
        execution_record,
        validation
    ) -> Dict[str, Any]:
        """Update formula confidence based on execution"""
        old_confidence = formula.confidence_score
        
        await self.tinker.update_confidence_from_execution(
            db=db,
            formula_id=formula.id,
            execution_success=execution_record.status == ExecutionStatus.COMPLETED,
            context=execution_record.context_data,
            error_magnitude=validation.get("error_magnitude")
        )
        
        # Refresh to get updated confidence
        db.refresh(formula)
        
        return {
            "confidence_change": formula.confidence_score - old_confidence,
            "new_confidence": formula.confidence_score,
            "old_confidence": old_confidence
        }
    
    async def _update_context_performance(
        self,
        db: Session,
        formula,
        context_data: Dict[str, Any],
        success: bool
    ):
        """Update context-specific performance"""
        learning_repo = get_learning_repository(db)
        context_hash = self._hash_context(context_data)
        
        learning_repo.update_context_performance(
            formula_id=formula.id,
            context_hash=context_hash,
            context_data=context_data,
            success=success
        )
    
    def _hash_context(self, context_data: Dict[str, Any]) -> str:
        """Generate hash for context data"""
        if not context_data:
            return "null_context"
        
        # Sort keys for consistent hashing
        sorted_context = json.dumps(context_data, sort_keys=True)
        return hashlib.md5(sorted_context.encode()).hexdigest()[:16]
    
    def _error_response(self, error_type: str, message: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            "success": False,
            "error_type": error_type,
            "error_message": message,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global instance
orchestration_pipeline = OrchestrationPipeline()
