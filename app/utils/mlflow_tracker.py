"""MLflow integration for experiment tracking with optional dependency."""

import os
from typing import Dict, Any, Optional

from app.core.config import settings

try:
    import mlflow  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    mlflow = None


class MLflowTracker:
    """MLflow integration for tracking formula executions"""

    def __init__(self):
        self.enabled = mlflow is not None and os.getenv("MLFLOW_TRACKING_URI") is not None
        if self.enabled:
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
            mlflow.set_experiment(settings.PROJECT_NAME)

    def log_execution(
        self,
        formula_id: str,
        input_values: Dict[str, float],
        result: Optional[float],
        unit: Optional[str],
        execution_time_ms: float,
        success: bool,
        error: Optional[str] = None
    ) -> Optional[str]:
        """
        Log formula execution to MLflow

        Args:
            formula_id: ID of the executed formula
            input_values: Input parameters
            result: Calculation result
            unit: Unit of result
            execution_time_ms: Execution time in milliseconds
            success: Whether execution was successful
            error: Error message if failed

        Returns:
            MLflow run ID if tracking is enabled, None otherwise
        """
        if not self.enabled:
            return None

        try:
            with mlflow.start_run(run_name=f"{formula_id}_execution"):
                # Log parameters
                for key, value in input_values.items():
                    mlflow.log_param(key, value)

                mlflow.log_param("formula_id", formula_id)

                # Log metrics
                if result is not None:
                    mlflow.log_metric("result", result)
                mlflow.log_metric("execution_time_ms", execution_time_ms)
                mlflow.log_metric("success", 1.0 if success else 0.0)

                # Log tags
                mlflow.set_tag("formula_id", formula_id)
                mlflow.set_tag("unit", unit or "N/A")
                if error:
                    mlflow.set_tag("error", error)

                run = mlflow.active_run()
                return run.info.run_id if run else None

        except Exception as e:
            # Don't fail the request if MLflow tracking fails
            print(f"MLflow tracking error: {str(e)}")
            return None


mlflow_tracker = MLflowTracker()
