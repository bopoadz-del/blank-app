"""
MLflow Integration for The Reasoner AI Platform.

Tracks:
- Formula executions as experiments
- Confidence scores over time
- Context performance
- Model metrics
"""
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

from app.core.config import settings


class MLflowTracker:
    """
    MLflow integration for tracking formula performance and experiments.
    """
    
    def __init__(self):
        self.tracking_uri = settings.MLFLOW_TRACKING_URI
        self.experiment_name = settings.MLFLOW_EXPERIMENT_NAME
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            self.experiment_id = experiment.experiment_id if experiment else None
        
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        logger.info(f"MLflow tracker initialized: {self.tracking_uri}")
    
    def track_formula_execution(
        self,
        formula_id: str,
        formula_name: str,
        input_values: Dict[str, Any],
        output_values: Any,
        context: Dict[str, Any],
        execution_time: float,
        success: bool,
        confidence_score: float,
        validation_passed: Optional[bool] = None,
        error: Optional[str] = None
    ) -> str:
        """
        Track formula execution in MLflow.
        
        Returns:
            MLflow run_id
        """
        try:
            with mlflow.start_run(experiment_id=self.experiment_id) as run:
                # Log parameters
                mlflow.log_param("formula_id", formula_id)
                mlflow.log_param("formula_name", formula_name)
                
                # Log inputs
                for key, value in input_values.items():
                    mlflow.log_param(f"input_{key}", value)
                
                # Log context
                for key, value in context.items():
                    mlflow.log_param(f"context_{key}", value)
                
                # Log metrics
                mlflow.log_metric("execution_time", execution_time)
                mlflow.log_metric("confidence_score", confidence_score)
                mlflow.log_metric("success", 1.0 if success else 0.0)
                
                if validation_passed is not None:
                    mlflow.log_metric("validation_passed", 1.0 if validation_passed else 0.0)
                
                # Log output
                mlflow.log_param("output", str(output_values))
                
                # Log error if any
                if error:
                    mlflow.log_param("error", error)
                
                # Tags
                mlflow.set_tag("formula_id", formula_id)
                mlflow.set_tag("timestamp", datetime.utcnow().isoformat())
                mlflow.set_tag("success", str(success))
                
                logger.debug(f"Logged execution to MLflow: {run.info.run_id}")
                return run.info.run_id
                
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")
            return None
    
    def track_confidence_update(
        self,
        formula_id: str,
        old_confidence: float,
        new_confidence: float,
        reason: str,
        total_executions: int,
        success_rate: float
    ):
        """Track confidence score updates."""
        try:
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"confidence_update_{formula_id}") as run:
                mlflow.log_param("formula_id", formula_id)
                mlflow.log_param("reason", reason)
                
                mlflow.log_metric("old_confidence", old_confidence)
                mlflow.log_metric("new_confidence", new_confidence)
                mlflow.log_metric("confidence_delta", new_confidence - old_confidence)
                mlflow.log_metric("total_executions", total_executions)
                mlflow.log_metric("success_rate", success_rate)
                
                mlflow.set_tag("event_type", "confidence_update")
                mlflow.set_tag("timestamp", datetime.utcnow().isoformat())
                
        except Exception as e:
            logger.error(f"Failed to track confidence update: {e}")
    
    def track_validation(
        self,
        formula_id: str,
        validation_stages: Dict[str, Any],
        overall_passed: bool
    ):
        """Track validation results."""
        try:
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"validation_{formula_id}") as run:
                mlflow.log_param("formula_id", formula_id)
                mlflow.log_metric("overall_passed", 1.0 if overall_passed else 0.0)
                
                # Log each stage
                for stage in validation_stages:
                    stage_name = stage['stage']
                    passed = stage['passed']
                    confidence = stage.get('confidence', 0.0)
                    
                    mlflow.log_metric(f"stage_{stage_name}_passed", 1.0 if passed else 0.0)
                    mlflow.log_metric(f"stage_{stage_name}_confidence", confidence)
                
                mlflow.set_tag("event_type", "validation")
                mlflow.set_tag("timestamp", datetime.utcnow().isoformat())
                
        except Exception as e:
            logger.error(f"Failed to track validation: {e}")
    
    def track_tier_change(
        self,
        formula_id: str,
        old_tier: str,
        new_tier: str,
        reason: str,
        auto_deploy: bool
    ):
        """Track credibility tier changes."""
        try:
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"tier_change_{formula_id}") as run:
                mlflow.log_param("formula_id", formula_id)
                mlflow.log_param("old_tier", old_tier)
                mlflow.log_param("new_tier", new_tier)
                mlflow.log_param("reason", reason)
                mlflow.log_metric("auto_deploy", 1.0 if auto_deploy else 0.0)
                
                mlflow.set_tag("event_type", "tier_change")
                mlflow.set_tag("timestamp", datetime.utcnow().isoformat())
                
        except Exception as e:
            logger.error(f"Failed to track tier change: {e}")
    
    def get_formula_metrics(self, formula_id: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get historical metrics for a formula from MLflow.
        
        Returns:
            Dictionary with confidence trends, success rates, etc.
        """
        try:
            # Search runs for this formula
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"params.formula_id = '{formula_id}'",
                max_results=limit,
                order_by=["start_time DESC"]
            )
            
            confidence_history = []
            execution_times = []
            success_count = 0
            total_count = 0
            
            for run in runs:
                metrics = run.data.metrics
                params = run.data.params
                
                if 'confidence_score' in metrics:
                    confidence_history.append({
                        'timestamp': run.info.start_time,
                        'confidence': metrics['confidence_score']
                    })
                
                if 'execution_time' in metrics:
                    execution_times.append(metrics['execution_time'])
                
                if 'success' in metrics:
                    total_count += 1
                    if metrics['success'] == 1.0:
                        success_count += 1
            
            return {
                "formula_id": formula_id,
                "total_runs": len(runs),
                "confidence_history": confidence_history,
                "average_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
                "success_rate": success_count / total_count if total_count > 0 else 0,
                "last_run_time": runs[0].info.start_time if runs else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get formula metrics: {e}")
            return {}
    
    def check_connection(self) -> bool:
        """Check if MLflow server is reachable."""
        try:
            self.client.list_experiments()
            return True
        except:
            return False


# Global instance
mlflow_tracker = MLflowTracker()
