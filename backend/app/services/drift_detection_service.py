"""
Drift Detection Service using ADWIN (Adaptive Windowing) algorithm.
Monitors correction rates and detects concept drift in deployed models.
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..models.edge_devices import DriftMetric, RetrainJob
from ..models.corrections import Correction, CorrectionStatus
from ..models.database import Formula, FormulaExecution
import math


class ADWIN:
    """
    ADWIN (ADaptive WINdowing) algorithm for drift detection.
    Maintains a window of recent data and detects when distribution changes.
    """

    def __init__(self, delta: float = 0.002):
        """
        Initialize ADWIN.

        Args:
            delta: Confidence level (smaller = more sensitive, default 0.002)
        """
        self.delta = delta
        self.window = []
        self.total = 0.0
        self.variance = 0.0
        self.width = 0

    def add_element(self, value: float) -> bool:
        """
        Add a new element and check for drift.

        Args:
            value: New value to add (e.g., 1.0 for correction, 0.0 for no correction)

        Returns:
            True if drift detected, False otherwise
        """
        self.window.append(value)
        self.width += 1
        self.total += value

        drift_detected = False

        # Check for drift by splitting window
        if self.width >= 2:
            drift_detected = self._check_for_change()

        return drift_detected

    def _check_for_change(self) -> bool:
        """
        Check if there's a significant change in the window.
        Uses the ADWIN algorithm to detect distribution changes.
        """
        n = len(self.window)
        if n < 5:  # Need minimum window size
            return False

        # Try different split points
        for i in range(1, n):
            # Split window into two parts
            w0 = self.window[:i]
            w1 = self.window[i:]

            n0 = len(w0)
            n1 = len(w1)

            # Calculate means
            mean0 = sum(w0) / n0 if n0 > 0 else 0
            mean1 = sum(w1) / n1 if n1 > 0 else 0

            # Calculate difference
            diff = abs(mean0 - mean1)

            # Calculate threshold using Hoeffding bound
            m = 1 / ((1 / n0) + (1 / n1))
            threshold = math.sqrt((2 / m) * math.log(2 / self.delta))

            # If difference exceeds threshold, drift detected
            if diff > threshold:
                # Remove old data
                self.window = w1
                self.width = n1
                self.total = sum(w1)
                return True

        return False

    def reset(self):
        """Reset the window."""
        self.window = []
        self.total = 0.0
        self.variance = 0.0
        self.width = 0


class DriftDetectionService:
    """
    Service for detecting concept drift in deployed models.
    Monitors correction rates and triggers retraining when drift is detected.
    """

    @staticmethod
    def calculate_drift_metrics(
        db: Session,
        formula_id: int,
        time_window_hours: int = 24,
        device_id: Optional[int] = None
    ) -> Optional[DriftMetric]:
        """
        Calculate drift metrics for a formula over a time window.

        Args:
            db: Database session
            formula_id: Formula to analyze
            time_window_hours: Time window in hours (default 24)
            device_id: Optional specific device to analyze

        Returns:
            DriftMetric object if drift analysis completed, None otherwise
        """
        window_start = datetime.utcnow() - timedelta(hours=time_window_hours)
        window_end = datetime.utcnow()

        # Get executions in time window
        query = db.query(FormulaExecution).filter(
            FormulaExecution.formula_id == formula_id,
            FormulaExecution.execution_timestamp >= window_start,
            FormulaExecution.execution_timestamp <= window_end
        )

        if device_id:
            query = query.filter(FormulaExecution.edge_device_id == device_id)

        executions = query.all()

        if not executions:
            return None

        executions_count = len(executions)

        # Get corrections for these executions
        execution_ids = [e.id for e in executions]
        corrections = db.query(Correction).filter(
            Correction.execution_id.in_(execution_ids)
        ).all()

        corrections_count = len(corrections)
        correction_rate = (corrections_count / executions_count) * 100 if executions_count > 0 else 0

        # Get baseline correction rate (last 7 days before this window)
        baseline_start = window_start - timedelta(days=7)
        baseline_executions = db.query(FormulaExecution).filter(
            FormulaExecution.formula_id == formula_id,
            FormulaExecution.execution_timestamp >= baseline_start,
            FormulaExecution.execution_timestamp < window_start
        )

        if device_id:
            baseline_executions = baseline_executions.filter(
                FormulaExecution.edge_device_id == device_id
            )

        baseline_executions_list = baseline_executions.all()
        baseline_count = len(baseline_executions_list)

        baseline_correction_rate = 0.0
        if baseline_count > 0:
            baseline_execution_ids = [e.id for e in baseline_executions_list]
            baseline_corrections_count = db.query(Correction).filter(
                Correction.execution_id.in_(baseline_execution_ids)
            ).count()
            baseline_correction_rate = (baseline_corrections_count / baseline_count) * 100

        # Calculate change from baseline
        correction_rate_change = correction_rate - baseline_correction_rate

        # Run ADWIN drift detection
        adwin = ADWIN(delta=0.002)
        drift_detected = False
        drift_score = 0.0

        # Feed each execution to ADWIN (1 if corrected, 0 if not)
        correction_map = {c.execution_id: True for c in corrections}
        for execution in executions:
            is_corrected = 1.0 if execution.id in correction_map else 0.0
            if adwin.add_element(is_corrected):
                drift_detected = True

        # Calculate drift score (normalized change)
        if baseline_correction_rate > 0:
            drift_score = abs(correction_rate_change) / baseline_correction_rate
        else:
            drift_score = correction_rate / 100.0 if correction_rate > 0 else 0.0

        # Create drift metric record
        drift_metric = DriftMetric(
            formula_id=formula_id,
            device_id=device_id,
            window_start=window_start,
            window_end=window_end,
            correction_rate=correction_rate,
            executions_count=executions_count,
            corrections_count=corrections_count,
            drift_detected=drift_detected,
            drift_score=drift_score,
            drift_threshold=0.05,
            baseline_correction_rate=baseline_correction_rate,
            correction_rate_change=correction_rate_change,
            retrain_triggered=False
        )

        db.add(drift_metric)
        db.commit()
        db.refresh(drift_metric)

        return drift_metric

    @staticmethod
    def check_all_formulas_for_drift(
        db: Session,
        time_window_hours: int = 24,
        auto_trigger_retrain: bool = True
    ) -> List[DriftMetric]:
        """
        Check all deployed formulas for drift.

        Args:
            db: Database session
            time_window_hours: Time window in hours
            auto_trigger_retrain: Whether to auto-trigger retraining on drift

        Returns:
            List of drift metrics for formulas with drift detected
        """
        # Get all formulas that have been executed recently
        recent_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        formula_ids = db.query(FormulaExecution.formula_id).filter(
            FormulaExecution.execution_timestamp >= recent_time
        ).distinct().all()

        drift_metrics = []

        for (formula_id,) in formula_ids:
            metric = DriftDetectionService.calculate_drift_metrics(
                db=db,
                formula_id=formula_id,
                time_window_hours=time_window_hours
            )

            if metric and metric.drift_detected:
                drift_metrics.append(metric)

                # Auto-trigger retrain if enabled
                if auto_trigger_retrain and not metric.retrain_triggered:
                    from .retrain_service import RetrainService
                    job = RetrainService.trigger_retrain(
                        db=db,
                        formula_id=formula_id,
                        trigger_type="drift_detected",
                        drift_metric_id=metric.id
                    )

                    if job:
                        metric.retrain_triggered = True
                        metric.retrain_job_id = job.job_id
                        db.commit()

        return drift_metrics

    @staticmethod
    def get_drift_history(
        db: Session,
        formula_id: int,
        days: int = 30
    ) -> List[DriftMetric]:
        """
        Get drift detection history for a formula.

        Args:
            db: Database session
            formula_id: Formula to query
            days: Number of days of history

        Returns:
            List of drift metrics ordered by time
        """
        start_time = datetime.utcnow() - timedelta(days=days)

        metrics = db.query(DriftMetric).filter(
            DriftMetric.formula_id == formula_id,
            DriftMetric.created_at >= start_time
        ).order_by(DriftMetric.window_start.asc()).all()

        return metrics

    @staticmethod
    def get_drift_summary(db: Session, formula_id: int) -> Dict[str, Any]:
        """
        Get a summary of drift metrics for a formula.

        Args:
            db: Database session
            formula_id: Formula to summarize

        Returns:
            Summary dictionary
        """
        # Last 30 days
        metrics = DriftDetectionService.get_drift_history(db, formula_id, days=30)

        if not metrics:
            return {
                "total_checks": 0,
                "drift_detected_count": 0,
                "latest_correction_rate": 0.0,
                "avg_correction_rate": 0.0,
                "drift_percentage": 0.0,
                "retrains_triggered": 0
            }

        total_checks = len(metrics)
        drift_detected_count = sum(1 for m in metrics if m.drift_detected)
        retrains_triggered = sum(1 for m in metrics if m.retrain_triggered)

        latest_correction_rate = metrics[-1].correction_rate if metrics else 0.0
        avg_correction_rate = sum(m.correction_rate for m in metrics) / total_checks

        return {
            "total_checks": total_checks,
            "drift_detected_count": drift_detected_count,
            "latest_correction_rate": latest_correction_rate,
            "avg_correction_rate": avg_correction_rate,
            "drift_percentage": (drift_detected_count / total_checks * 100) if total_checks > 0 else 0.0,
            "retrains_triggered": retrains_triggered,
            "latest_drift_score": metrics[-1].drift_score if metrics else 0.0
        }
