"""
Repository Pattern for Database Access
Provides clean abstraction layer over SQLAlchemy ORM
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from datetime import datetime, timedelta

from app.models.database import (
    Formula, FormulaExecution, ValidationResult, 
    LearningEvent, ContextPerformance, FormulaStatus
)


class FormulaRepository:
    """Repository for Formula CRUD operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, formula: Formula) -> Formula:
        """Create new formula"""
        self.db.add(formula)
        self.db.commit()
        self.db.refresh(formula)
        return formula
    
    def get_by_id(self, formula_id: str) -> Optional[Formula]:
        """Get formula by ID"""
        return self.db.query(Formula).filter(
            Formula.formula_id == formula_id
        ).first()
    
    def get_by_name(self, name: str) -> Optional[Formula]:
        """Get formula by name"""
        return self.db.query(Formula).filter(
            Formula.name == name
        ).first()
    
    def list_all(
        self,
        domain: Optional[str] = None,
        status: Optional[FormulaStatus] = None,
        min_confidence: float = 0.0,
        limit: int = 100,
        offset: int = 0
    ) -> List[Formula]:
        """List formulas with filters"""
        query = self.db.query(Formula)
        
        if domain:
            query = query.filter(Formula.domain == domain)
        if status:
            query = query.filter(Formula.status == status)
        if min_confidence > 0:
            query = query.filter(Formula.confidence_score >= min_confidence)
        
        return query.order_by(
            Formula.confidence_score.desc()
        ).limit(limit).offset(offset).all()
    
    def update(self, formula: Formula) -> Formula:
        """Update formula"""
        formula.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(formula)
        return formula
    
    def delete(self, formula_id: str) -> bool:
        """Delete formula"""
        formula = self.get_by_id(formula_id)
        if formula:
            self.db.delete(formula)
            self.db.commit()
            return True
        return False
    
    def get_by_domain(self, domain: str) -> List[Formula]:
        """Get all formulas for a domain"""
        return self.db.query(Formula).filter(
            Formula.domain == domain
        ).all()
    
    def search(self, query: str) -> List[Formula]:
        """Search formulas by name or description"""
        search = f"%{query}%"
        return self.db.query(Formula).filter(
            or_(
                Formula.name.ilike(search),
                Formula.description.ilike(search)
            )
        ).all()
    
    def get_top_performers(self, limit: int = 10) -> List[Formula]:
        """Get top performing formulas by confidence"""
        return self.db.query(Formula).filter(
            Formula.total_executions > 10
        ).order_by(
            Formula.confidence_score.desc()
        ).limit(limit).all()
    
    def increment_execution_count(self, formula_id: int, success: bool):
        """Increment execution counters"""
        formula = self.db.query(Formula).filter(
            Formula.id == formula_id
        ).first()
        
        if formula:
            formula.total_executions += 1
            if success:
                formula.successful_executions += 1
            else:
                formula.failed_executions += 1
            self.db.commit()


class ExecutionRepository:
    """Repository for Formula Execution records"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, execution: FormulaExecution) -> FormulaExecution:
        """Create execution record"""
        self.db.add(execution)
        self.db.commit()
        self.db.refresh(execution)
        return execution
    
    def get_by_id(self, execution_id: str) -> Optional[FormulaExecution]:
        """Get execution by ID"""
        return self.db.query(FormulaExecution).filter(
            FormulaExecution.execution_id == execution_id
        ).first()
    
    def get_recent_for_formula(
        self,
        formula_id: int,
        limit: int = 100
    ) -> List[FormulaExecution]:
        """Get recent executions for a formula"""
        return self.db.query(FormulaExecution).filter(
            FormulaExecution.formula_id == formula_id
        ).order_by(
            FormulaExecution.executed_at.desc()
        ).limit(limit).all()
    
    def get_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        formula_id: Optional[int] = None
    ) -> List[FormulaExecution]:
        """Get executions in date range"""
        query = self.db.query(FormulaExecution).filter(
            and_(
                FormulaExecution.executed_at >= start_date,
                FormulaExecution.executed_at <= end_date
            )
        )
        
        if formula_id:
            query = query.filter(FormulaExecution.formula_id == formula_id)
        
        return query.order_by(FormulaExecution.executed_at).all()
    
    def get_statistics(
        self,
        formula_id: int,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get execution statistics for a formula"""
        since = datetime.utcnow() - timedelta(days=days)
        
        executions = self.db.query(FormulaExecution).filter(
            and_(
                FormulaExecution.formula_id == formula_id,
                FormulaExecution.executed_at >= since
            )
        ).all()
        
        total = len(executions)
        successful = sum(1 for e in executions if e.status == "completed")
        failed = total - successful
        
        errors = [e.actual_vs_expected_error for e in executions 
                 if e.actual_vs_expected_error is not None]
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "average_error": sum(errors) / len(errors) if errors else None,
            "median_error": sorted(errors)[len(errors)//2] if errors else None
        }


class ValidationRepository:
    """Repository for Validation Results"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, validation: ValidationResult) -> ValidationResult:
        """Create validation record"""
        self.db.add(validation)
        self.db.commit()
        self.db.refresh(validation)
        return validation
    
    def get_latest_for_formula(self, formula_id: int) -> Optional[ValidationResult]:
        """Get latest validation for formula"""
        return self.db.query(ValidationResult).filter(
            ValidationResult.formula_id == formula_id
        ).order_by(
            ValidationResult.validated_at.desc()
        ).first()
    
    def get_all_for_formula(self, formula_id: int) -> List[ValidationResult]:
        """Get all validations for a formula"""
        return self.db.query(ValidationResult).filter(
            ValidationResult.formula_id == formula_id
        ).order_by(
            ValidationResult.validated_at.desc()
        ).all()


class LearningRepository:
    """Repository for Learning Events and Context Performance"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_learning_event(self, event: LearningEvent) -> LearningEvent:
        """Create learning event"""
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event
    
    def get_learning_history(
        self,
        formula_id: int,
        limit: int = 100
    ) -> List[LearningEvent]:
        """Get learning history for formula"""
        return self.db.query(LearningEvent).filter(
            LearningEvent.formula_id == formula_id
        ).order_by(
            LearningEvent.timestamp.desc()
        ).limit(limit).all()
    
    def get_context_performance(
        self,
        formula_id: int
    ) -> List[ContextPerformance]:
        """Get context performance records"""
        return self.db.query(ContextPerformance).filter(
            ContextPerformance.formula_id == formula_id
        ).order_by(
            ContextPerformance.confidence_in_context.desc()
        ).all()
    
    def update_context_performance(
        self,
        formula_id: int,
        context_hash: str,
        context_data: Dict[str, Any],
        success: bool
    ):
        """Update or create context performance record"""
        perf = self.db.query(ContextPerformance).filter(
            and_(
                ContextPerformance.formula_id == formula_id,
                ContextPerformance.context_hash == context_hash
            )
        ).first()
        
        if perf:
            perf.total_executions += 1
            if success:
                perf.successful_executions += 1
            perf.last_execution_at = datetime.utcnow()
        else:
            perf = ContextPerformance(
                formula_id=formula_id,
                context_hash=context_hash,
                context_data=context_data,
                total_executions=1,
                successful_executions=1 if success else 0,
                confidence_in_context=0.5,
                last_execution_at=datetime.utcnow()
            )
            self.db.add(perf)
        
        self.db.commit()


# Factory functions for easy instantiation
def get_formula_repository(db: Session) -> FormulaRepository:
    """Get formula repository instance"""
    return FormulaRepository(db)

def get_execution_repository(db: Session) -> ExecutionRepository:
    """Get execution repository instance"""
    return ExecutionRepository(db)

def get_validation_repository(db: Session) -> ValidationRepository:
    """Get validation repository instance"""
    return ValidationRepository(db)

def get_learning_repository(db: Session) -> LearningRepository:
    """Get learning repository instance"""
    return LearningRepository(db)
