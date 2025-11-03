"""Formula execution database model"""

from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, DateTime
from sqlalchemy.sql import func
from app.database.session import Base


class FormulaExecution(Base):
    """Model for storing formula execution history"""

    __tablename__ = "formula_executions"

    id = Column(Integer, primary_key=True, index=True)
    formula_id = Column(String, index=True, nullable=False)
    input_values = Column(JSON, nullable=False)
    result = Column(Float, nullable=True)
    unit = Column(String, nullable=True)
    success = Column(Boolean, default=True, nullable=False)
    error = Column(String, nullable=True)
    execution_time_ms = Column(Float, nullable=False)
    api_key_hash = Column(String, nullable=False)  # Hashed for privacy
    mlflow_run_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    def __repr__(self):
        return f"<FormulaExecution(id={self.id}, formula_id='{self.formula_id}', success={self.success})>"
