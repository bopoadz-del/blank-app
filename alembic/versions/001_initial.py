"""Initial schema

Revision ID: 001_initial
Revises: 
Create Date: 2025-11-03 06:00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create formulas table
    op.create_table(
        'formulas',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('formula_id', sa.String(100), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('domain', sa.String(100), nullable=False),
        sa.Column('formula_expression', sa.Text(), nullable=False),
        sa.Column('input_parameters', postgresql.JSONB(), nullable=False),
        sa.Column('output_parameters', postgresql.JSONB(), nullable=False),
        sa.Column('source', sa.String(50), nullable=False),
        sa.Column('source_reference', sa.String(500), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='active'),
        sa.Column('confidence_score', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('credibility_tier', sa.String(20), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('formula_id')
    )
    op.create_index('ix_formulas_domain', 'formulas', ['domain'])
    op.create_index('ix_formulas_status', 'formulas', ['status'])
    op.create_index('ix_formulas_confidence', 'formulas', ['confidence_score'])

    # Create formula_executions table
    op.create_table(
        'formula_executions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('execution_id', sa.String(100), nullable=False),
        sa.Column('formula_id', sa.Integer(), nullable=False),
        sa.Column('input_values', postgresql.JSONB(), nullable=False),
        sa.Column('output_values', postgresql.JSONB(), nullable=True),
        sa.Column('context_data', postgresql.JSONB(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('execution_time', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('expected_output', postgresql.JSONB(), nullable=True),
        sa.Column('actual_vs_expected_error', sa.Float(), nullable=True),
        sa.Column('validation_passed', sa.Boolean(), nullable=True),
        sa.Column('edge_node_id', sa.String(100), nullable=True),
        sa.Column('user_id', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['formula_id'], ['formulas.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('execution_id')
    )
    op.create_index('ix_executions_formula', 'formula_executions', ['formula_id'])
    op.create_index('ix_executions_status', 'formula_executions', ['status'])
    op.create_index('ix_executions_created', 'formula_executions', ['created_at'])

    # Create validation_results table
    op.create_table(
        'validation_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('execution_id', sa.Integer(), nullable=False),
        sa.Column('validation_type', sa.String(50), nullable=False),
        sa.Column('passed', sa.Boolean(), nullable=False),
        sa.Column('details', postgresql.JSONB(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['execution_id'], ['formula_executions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_validation_execution', 'validation_results', ['execution_id'])

    # Create confidence_updates table
    op.create_table(
        'confidence_updates',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('formula_id', sa.Integer(), nullable=False),
        sa.Column('previous_score', sa.Float(), nullable=False),
        sa.Column('new_score', sa.Float(), nullable=False),
        sa.Column('reason', sa.String(200), nullable=True),
        sa.Column('context_factors', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['formula_id'], ['formulas.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_confidence_formula', 'confidence_updates', ['formula_id'])
    op.create_index('ix_confidence_created', 'confidence_updates', ['created_at'])


def downgrade() -> None:
    op.drop_table('confidence_updates')
    op.drop_table('validation_results')
    op.drop_table('formula_executions')
    op.drop_table('formulas')
