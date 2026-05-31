"""job runtime metadata
Revision ID: 0002_job_runtime_metadata
Revises: 0001_init
Create Date: 2026-05-31
"""
from alembic import op
import sqlalchemy as sa

revision='0002_job_runtime_metadata'
down_revision='0001_init'
branch_labels=None
depends_on=None


def upgrade():
    op.add_column('kaggle_jobs', sa.Column('kaggle_ref', sa.Text(), nullable=True))
    op.add_column('kaggle_jobs', sa.Column('staging_path', sa.Text(), nullable=True))
    op.add_column('kaggle_jobs', sa.Column('last_polled_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('kaggle_jobs', sa.Column('timeout_seconds', sa.Integer(), nullable=True))


def downgrade():
    op.drop_column('kaggle_jobs', 'timeout_seconds')
    op.drop_column('kaggle_jobs', 'last_polled_at')
    op.drop_column('kaggle_jobs', 'staging_path')
    op.drop_column('kaggle_jobs', 'kaggle_ref')
