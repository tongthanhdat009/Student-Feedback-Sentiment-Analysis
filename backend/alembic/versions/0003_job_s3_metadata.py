"""job s3 metadata
Revision ID: 0003_job_s3_metadata
Revises: 0002_job_runtime_metadata
Create Date: 2026-05-31
"""
from alembic import op
import sqlalchemy as sa

revision='0003_job_s3_metadata'
down_revision='0002_job_runtime_metadata'
branch_labels=None
depends_on=None


def upgrade():
    op.add_column('kaggle_jobs', sa.Column('staging_s3_prefix', sa.Text(), nullable=True))
    op.add_column('kaggle_jobs', sa.Column('output_s3_prefix', sa.Text(), nullable=True))
    op.add_column('kaggle_jobs', sa.Column('result_metadata', sa.JSON(), nullable=True))


def downgrade():
    op.drop_column('kaggle_jobs', 'result_metadata')
    op.drop_column('kaggle_jobs', 'output_s3_prefix')
    op.drop_column('kaggle_jobs', 'staging_s3_prefix')
