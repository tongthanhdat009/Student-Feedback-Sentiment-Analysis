"""init
Revision ID: 0001_init
Revises:
Create Date: 2026-05-30
"""
from alembic import op
import sqlalchemy as sa
revision='0001_init'
down_revision=None
branch_labels=None
depends_on=None

def upgrade():
    op.create_table('kaggle_accounts',
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('kaggle_username', sa.String(length=255), nullable=False),
        sa.Column('kaggle_key_encrypted', sa.Text(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True))
    op.create_index('ix_kaggle_accounts_name', 'kaggle_accounts', ['name'], unique=True)
    op.create_table('kaggle_jobs',
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('account_id', sa.Uuid(), sa.ForeignKey('kaggle_accounts.id'), nullable=True),
        sa.Column('job_type', sa.String(length=50), nullable=False),
        sa.Column('target_ref', sa.String(length=255), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('output_path', sa.Text(), nullable=True),
        sa.Column('s3_object_key', sa.Text(), nullable=True),
        sa.Column('s3_presigned_url', sa.Text(), nullable=True),
        sa.Column('s3_presigned_url_expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True))

def downgrade():
    op.drop_table('kaggle_jobs'); op.drop_index('ix_kaggle_accounts_name', table_name='kaggle_accounts'); op.drop_table('kaggle_accounts')
