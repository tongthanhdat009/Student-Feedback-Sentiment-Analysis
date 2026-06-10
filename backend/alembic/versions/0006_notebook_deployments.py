"""notebook deployments
Revision ID: 0006_notebook_deployments
Revises: 0005_reconcile_kaggle_datasets
Create Date: 2026-06-02
"""
from alembic import op

revision='0006_notebook_deployments'
down_revision='0005_reconcile_kaggle_datasets'
branch_labels=None
depends_on=None


def upgrade():
    op.execute("""
        CREATE TABLE IF NOT EXISTS notebook_deployments (
            id UUID PRIMARY KEY,
            account_id UUID NOT NULL REFERENCES kaggle_accounts(id),
            notebook_id VARCHAR(255) NOT NULL,
            kaggle_ref VARCHAR(255) NOT NULL,
            remote_slug VARCHAR(255) NOT NULL,
            remote_title VARCHAR(255),
            source_path TEXT,
            is_active BOOLEAN NOT NULL DEFAULT true,
            last_synced_at TIMESTAMP WITH TIME ZONE,
            last_triggered_at TIMESTAMP WITH TIME ZONE,
            last_status VARCHAR(50),
            deployment_metadata JSON,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL
        )
    """)
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_notebook_deployments_account_notebook ON notebook_deployments (account_id, notebook_id)")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_notebook_deployments_account_ref ON notebook_deployments (account_id, kaggle_ref)")


def downgrade():
    op.execute("DROP TABLE IF EXISTS notebook_deployments")
