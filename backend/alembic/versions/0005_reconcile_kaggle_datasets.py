"""reconcile legacy kaggle datasets table
Revision ID: 0005_reconcile_kaggle_datasets
Revises: 0004_datasets
Create Date: 2026-06-01
"""
from alembic import op

revision='0005_reconcile_kaggle_datasets'
down_revision='0004_datasets'
branch_labels=None
depends_on=None


def upgrade():
    # Safe no-op if 0004 already created the current schema; fixes DBs that had
    # the legacy table but then stamped past 0004.
    op.execute("ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS slug VARCHAR(100)")
    op.execute("ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS title VARCHAR(255)")
    op.execute("ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS description TEXT")
    op.execute("ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS local_path TEXT")
    op.execute("ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS status VARCHAR(50) NOT NULL DEFAULT 'active'")
    op.execute("ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS last_synced_at TIMESTAMP WITH TIME ZONE")
    op.execute("ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS last_validated_at TIMESTAMP WITH TIME ZONE")
    op.execute("ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS validation_result JSON")
    op.execute("ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT true")
    op.execute("UPDATE kaggle_datasets SET slug = dataset_ref WHERE slug IS NULL OR slug = ''")
    op.execute("ALTER TABLE kaggle_datasets ALTER COLUMN slug SET NOT NULL")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_kaggle_datasets_slug ON kaggle_datasets (slug)")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_kaggle_datasets_dataset_ref ON kaggle_datasets (dataset_ref)")


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_kaggle_datasets_dataset_ref")
    op.execute("DROP INDEX IF EXISTS ix_kaggle_datasets_slug")
