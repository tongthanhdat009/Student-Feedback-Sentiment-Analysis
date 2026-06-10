from uuid import UUID
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from ..models import NotebookDeployment


class NotebookDeploymentRepository:
    def __init__(self, session: AsyncSession): self.session = session

    async def ensure_schema(self):
        await self.session.execute(text("""
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
        """))
        await self.session.execute(text("ALTER TABLE notebook_deployments ADD COLUMN IF NOT EXISTS remote_title VARCHAR(255)"))
        await self.session.execute(text("ALTER TABLE notebook_deployments ADD COLUMN IF NOT EXISTS source_path TEXT"))
        await self.session.execute(text("ALTER TABLE notebook_deployments ADD COLUMN IF NOT EXISTS deployment_metadata JSON"))
        await self.session.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_notebook_deployments_account_notebook ON notebook_deployments (account_id, notebook_id)"))
        await self.session.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_notebook_deployments_account_ref ON notebook_deployments (account_id, kaggle_ref)"))
        await self.session.commit()

    async def list(self, account_id: UUID | None = None):
        stmt = select(NotebookDeployment).order_by(NotebookDeployment.updated_at.desc())
        if account_id:
            stmt = stmt.where(NotebookDeployment.account_id == account_id)
        return (await self.session.execute(stmt)).scalars().all()

    async def get(self, deployment_id: UUID):
        return await self.session.get(NotebookDeployment, deployment_id)

    async def get_for(self, account_id: UUID, notebook_id: str):
        return (await self.session.execute(select(NotebookDeployment).where(
            NotebookDeployment.account_id == account_id,
            NotebookDeployment.notebook_id == notebook_id,
            NotebookDeployment.is_active == True,
        ))).scalar_one_or_none()

    async def add(self, deployment: NotebookDeployment):
        self.session.add(deployment); await self.session.commit(); await self.session.refresh(deployment); return deployment

    async def save(self, deployment: NotebookDeployment):
        self.session.add(deployment); await self.session.commit(); await self.session.refresh(deployment); return deployment
