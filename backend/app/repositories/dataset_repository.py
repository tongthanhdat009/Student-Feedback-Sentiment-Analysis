from uuid import UUID
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from ..models import KaggleDataset


class DatasetRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def ensure_schema(self):
        await self.session.execute(text("""
        CREATE TABLE IF NOT EXISTS kaggle_datasets (
            id UUID PRIMARY KEY,
            dataset_ref TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        )
        """))
        statements = [
            "ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS slug VARCHAR(100)",
            "ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS title VARCHAR(255)",
            "ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS description TEXT",
            "ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS local_path TEXT",
            "ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS status VARCHAR(50) NOT NULL DEFAULT 'active'",
            "ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS last_synced_at TIMESTAMP WITH TIME ZONE",
            "ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS last_validated_at TIMESTAMP WITH TIME ZONE",
            "ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS validation_result JSON",
            "ALTER TABLE kaggle_datasets ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT true",
            "UPDATE kaggle_datasets SET slug = dataset_ref WHERE slug IS NULL OR slug = ''",
            "ALTER TABLE kaggle_datasets ALTER COLUMN slug SET NOT NULL",
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_kaggle_datasets_slug ON kaggle_datasets (slug)",
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_kaggle_datasets_dataset_ref ON kaggle_datasets (dataset_ref)",
        ]
        for statement in statements:
            await self.session.execute(text(statement))
        await self.session.commit()

    async def list(self, limit: int | None = None, offset: int = 0):
        stmt = select(KaggleDataset).order_by(KaggleDataset.slug)
        if limit is not None:
            stmt = stmt.limit(limit).offset(offset)
        return (await self.session.execute(stmt)).scalars().all()
    async def list_active(self, limit: int | None = None, offset: int = 0):
        stmt = select(KaggleDataset).where(KaggleDataset.is_active == True).order_by(KaggleDataset.slug)
        if limit is not None:
            stmt = stmt.limit(limit).offset(offset)
        return (await self.session.execute(stmt)).scalars().all()
    async def count(self, active_only: bool = False):
        stmt = select(func.count()).select_from(KaggleDataset)
        if active_only:
            stmt = stmt.where(KaggleDataset.is_active == True)
        return (await self.session.execute(stmt)).scalar_one()
    async def get(self, dataset_id: UUID):
        return await self.session.get(KaggleDataset, dataset_id)
    async def get_by_id_text(self, dataset_id: str):
        return (await self.session.execute(select(KaggleDataset).where(KaggleDataset.id == dataset_id))).scalar_one_or_none()
    async def get_by_slug(self, slug: str):
        return (await self.session.execute(select(KaggleDataset).where(KaggleDataset.slug == slug))).scalar_one_or_none()
    async def get_by_ref(self, dataset_ref: str):
        return (await self.session.execute(select(KaggleDataset).where(KaggleDataset.dataset_ref == dataset_ref))).scalar_one_or_none()
    async def add(self, dataset: KaggleDataset):
        self.session.add(dataset); await self.session.commit(); await self.session.refresh(dataset); return dataset
    async def save(self, dataset: KaggleDataset):
        self.session.add(dataset); await self.session.commit(); await self.session.refresh(dataset); return dataset
    async def delete(self, dataset: KaggleDataset):
        await self.session.delete(dataset); await self.session.commit()
