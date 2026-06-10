from uuid import UUID
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from ..models import KaggleJob

class JobRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def list(self, limit: int | None = None, offset: int = 0):
        stmt = select(KaggleJob).order_by(KaggleJob.created_at.desc())
        if limit is not None:
            stmt = stmt.limit(limit).offset(offset)
        return (await self.session.execute(stmt)).scalars().all()
    async def count(self):
        return (await self.session.execute(select(func.count()).select_from(KaggleJob))).scalar_one()
    async def get(self, job_id: UUID):
        return await self.session.get(KaggleJob, job_id)
    async def add(self, job: KaggleJob):
        self.session.add(job); await self.session.commit(); await self.session.refresh(job); return job
    async def save(self, job: KaggleJob):
        self.session.add(job); await self.session.commit(); await self.session.refresh(job); return job
    async def stale_running(self):
        return (await self.session.execute(select(KaggleJob).where(KaggleJob.status.in_(['pending','staging','pushed','running'])))).scalars().all()
