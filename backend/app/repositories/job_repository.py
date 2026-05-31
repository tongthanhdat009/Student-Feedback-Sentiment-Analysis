from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ..models import KaggleJob

class JobRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def list(self):
        return (await self.session.execute(select(KaggleJob).order_by(KaggleJob.created_at.desc()))).scalars().all()
    async def get(self, job_id: UUID):
        return await self.session.get(KaggleJob, job_id)
    async def add(self, job: KaggleJob):
        self.session.add(job); await self.session.commit(); await self.session.refresh(job); return job
    async def save(self, job: KaggleJob):
        self.session.add(job); await self.session.commit(); await self.session.refresh(job); return job
    async def stale_running(self):
        return (await self.session.execute(select(KaggleJob).where(KaggleJob.status.in_(['pending','staging','pushed','running'])))).scalars().all()
