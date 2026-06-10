from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from ..models import KaggleAccount

class AccountRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def list(self, limit: int | None = None, offset: int = 0):
        stmt = select(KaggleAccount).order_by(KaggleAccount.name)
        if limit is not None:
            stmt = stmt.limit(limit).offset(offset)
        return (await self.session.execute(stmt)).scalars().all()
    async def count(self):
        return (await self.session.execute(select(func.count()).select_from(KaggleAccount))).scalar_one()
    async def get_by_name(self, name: str):
        return (await self.session.execute(select(KaggleAccount).where(KaggleAccount.name == name))).scalar_one_or_none()
    async def get(self, account_id):
        return await self.session.get(KaggleAccount, account_id)
    async def add(self, account: KaggleAccount):
        self.session.add(account); await self.session.commit(); await self.session.refresh(account); return account
    async def save(self, account: KaggleAccount):
        self.session.add(account); await self.session.commit(); await self.session.refresh(account); return account
    async def delete(self, account: KaggleAccount):
        await self.session.delete(account); await self.session.commit()
