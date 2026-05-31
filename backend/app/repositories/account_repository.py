from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ..models import KaggleAccount

class AccountRepository:
    def __init__(self, session: AsyncSession): self.session = session
    async def list(self):
        return (await self.session.execute(select(KaggleAccount).order_by(KaggleAccount.name))).scalars().all()
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
