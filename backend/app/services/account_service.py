from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ..config import get_settings
from ..models import KaggleAccount
from ..repositories.account_repository import AccountRepository
from ..schemas.account import AccountCreate
from ..utils.encryption import EncryptionService

class AccountService:
    def __init__(self, session: AsyncSession):
        self.repo = AccountRepository(session)
        self.crypto = EncryptionService(get_settings().fernet_key)
    async def list_accounts(self): return await self.repo.list()
    async def create_account(self, data: AccountCreate):
        if await self.repo.get_by_name(data.name): raise HTTPException(409, 'Account exists')
        return await self.repo.add(KaggleAccount(name=data.name, kaggle_username=data.kaggle_username, kaggle_key_encrypted=self.crypto.encrypt(data.kaggle_key)))
    async def delete_account(self, name: str):
        account = await self.repo.get_by_name(name)
        if not account: raise HTTPException(404, 'Account not found')
        await self.repo.delete(account)
    async def get_credentials(self, name: str):
        account = await self.repo.get_by_name(name)
        if not account: raise HTTPException(404, 'Account not found')
        return account, self.crypto.decrypt(account.kaggle_key_encrypted)
