from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ..config import get_settings
from ..models import KaggleAccount
from ..repositories.account_repository import AccountRepository
from ..schemas.account import AccountCreate, AccountUpdate
from ..utils.encryption import EncryptionService

class AccountService:
    def __init__(self, session: AsyncSession):
        self.repo = AccountRepository(session)
        self.crypto = EncryptionService(get_settings().fernet_key)
    async def list_accounts(self): return await self.repo.list()
    async def create_account(self, data: AccountCreate):
        if await self.repo.get_by_name(data.name): raise HTTPException(409, 'Account exists')
        return await self.repo.add(KaggleAccount(name=data.name, kaggle_username=data.kaggle_username, kaggle_key_encrypted=self.crypto.encrypt(data.kaggle_key)))
    async def update_account(self, name: str, data: AccountUpdate):
        account = await self.repo.get_by_name(name)
        if not account: raise HTTPException(404, 'Account not found')
        if data.name is not None: account.name = data.name
        if data.kaggle_username is not None: account.kaggle_username = data.kaggle_username
        if data.kaggle_key: account.kaggle_key_encrypted = self.crypto.encrypt(data.kaggle_key)
        if data.is_active is not None: account.is_active = data.is_active
        return await self.repo.save(account)

    async def delete_account(self, name: str):
        account = await self.repo.get_by_name(name)
        if not account: raise HTTPException(404, 'Account not found')
        await self.repo.delete(account)
    async def get_credentials(self, name: str):
        account = await self.repo.get_by_name(name)
        if not account: raise HTTPException(404, 'Account not found')
        return account, self.crypto.decrypt(account.kaggle_key_encrypted)
