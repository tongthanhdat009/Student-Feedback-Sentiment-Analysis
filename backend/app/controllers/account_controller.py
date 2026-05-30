from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from ..database import get_session
from ..schemas.account import AccountCreate, AccountRead
from ..services.account_service import AccountService
from ..utils.auth import require_api_key
from ..services.kaggle_client_factory import KaggleClientFactory
router=APIRouter(prefix='/api/kaggle/accounts', tags=['accounts'], dependencies=[Depends(require_api_key)])
@router.get('', response_model=list[AccountRead])
async def list_accounts(session: AsyncSession=Depends(get_session)): return await AccountService(session).list_accounts()
@router.post('', response_model=AccountRead)
async def create_account(data: AccountCreate, session: AsyncSession=Depends(get_session)): return await AccountService(session).create_account(data)
@router.delete('/{name}')
async def delete_account(name: str, session: AsyncSession=Depends(get_session)): await AccountService(session).delete_account(name); return {'ok': True}
@router.post('/{name}/test')
async def test_account(name: str, session: AsyncSession=Depends(get_session)):
    acc,key=await AccountService(session).get_credentials(name)
    KaggleClientFactory().create(acc.kaggle_username, key)
    return {'ok': True}
