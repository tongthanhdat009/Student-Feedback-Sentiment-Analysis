from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ..database import get_session
from ..schemas.account import AccountCreate, AccountRead, AccountUpdate
from ..services.account_service import AccountService
from ..utils.auth import require_api_key
from ..services.kaggle_client_factory import KaggleClientFactory
from ..services.kaggle_quota_service import fetch_kaggle_quota
router=APIRouter(prefix='/api/kaggle/accounts', tags=['accounts'], dependencies=[Depends(require_api_key)])
@router.get('')
async def list_accounts(page: int = 1, page_size: int = 20, session: AsyncSession=Depends(get_session)):
    items = await AccountService(session).list_accounts()
    total = len(items); start = (page-1)*page_size; end = start+page_size
    return {'items': items[start:end], 'total': total, 'page': page, 'page_size': page_size}
@router.post('', response_model=AccountRead)
async def create_account(data: AccountCreate, session: AsyncSession=Depends(get_session)): return await AccountService(session).create_account(data)
@router.patch('/{name}', response_model=AccountRead)
async def update_account(name: str, data: AccountUpdate, session: AsyncSession=Depends(get_session)): return await AccountService(session).update_account(name, data)
@router.delete('/{name}')
async def delete_account(name: str, session: AsyncSession=Depends(get_session)): await AccountService(session).delete_account(name); return {'ok': True}
@router.get('/{name}/quota')
async def account_quota(name: str, session: AsyncSession=Depends(get_session)):
    acc,key=await AccountService(session).get_credentials(name)
    return fetch_kaggle_quota(acc.kaggle_username, key)

@router.post('/{name}/test')
async def test_account(name: str, session: AsyncSession=Depends(get_session)):
    acc,key=await AccountService(session).get_credentials(name)
    try:
        api=KaggleClientFactory().create(acc.kaggle_username, key)
        api.kernels_list(mine=True, page_size=1)
    except Exception as exc:
        message = str(exc)
        if '401' in message:
            raise HTTPException(401, 'Kaggle credentials unauthorized. Regenerate API token for this exact Kaggle user, then update the account.')
        raise HTTPException(502, f'Kaggle credential test failed: {message}')
    return {'ok': True}
