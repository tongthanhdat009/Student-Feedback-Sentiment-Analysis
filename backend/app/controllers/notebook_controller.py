from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ..database import get_session
from ..schemas.job import NotebookTriggerRequest, JobRead
from ..services.notebook_service import NotebookService
from ..utils.auth import require_api_key
from ..services.account_service import AccountService
from ..services.kaggle_client_factory import KaggleClientFactory
router=APIRouter(prefix='/api/kaggle/notebooks', tags=['notebooks'], dependencies=[Depends(require_api_key)])
@router.get('/inventory')
async def inventory(session: AsyncSession=Depends(get_session)): return NotebookService(session).inventory()
@router.get('')
async def list_remote(account: str, session: AsyncSession=Depends(get_session)):
    acc,key=await AccountService(session).get_credentials(account)
    api=KaggleClientFactory().create(acc.kaggle_username, key)
    return api.kernels_list(user=acc.kaggle_username)
@router.post('/trigger', response_model=JobRead)
async def trigger(req: NotebookTriggerRequest, session: AsyncSession=Depends(get_session)): return await NotebookService(session).trigger(req.account, req.notebook_id)
@router.get('/status/{owner}/{slug}')
async def status(owner: str, slug: str, account: str, session: AsyncSession=Depends(get_session)):
    acc,key=await AccountService(session).get_credentials(account)
    api=KaggleClientFactory().create(acc.kaggle_username, key)
    return api.kernel_status(f'{owner}/{slug}')
