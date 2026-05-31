from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ..database import get_session
from ..schemas.job import NotebookTriggerRequest, JobRead
from ..services.notebook_service import NotebookService
from ..utils.auth import require_api_key
from ..services.account_service import AccountService
from ..services.kaggle_client_factory import KaggleClientFactory
from ..services.job_worker import get_kernel_status
router=APIRouter(prefix='/api/kaggle/notebooks', tags=['notebooks'], dependencies=[Depends(require_api_key)])
@router.get('/inventory')
async def inventory(page: int = 1, page_size: int = 20, session: AsyncSession=Depends(get_session)):
    items = NotebookService(session).inventory()
    total = len(items); start = (page-1)*page_size; end = start+page_size
    return {'items': items[start:end], 'total': total, 'page': page, 'page_size': page_size}
@router.post('/{slug}/validate')
async def validate_notebook(slug: str, session: AsyncSession=Depends(get_session)): return NotebookService(session).validate(slug)
@router.get('')
async def list_remote(account: str, session: AsyncSession=Depends(get_session)):
    acc,key=await AccountService(session).get_credentials(account)
    api=KaggleClientFactory().create(acc.kaggle_username, key)
    return api.kernels_list(user=acc.kaggle_username)
@router.post('/trigger')
async def trigger(req: NotebookTriggerRequest, session: AsyncSession=Depends(get_session)):
    accounts = req.accounts or ([req.account] if req.account else [])
    if not accounts: raise HTTPException(422, 'At least one account is required')
    jobs = await NotebookService(session).trigger_many(accounts, req.notebook_id)
    return jobs[0] if len(jobs) == 1 else jobs
@router.get('/status/{owner}/{slug}')
async def status(owner: str, slug: str, account: str, session: AsyncSession=Depends(get_session)):
    acc,key=await AccountService(session).get_credentials(account)
    api=KaggleClientFactory().create(acc.kaggle_username, key)
    return get_kernel_status(api, f'{owner}/{slug}')
