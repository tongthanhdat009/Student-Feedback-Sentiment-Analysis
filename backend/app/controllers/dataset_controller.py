from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from ..database import get_session
from ..schemas.dataset import DatasetCreate, DatasetRead, DatasetUpdate
from ..services.dataset_service import DatasetService
from ..utils.auth import require_api_key

router=APIRouter(prefix='/api/kaggle/datasets', tags=['datasets'], dependencies=[Depends(require_api_key)])

@router.get('')
async def list_datasets(page: int = 1, page_size: int = 20, active_only: bool = False, session: AsyncSession=Depends(get_session)):
    service = DatasetService(session); start = (page-1)*page_size
    items = await service.list_datasets(active_only, limit=page_size, offset=start)
    total = await service.count_datasets(active_only)
    return {'items': items, 'total': total, 'page': page, 'page_size': page_size}

@router.post('', response_model=DatasetRead)
async def create_dataset(data: DatasetCreate, session: AsyncSession=Depends(get_session)):
    return await DatasetService(session).create_dataset(data)

@router.patch('/id/{dataset_id}', response_model=DatasetRead)
async def update_dataset_by_id(dataset_id: str, data: DatasetUpdate, session: AsyncSession=Depends(get_session)):
    return await DatasetService(session).update_dataset_by_id(dataset_id, data)

@router.delete('/id/{dataset_id}')
async def delete_dataset_by_id(dataset_id: str, session: AsyncSession=Depends(get_session)):
    await DatasetService(session).delete_dataset_by_id(dataset_id); return {'ok': True}

@router.post('/id/{dataset_id}/validate-local', response_model=DatasetRead)
async def validate_dataset_by_id(dataset_id: str, session: AsyncSession=Depends(get_session)):
    return await DatasetService(session).validate_dataset_by_id(dataset_id)

@router.patch('/{slug}', response_model=DatasetRead)
async def update_dataset(slug: str, data: DatasetUpdate, session: AsyncSession=Depends(get_session)):
    return await DatasetService(session).update_dataset(slug, data)

@router.delete('/{slug}')
async def delete_dataset(slug: str, session: AsyncSession=Depends(get_session)):
    await DatasetService(session).delete_dataset(slug); return {'ok': True}

@router.post('/{slug}/validate-local', response_model=DatasetRead)
async def validate_dataset(slug: str, session: AsyncSession=Depends(get_session)):
    return await DatasetService(session).validate_dataset(slug)
