from datetime import datetime, timezone
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from ..models import KaggleDataset
from ..repositories.dataset_repository import DatasetRepository
from ..schemas.dataset import DatasetCreate, DatasetUpdate
from .dataset_validation import validate_local_uit_vsfc

PLACEHOLDER_DATASET_SOURCE = 'owner/dataset-slug'


def normalize_dataset_ref(dataset_ref: str) -> str:
    dataset_ref = (dataset_ref or '').strip()
    if not dataset_ref:
        raise HTTPException(400, 'dataset_ref is required')
    if dataset_ref == PLACEHOLDER_DATASET_SOURCE:
        raise HTTPException(400, 'dataset_ref must be a real Kaggle dataset ref')
    parts = dataset_ref.split('/')
    if len(parts) != 2 or not all(parts):
        raise HTTPException(400, 'dataset_ref must use owner/dataset-slug format')
    return dataset_ref


class DatasetService:
    def __init__(self, session: AsyncSession): self.repo = DatasetRepository(session)

    async def ensure_schema(self):
        await self.repo.ensure_schema()

    async def list_datasets(self, active_only: bool = False):
        await self.ensure_schema()
        return await (self.repo.list_active() if active_only else self.repo.list())

    async def create_dataset(self, data: DatasetCreate):
        await self.ensure_schema()
        dataset_ref = normalize_dataset_ref(data.dataset_ref)
        slug = data.slug.strip()
        if await self.repo.get_by_slug(slug): raise HTTPException(409, 'Dataset slug exists')
        if await self.repo.get_by_ref(dataset_ref): raise HTTPException(409, 'Dataset ref exists')
        dataset = KaggleDataset(
            slug=slug,
            dataset_ref=dataset_ref,
            title=data.title,
            description=data.description,
            local_path=data.local_path,
            status=data.status,
        )
        try:
            return await self.repo.add(dataset)
        except IntegrityError as exc:
            raise HTTPException(409, 'Dataset exists') from exc

    async def update_dataset_by_id(self, dataset_id: str, data: DatasetUpdate):
        await self.ensure_schema()
        dataset = await self.repo.get_by_id_text(dataset_id)
        if not dataset: raise HTTPException(404, 'Dataset not found')
        return await self._apply_update(dataset, data)

    async def delete_dataset_by_id(self, dataset_id: str):
        await self.ensure_schema()
        dataset = await self.repo.get_by_id_text(dataset_id)
        if not dataset: raise HTTPException(404, 'Dataset not found')
        dataset.is_active = False
        dataset.status = 'inactive'
        return await self.repo.save(dataset)

    async def validate_dataset_by_id(self, dataset_id: str):
        await self.ensure_schema()
        dataset = await self.repo.get_by_id_text(dataset_id)
        if not dataset: raise HTTPException(404, 'Dataset not found')
        return await self._run_validation(dataset)

    async def _apply_update(self, dataset: KaggleDataset, data: DatasetUpdate):
        if data.slug is not None: dataset.slug = data.slug.strip()
        if data.dataset_ref is not None: dataset.dataset_ref = normalize_dataset_ref(data.dataset_ref)
        if data.title is not None: dataset.title = data.title
        if data.description is not None: dataset.description = data.description
        if data.local_path is not None: dataset.local_path = data.local_path
        if data.status is not None: dataset.status = data.status
        if data.is_active is not None: dataset.is_active = data.is_active
        try:
            return await self.repo.save(dataset)
        except IntegrityError as exc:
            raise HTTPException(409, 'Dataset exists') from exc

    async def _run_validation(self, dataset: KaggleDataset):
        if not dataset.local_path: raise HTTPException(400, 'local_path is required')
        result = validate_local_uit_vsfc(dataset.local_path)
        dataset.validation_result = result
        dataset.last_validated_at = datetime.now(timezone.utc)
        dataset.status = 'active' if result['valid'] else 'failed'
        return await self.repo.save(dataset)

    async def update_dataset(self, slug: str, data: DatasetUpdate):
        await self.ensure_schema()
        dataset = await self.repo.get_by_slug(slug)
        if not dataset: raise HTTPException(404, 'Dataset not found')
        return await self._apply_update(dataset, data)

    async def delete_dataset(self, slug: str):
        await self.ensure_schema()
        dataset = await self.repo.get_by_slug(slug)
        if not dataset: raise HTTPException(404, 'Dataset not found')
        dataset.is_active = False
        dataset.status = 'inactive'
        return await self.repo.save(dataset)

    async def validate_dataset(self, slug: str):
        await self.ensure_schema()
        dataset = await self.repo.get_by_slug(slug)
        if not dataset: raise HTTPException(404, 'Dataset not found')
        return await self._run_validation(dataset)
