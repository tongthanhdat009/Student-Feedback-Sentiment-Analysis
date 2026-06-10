from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ..models import KaggleJob
from ..repositories.account_repository import AccountRepository
from ..repositories.job_repository import JobRepository
from .notebook_inventory import NotebookInventory
from .job_worker import worker
from ..repositories.notebook_deployment_repository import NotebookDeploymentRepository

PLACEHOLDER_DATASET_SOURCE = 'owner/dataset-slug'


def normalize_dataset_source(dataset_source: str | None) -> str:
    dataset_source = (dataset_source or '').strip()
    if not dataset_source:
        raise HTTPException(400, 'dataset_source is required')
    if dataset_source == PLACEHOLDER_DATASET_SOURCE:
        raise HTTPException(400, 'dataset_source must be a real Kaggle dataset ref, not owner/dataset-slug')
    parts = dataset_source.split('/')
    if len(parts) != 2 or not all(parts):
        raise HTTPException(400, 'dataset_source must use owner/dataset-slug format')
    return dataset_source


class NotebookService:
    def __init__(self, session: AsyncSession): self.session=session
    def inventory(self): return NotebookInventory().list()
    def validate(self, notebook_id: str): return NotebookInventory().validate(notebook_id)
    async def trigger_many(self, account_names: list[str], notebook_id: str, dataset_source: str | None = None):
        dataset_source = normalize_dataset_source(dataset_source) if dataset_source else None
        jobs=[]
        for account_name in account_names:
            jobs.append(await self.trigger(account_name, notebook_id, dataset_source))
        return jobs

    async def trigger(self, account_name: str, notebook_id: str, dataset_source: str | None = None):
        dataset_source = normalize_dataset_source(dataset_source) if dataset_source else None
        acc=await AccountRepository(self.session).get_by_name(account_name)
        if not acc: raise HTTPException(404, 'Account not found')
        deployment_repo = NotebookDeploymentRepository(self.session); await deployment_repo.ensure_schema()
        if not await deployment_repo.get_for(acc.id, notebook_id):
            raise HTTPException(409, 'Notebook is not synced to this account. Sync first.')
        validation = NotebookInventory().validate(notebook_id)
        if not validation['valid']:
            raise HTTPException(422, {'slug': notebook_id, 'errors': validation['errors']})
        job=await JobRepository(self.session).add(KaggleJob(account_id=acc.id, job_type='notebook_trigger', target_ref=notebook_id, status='pending', message=account_name, result_metadata={'dataset_source': dataset_source}))
        worker.enqueue_trigger(job.id)
        return job
