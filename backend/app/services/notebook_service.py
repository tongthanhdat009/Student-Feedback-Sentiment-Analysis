from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ..models import KaggleJob
from ..repositories.account_repository import AccountRepository
from ..repositories.job_repository import JobRepository
from .notebook_inventory import NotebookInventory
from .job_worker import worker

class NotebookService:
    def __init__(self, session: AsyncSession): self.session=session
    def inventory(self): return NotebookInventory().list()
    def validate(self, notebook_id: str): return NotebookInventory().validate(notebook_id)
    async def trigger_many(self, account_names: list[str], notebook_id: str):
        jobs=[]
        for account_name in account_names:
            jobs.append(await self.trigger(account_name, notebook_id))
        return jobs

    async def trigger(self, account_name: str, notebook_id: str):
        acc=await AccountRepository(self.session).get_by_name(account_name)
        if not acc: raise HTTPException(404, 'Account not found')
        validation = NotebookInventory().validate(notebook_id)
        if not validation['valid']:
            raise HTTPException(422, {'slug': notebook_id, 'errors': validation['errors']})
        job=await JobRepository(self.session).add(KaggleJob(account_id=acc.id, job_type='notebook_trigger', target_ref=notebook_id, status='pending', message=account_name))
        worker.enqueue_trigger(job.id)
        return job
