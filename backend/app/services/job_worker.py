import asyncio, zipfile
from datetime import datetime, timezone
from pathlib import Path
from starlette.concurrency import run_in_threadpool
from ..database import AsyncSessionLocal
from ..models import KaggleJob
from ..repositories.job_repository import JobRepository
from ..repositories.account_repository import AccountRepository
from ..config import get_settings
from .kaggle_client_factory import KaggleClientFactory
from .notebook_staging import NotebookStaging
from .s3_service import S3Service

def now(): return datetime.now(timezone.utc)

class JobWorker:
    def enqueue_trigger(self, job_id): asyncio.create_task(self._run_trigger(job_id))
    def enqueue_download(self, job_id): asyncio.create_task(self._run_download(job_id))
    async def recover(self):
        async with AsyncSessionLocal() as s:
            repo=JobRepository(s)
            for job in await repo.stale_running():
                if job.status == 'running':
                    job.status='failed'; job.message='Interrupted by server restart'; job.finished_at=now(); await repo.save(job)
                elif job.status == 'pending': self.enqueue_trigger(job.id)
    async def _run_trigger(self, job_id):
        async with AsyncSessionLocal() as s:
            repo=JobRepository(s); job=await repo.get(job_id)
            if not job: return
            job.status='running'; job.started_at=now(); await repo.save(job)
            try:
                acc=await AccountRepository(s).get_by_name(job.message or '')
                if not acc: raise RuntimeError('Account not found')
                from ..utils.encryption import EncryptionService
                key=EncryptionService(get_settings().fernet_key).decrypt(acc.kaggle_key_encrypted)
                api=await run_in_threadpool(KaggleClientFactory().create, acc.kaggle_username, key)
                folder=NotebookStaging().stage(job.target_ref)
                await run_in_threadpool(api.kernels_push, str(folder))
                job.status='completed'; job.message=f'{acc.kaggle_username}/{job.target_ref}'
            except Exception as exc:
                job.status='failed'; job.message=str(exc)
            job.finished_at=now(); await repo.save(job)
    async def _run_download(self, job_id):
        async with AsyncSessionLocal() as s:
            repo=JobRepository(s); job=await repo.get(job_id)
            if not job: return
            job.status='running'; job.started_at=now(); await repo.save(job)
            try:
                acc=job.account
                if not acc: raise RuntimeError('Account not found')
                from ..utils.encryption import EncryptionService
                key=EncryptionService(get_settings().fernet_key).decrypt(acc.kaggle_key_encrypted)
                api=await run_in_threadpool(KaggleClientFactory().create, acc.kaggle_username, key)
                out=Path(get_settings().kaggle_output_dir)/str(job.id); out.mkdir(parents=True, exist_ok=True)
                await run_in_threadpool(api.kernels_output, job.message or job.target_ref, path=str(out))
                primary=next((p for p in out.rglob('*') if p.is_file() and p.suffix.lower() in {'.zip','.pt','.bin','.pkl','.joblib'}), None)
                if not primary:
                    primary=out/'output.zip'
                    with zipfile.ZipFile(primary, 'w') as z:
                        for p in out.rglob('*'):
                            if p.is_file() and p != primary: z.write(p, p.relative_to(out))
                keyname=f'kaggle-outputs/{acc.name}/{job.target_ref}/{job.id}/{primary.name}'
                svc=S3Service(); obj=await run_in_threadpool(svc.upload_file, primary, keyname)
                url, exp = await run_in_threadpool(svc.presign_get, obj)
                job.output_path=str(out); job.s3_object_key=obj; job.s3_presigned_url=url; job.s3_presigned_url_expires_at=exp; job.status='completed'
            except Exception as exc:
                job.status='failed'; job.message=str(exc)
            job.finished_at=now(); await repo.save(job)
worker = JobWorker()
