import asyncio, zipfile
from requests import HTTPError
from datetime import datetime, timezone
from pathlib import Path
from starlette.concurrency import run_in_threadpool
from ..database import AsyncSessionLocal
from ..repositories.job_repository import JobRepository
from ..repositories.account_repository import AccountRepository
from ..config import get_settings
from .kaggle_client_factory import KaggleClientFactory
from .notebook_staging import NotebookStaging
from .s3_service import S3Service


def now(): return datetime.now(timezone.utc)


def get_kernel_status(api, ref: str):
    if hasattr(api, 'kernel_status'):
        return api.kernel_status(ref)
    return api.kernels_status(ref)


def format_job_error(exc: Exception) -> str:
    message = str(exc)
    if '401' in message and 'SaveKernel' in message:
        return 'Kaggle push unauthorized. Check Kaggle username/API key, account permissions, and whether this account can create/update kernels.'
    return message


def normalize_kaggle_status(value):
    text = str(value or '').lower()
    if hasattr(value, 'status'):
        text = str(value.status or '').lower()
    elif isinstance(value, dict):
        text = str(value.get('status') or value.get('state') or value).lower()
    if any(x in text for x in ['complete', 'success', 'succeeded']): return 'completed'
    if any(x in text for x in ['fail', 'error', 'cancel']): return 'failed'
    return 'running'


class JobWorker:
    poll_interval_seconds = 30
    status_not_found_grace_seconds = 180
    def enqueue_trigger(self, job_id): asyncio.create_task(self._run_trigger(job_id))
    def enqueue_download(self, job_id): asyncio.create_task(self._run_download(job_id))
    async def recover(self):
        async with AsyncSessionLocal() as s:
            repo=JobRepository(s)
            for job in await repo.stale_running():
                if job.status in ['running','staging','pushed']:
                    job.status='failed'; job.message='Interrupted by server restart'; job.finished_at=now(); await repo.save(job)
                elif job.status == 'pending': self.enqueue_trigger(job.id)

    async def _poll_kernel(self, api, ref: str, timeout_seconds: int):
        started = now()
        not_found_started = None
        while (now() - started).total_seconds() <= timeout_seconds:
            try:
                status = await run_in_threadpool(get_kernel_status, api, ref)
            except HTTPError as exc:
                code = getattr(getattr(exc, 'response', None), 'status_code', None)
                if code == 404:
                    not_found_started = not_found_started or now()
                    elapsed_404 = (now() - not_found_started).total_seconds()
                    if elapsed_404 >= self.status_not_found_grace_seconds:
                        yield 'failed', f'Kaggle kernel session status stayed 404 for {int(elapsed_404)} seconds after push. Check Kaggle UI for ref {ref}; the kernel may not have started.'
                        return
                    status = f'Kaggle status not ready yet (404); waiting for kernel session ({int(elapsed_404)}s)'
                    yield 'running', status
                    await asyncio.sleep(self.poll_interval_seconds)
                    continue
                raise
            mapped = normalize_kaggle_status(status)
            yield mapped, status
            if mapped in {'completed', 'failed'}: return
            await asyncio.sleep(self.poll_interval_seconds)
        yield 'failed', f'Timeout exceeded after {timeout_seconds} seconds'

    async def _run_trigger(self, job_id):
        async with AsyncSessionLocal() as s:
            repo=JobRepository(s); job=await repo.get(job_id)
            if not job: return
            job.status='staging'; job.started_at=now(); await repo.save(job)
            try:
                acc=await AccountRepository(s).get(job.account_id)
                if not acc and job.message:
                    acc=await AccountRepository(s).get_by_name(job.message)
                if not acc: raise RuntimeError('Account not found')
                from ..utils.encryption import EncryptionService
                key=EncryptionService(get_settings().fernet_key).decrypt(acc.kaggle_key_encrypted)
                api=await run_in_threadpool(KaggleClientFactory().create, acc.kaggle_username, key)
                folder, kaggle_ref, timeout_seconds = NotebookStaging().stage(job.target_ref, job.id, acc.kaggle_username)
                job.staging_path=str(folder); job.kaggle_ref=kaggle_ref; job.timeout_seconds=timeout_seconds or 3600; await repo.save(job)
                push_response = await run_in_threadpool(api.kernels_push, str(folder))
                job.status='pushed'; job.message=f'Pushed {kaggle_ref}: {push_response}'; await repo.save(job)
                async for mapped, raw in self._poll_kernel(api, kaggle_ref, job.timeout_seconds or 3600):
                    job.status=mapped; job.last_polled_at=now(); job.message=str(raw); await repo.save(job)
                    if mapped in {'completed','failed'}: break
            except Exception as exc:
                job.status='failed'; job.message=format_job_error(exc)
            job.finished_at=now(); await repo.save(job)

    async def _run_download(self, job_id):
        async with AsyncSessionLocal() as s:
            repo=JobRepository(s); job=await repo.get(job_id)
            if not job: return
            if not job.kaggle_ref:
                job.status='failed'; job.message='Kaggle ref missing'; job.finished_at=now(); await repo.save(job); return
            job.status='running'; job.started_at=now(); await repo.save(job)
            try:
                acc=await AccountRepository(s).get(job.account_id)
                if not acc: raise RuntimeError('Account not found')
                from ..utils.encryption import EncryptionService
                key=EncryptionService(get_settings().fernet_key).decrypt(acc.kaggle_key_encrypted)
                api=await run_in_threadpool(KaggleClientFactory().create, acc.kaggle_username, key)
                remote_status = await run_in_threadpool(get_kernel_status, api, job.kaggle_ref)
                if normalize_kaggle_status(remote_status) != 'completed':
                    raise RuntimeError(f'Kaggle output not ready: {remote_status}')
                out=Path(get_settings().kaggle_output_dir)/str(job.id); out.mkdir(parents=True, exist_ok=True)
                await run_in_threadpool(api.kernels_output, job.kaggle_ref, path=str(out))
                primary=next((p for p in out.rglob('*') if p.is_file() and p.suffix.lower() in {'.zip','.pt','.bin','.pkl','.joblib','.csv','.json'}), None)
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
                job.status='failed'; job.message=format_job_error(exc)
            job.finished_at=now(); await repo.save(job)
worker = JobWorker()
