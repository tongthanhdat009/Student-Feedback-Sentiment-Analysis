import asyncio, json, shutil, zipfile
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
from .training_result_parser import parse_training_results




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





def describe_kaggle_status(value) -> str:
    if isinstance(value, dict):
        status = value.get('status') or value.get('state') or value.get('statusName')
        failure = value.get('failureMessage') or value.get('errorMessage') or value.get('message')
        if failure:
            return f'Kaggle {status or "ERROR"}: {failure}'
        if status and str(status).upper() in {'ERROR', 'FAILED', 'FAILURE'}:
            return 'Kaggle job failed without a failureMessage. Open the Kaggle kernel run log for details.'
        return str(value)
    if hasattr(value, 'failureMessage') and getattr(value, 'failureMessage'):
        return f'Kaggle {getattr(value, "status", "ERROR")}: {getattr(value, "failureMessage")}'
    return str(value)


def normalize_kaggle_status(value):

    text = str(value or '').lower()

    if hasattr(value, 'status'):

        text = str(value.status or '').lower()

    elif isinstance(value, dict):

        text = str(value.get('status') or value.get('state') or value).lower()

    if any(x in text for x in ['complete', 'success', 'succeeded']): return 'completed'

    if any(x in text for x in ['fail', 'error', 'cancel']): return 'failed'

    return 'running'






def job_s3_prefix(account_name: str, target_ref: str, job_id) -> str:
    return f'kaggle-outputs/{account_name}/{target_ref}/{job_id}'


def choose_primary_artifact(out: Path, patterns: list[str] | None = None) -> Path | None:
    files = [p for p in out.rglob('*') if p.is_file()]
    if patterns:
        for pattern in patterns:
            match = next((p for p in files if p.match(pattern) or p.name == pattern), None)
            if match: return match
    priority = ['.zip','.pt','.bin','.pkl','.joblib','.csv','.json']
    for suffix in priority:
        match = next((p for p in files if p.suffix.lower() == suffix), None)
        if match: return match
    return None

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
            yield mapped, describe_kaggle_status(status)

            if mapped in {'completed', 'failed'}: return

            await asyncio.sleep(self.poll_interval_seconds)

        yield 'failed', f'Timeout exceeded after {timeout_seconds} seconds'



    async def _run_trigger(self, job_id):

        async with AsyncSessionLocal() as s:

            repo=JobRepository(s); job=await repo.get(job_id)

            if not job: return
            folder = None
            job.status='staging'; job.started_at=now(); await repo.save(job)
            try:
                acc=await AccountRepository(s).get(job.account_id)

                if not acc and job.message:

                    acc=await AccountRepository(s).get_by_name(job.message)

                if not acc: raise RuntimeError('Account not found')

                from ..utils.encryption import EncryptionService

                key=EncryptionService(get_settings().fernet_key).decrypt(acc.kaggle_key_encrypted)

                api=await run_in_threadpool(KaggleClientFactory().create, acc.kaggle_username, key)

                dataset_source = (job.result_metadata or {}).get('dataset_source')
                folder, kaggle_ref, timeout_seconds = NotebookStaging().stage(job.target_ref, job.id, acc.kaggle_username, dataset_source)
                job.staging_path=str(folder); job.kaggle_ref=kaggle_ref; job.timeout_seconds=timeout_seconds or 3600; await repo.save(job)
                staging_prefix = f'{job_s3_prefix(acc.name, job.target_ref, job.id)}/staging/'
                try:
                    uploaded_staging = await run_in_threadpool(S3Service().upload_directory, folder, staging_prefix)
                except Exception as exc:
                    raise RuntimeError(f'Failed to upload staging files to S3: {exc}') from exc
                if not uploaded_staging:
                    raise RuntimeError('Failed to upload staging files to S3: no files were uploaded')
                job.staging_s3_prefix = staging_prefix; job.message=f'Uploaded {len(uploaded_staging)} staging file(s) to S3: {staging_prefix}'; await repo.save(job)
                push_response = await run_in_threadpool(api.kernels_push, str(folder))

                job.status='pushed'; job.message=f'Pushed {kaggle_ref}: {push_response}'; await repo.save(job)

                async for mapped, raw in self._poll_kernel(api, kaggle_ref, job.timeout_seconds or 3600):

                    job.status=mapped; job.last_polled_at=now(); job.message=str(raw); await repo.save(job)

                    if mapped in {'completed','failed'}: break

            except Exception as exc:
                job.status='failed'; job.message=format_job_error(exc)
            finally:
                if folder:
                    await run_in_threadpool(shutil.rmtree, folder, True)
                    job.staging_path = None
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

                if not acc and job.message:

                    acc=await AccountRepository(s).get_by_name(job.message)

                if not acc: raise RuntimeError('Account not found')

                from ..utils.encryption import EncryptionService

                key=EncryptionService(get_settings().fernet_key).decrypt(acc.kaggle_key_encrypted)

                api=await run_in_threadpool(KaggleClientFactory().create, acc.kaggle_username, key)

                remote_status = await run_in_threadpool(get_kernel_status, api, job.kaggle_ref)

                if normalize_kaggle_status(remote_status) != 'completed':

                    raise RuntimeError(f'Kaggle output not ready: {remote_status}')

                out=Path(get_settings().kaggle_output_dir)/str(job.id); out.mkdir(parents=True, exist_ok=True)

                await run_in_threadpool(api.kernels_output, job.kaggle_ref, path=str(out))

                patterns = []
                manifest_path = Path(job.staging_path or '') / 'notebook.yaml'
                if manifest_path.exists():
                    try:
                        import yaml
                        manifest = yaml.safe_load(manifest_path.read_text(encoding='utf-8')) or {}
                        patterns = list(manifest.get('artifacts') or [])
                    except Exception:
                        patterns = []
                primary=choose_primary_artifact(out, patterns)
                if not primary:
                    primary=out/'output.zip'
                    with zipfile.ZipFile(primary, 'w') as z:
                        for p in out.rglob('*'):
                            if p.is_file() and p != primary: z.write(p, p.relative_to(out))
                svc=S3Service()
                base_prefix = job_s3_prefix(acc.name, job.target_ref, job.id)
                output_prefix=f'{base_prefix}/artifacts/'
                uploaded = await run_in_threadpool(svc.upload_directory, out, output_prefix)
                obj=f'{output_prefix}{primary.relative_to(out).as_posix()}'
                if obj not in uploaded: obj=await run_in_threadpool(svc.upload_file, primary, obj)
                results = parse_training_results(out)
                results['artifacts'] = uploaded
                metadata_dir = out / '_metadata'; metadata_dir.mkdir(exist_ok=True)
                normalized = metadata_dir / 'training_results.json'
                normalized.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
                await run_in_threadpool(svc.upload_file, normalized, f'{base_prefix}/metadata/training_results.json')
                url, exp = await run_in_threadpool(svc.presign_get, obj)
                job.output_path=str(out); job.output_s3_prefix=output_prefix; job.result_metadata=results; job.s3_object_key=obj; job.s3_presigned_url=url; job.s3_presigned_url_expires_at=exp; job.status='completed'

            except Exception as exc:

                job.status='failed'; job.message=format_job_error(exc)

            job.finished_at=now(); await repo.save(job)

worker = JobWorker()
