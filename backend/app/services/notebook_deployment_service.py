from __future__ import annotations
import asyncio, json, re, shutil
from pathlib import Path
from datetime import datetime, timezone
from fastapi import HTTPException
from requests import HTTPError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool
from ..models import NotebookDeployment
from ..repositories.account_repository import AccountRepository
from ..repositories.notebook_deployment_repository import NotebookDeploymentRepository
from ..config import get_settings
from ..utils.encryption import EncryptionService
from .kaggle_client_factory import KaggleClientFactory
from .notebook_inventory import NotebookInventory
from .notebook_staging import NotebookStaging, resolve_dataset_source
from .job_worker import describe_kaggle_status, get_kernel_status, normalize_kaggle_status


def now(): return datetime.now(timezone.utc)


def slugify(value: str) -> str:
    value = re.sub(r'[^a-zA-Z0-9_-]+', '-', value.strip().lower()).strip('-')
    return value or 'notebook'


def normalize_remote_slug(value: str) -> str:
    value = (value or '').strip()
    if '/' in value:
        value = value.rsplit('/', 1)[-1]
    return slugify(value)


def stable_remote_slug(remote_slug: str | None, notebook_id: str, title: str | None = None) -> str:
    slug = normalize_remote_slug(remote_slug) if remote_slug else slugify(title or notebook_id)
    return slug[:50].rstrip('-') or 'notebook'


def stable_remote_title(remote_slug: str) -> str:
    # Kaggle warns/fails when title slug does not resolve to metadata id.
    # Keep title identical to slug (and <= 50 chars) so Kaggle-derived slug matches.
    return remote_slug[:50].rstrip('-') or 'notebook'


def auto_remote_slug(notebook_id: str, title: str | None, at: datetime | None = None) -> str:
    base = slugify(title or notebook_id)
    timestamp = (at or now()).strftime('%Y%m%d%H%M%S')
    suffix = f'-{timestamp}'
    return f'{base[:63 - len(suffix)].rstrip("-")}{suffix}'


def remote_slug_with_sync_timestamp(remote_slug: str | None, notebook_id: str, title: str | None, at: datetime | None = None) -> str:
    timestamp = (at or now()).strftime('%Y%m%d%H%M')
    base = normalize_remote_slug(remote_slug) if remote_slug else slugify(title or notebook_id)
    suffix = f'-{timestamp}'
    return f'{base[:63 - len(suffix)].rstrip("-")}{suffix}'


def title_with_sync_timestamp(base_title: str | None, at: datetime | None = None) -> str | None:
    if not base_title:
        return None
    timestamp = (at or now()).strftime('%Y%m%d %H%M')
    suffix = f' {timestamp}'
    return f'{base_title[:50 - len(suffix)].rstrip()}{suffix}'


def http_error_detail(exc: HTTPError) -> tuple[int | None, str]:
    response = getattr(exc, 'response', None)
    code = getattr(response, 'status_code', None)
    detail = str(exc)
    if response is not None:
        try:
            detail = response.text or detail
        except Exception:
            pass
    return code, detail


def is_kaggle_no_runs_found_error(exc: Exception) -> bool:
    detail = kaggle_status_poll_error_detail(exc) if isinstance(exc, HTTPError) else str(exc)
    return 'No runs found for this kernel' in detail


def is_kaggle_status_poll_access_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        code, _ = http_error_detail(exc)
        return code in {403, 404}
    message = str(exc)
    return (
        isinstance(exc, ValueError)
        and 'Cannot access kernel' in message
        and ('Permission' in message or 'wrong kernel slug' in message or 'private' in message)
    )


def kaggle_status_poll_error_detail(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        return http_error_detail(exc)[1]
    return str(exc)


def kaggle_code_url(kaggle_ref: str) -> str:
    return f'https://www.kaggle.com/code/{kaggle_ref}'




def kernel_list_item_ref(item) -> str:
    if isinstance(item, dict):
        return str(item.get('ref') or item.get('id') or item.get('kernelRef') or '')
    return str(getattr(item, 'ref', None) or getattr(item, 'id', None) or getattr(item, 'kernelRef', None) or '')


def kernel_list_item_slug(item) -> str:
    ref = kernel_list_item_ref(item)
    if '/' in ref:
        return ref.rsplit('/', 1)[-1]
    if isinstance(item, dict):
        return str(item.get('slug') or item.get('currentUrlSlug') or '')
    return str(getattr(item, 'slug', None) or getattr(item, 'currentUrlSlug', None) or '')


def list_contains_kernel(items, kaggle_ref: str, remote_slug: str) -> bool:
    return any(kernel_list_item_ref(item) == kaggle_ref or kernel_list_item_slug(item) == remote_slug for item in (items or []))


async def verify_kernel_visible(api, kaggle_ref: str, remote_slug: str, attempts: int = 3) -> tuple[bool, str]:
    # Kaggle list/search can lag after SaveKernel, especially for private notebooks.
    # Treat direct status access as enough proof; only return False when both list and status fail.
    last_message = ''
    for attempt in range(1, attempts + 1):
        try:
            try:
                items = await run_in_threadpool(api.kernels_list, mine=True, search=remote_slug, page_size=20)
            except TypeError:
                items = await run_in_threadpool(api.kernels_list, mine=True, page_size=20)
            if list_contains_kernel(items, kaggle_ref, remote_slug):
                return True, 'Kaggle notebook visible in account kernel list'
            last_message = f'not found in kernels_list attempt {attempt}/{attempts}'
        except Exception as exc:
            last_message = f'kernels_list failed attempt {attempt}/{attempts}: {exc}'
        if attempt < attempts:
            await asyncio.sleep(5)
    try:
        status = await run_in_threadpool(get_kernel_status, api, kaggle_ref)
        return True, f'Kaggle notebook not listed yet, but direct status is readable: {describe_kaggle_status(status)}'
    except Exception as exc:
        if is_kaggle_no_runs_found_error(exc):
            return True, f'Kaggle notebook pushed; no run is visible yet: {kaggle_status_poll_error_detail(exc)}'
        if is_kaggle_status_poll_access_error(exc):
            return False, f'Kaggle notebook is not accessible after push: {kaggle_status_poll_error_detail(exc)}'
        return False, f'Kaggle push returned success, but {kaggle_ref} was not found in kernels_list ({last_message}) and direct status failed: {exc}'

def sync_output_dir(account_name: str, notebook_id: str, remote_slug: str) -> Path:
    safe_account = slugify(account_name)
    safe_notebook = slugify(notebook_id)
    safe_slug = slugify(remote_slug)
    return Path(get_settings().kaggle_output_dir) / 'sync' / safe_account / safe_notebook / safe_slug


def sync_debug_dir(account_name: str, notebook_id: str, remote_slug: str) -> Path:
    return sync_output_dir(account_name, notebook_id, remote_slug) / '_debug'


def write_local_debug_file(account_name: str, notebook_id: str, remote_slug: str, name: str, content: str) -> str:
    out = sync_debug_dir(account_name, notebook_id, remote_slug)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(content, encoding='utf-8')
    return str(path)


async def capture_kaggle_debug_logs(api, account_name: str, notebook_id: str, remote_slug: str, kaggle_ref: str, reason: str) -> dict:
    debug = {'reason': reason}
    try:
        logs = await run_in_threadpool(api.kernels_logs, kaggle_ref)
        debug['log_path'] = write_local_debug_file(account_name, notebook_id, remote_slug, 'kaggle.log', logs or '')
        debug['log_tail'] = (logs or '')[-4000:]
    except Exception as exc:
        debug['log_error'] = str(exc)
        debug['log_error_path'] = write_local_debug_file(account_name, notebook_id, remote_slug, 'kaggle_log_error.txt', str(exc))
    return debug


async def download_sync_outputs(api, account_name: str, notebook_id: str, remote_slug: str, kaggle_ref: str) -> str:
    out = sync_output_dir(account_name, notebook_id, remote_slug)
    out.mkdir(parents=True, exist_ok=True)
    await run_in_threadpool(api.kernels_output, kaggle_ref, path=str(out))
    return str(out)


def write_sync_manifest(deployment: NotebookDeployment) -> None:
    path = Path(get_settings().kaggle_output_dir) / 'synced_notebooks.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        try:
            rows = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            rows = []
    rows = [row for row in rows if row.get('deployment_id') != str(deployment.id)]
    metadata = deployment.deployment_metadata or {}
    rows.append({
        'deployment_id': str(deployment.id),
        'account_id': str(deployment.account_id),
        'notebook_id': deployment.notebook_id,
        'kaggle_ref': deployment.kaggle_ref,
        'url': metadata.get('kaggle_url') or kaggle_code_url(deployment.kaggle_ref),
        'remote_slug': deployment.remote_slug,
        'remote_title': deployment.remote_title,
        'status': deployment.last_status,
        'local_output_path': metadata.get('local_output_path'),
        'debug_log_path': metadata.get('debug_log_path'),
        'debug_log_tail': metadata.get('debug_log_tail'),
        'last_synced_at': deployment.last_synced_at.isoformat() if deployment.last_synced_at else None,
        'updated_at': deployment.updated_at.isoformat() if deployment.updated_at else None,
    })
    rows.sort(key=lambda row: (row.get('notebook_id') or '', row.get('updated_at') or ''), reverse=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class NotebookDeploymentService:
    def __init__(self, session: AsyncSession): self.session = session

    async def list(self, account: str | None = None):
        repo = NotebookDeploymentRepository(self.session); await repo.ensure_schema()
        account_id = None
        if account:
            acc = await AccountRepository(self.session).get_by_name(account)
            if not acc: raise HTTPException(404, 'Account not found')
            account_id = acc.id
        return await repo.list(account_id)

    async def sync(self, notebook_id: str, account_name: str, remote_slug: str | None = None, title: str | None = None, dataset_sources: list[str] | None = None, enable_gpu: bool | None = None):
        validation = NotebookInventory().validate(notebook_id)
        if not validation['valid']:
            raise HTTPException(422, {'slug': notebook_id, 'errors': validation['errors']})
        acc = await AccountRepository(self.session).get_by_name(account_name)
        if not acc: raise HTTPException(404, 'Account not found')
        dataset_source = resolve_dataset_source(dataset_sources[0], required=False) if dataset_sources else None
        ts = now()
        base_title = title or validation.get('title')
        remote_slug = stable_remote_slug(remote_slug, notebook_id, base_title)
        kaggle_ref = f'{acc.kaggle_username}/{remote_slug}'
        effective_title = stable_remote_title(remote_slug)
        key = EncryptionService(get_settings().fernet_key).decrypt(acc.kaggle_key_encrypted)
        api = await run_in_threadpool(KaggleClientFactory().create, acc.kaggle_username, key)
        folder, _, _ = NotebookStaging().stage(
            notebook_id, f'sync-{notebook_id}-{remote_slug}', acc.kaggle_username, dataset_source,
            fixed_kaggle_ref=kaggle_ref, remote_slug=remote_slug, title=effective_title,
            append_job_suffix=False, require_dataset_source=False, is_private=True,
        )
        repo = NotebookDeploymentRepository(self.session); await repo.ensure_schema()
        deployment = await repo.get_for(acc.id, notebook_id)
        if not deployment:
            deployment = NotebookDeployment(account_id=acc.id, notebook_id=notebook_id, kaggle_ref=kaggle_ref, remote_slug=remote_slug)
        deployment.kaggle_ref = kaggle_ref
        deployment.remote_slug = remote_slug
        deployment.remote_title = effective_title
        deployment.source_path = validation.get('path')
        deployment.is_active = True
        deployment.last_synced_at = ts
        deployment.last_status = 'syncing'
        deployment.deployment_metadata = {
            'dataset_sources': dataset_sources or [],
            'enable_gpu': enable_gpu,
            'account': account_name,
            'kaggle_url': kaggle_code_url(kaggle_ref),
            'is_private': True,
        }
        deployment = await repo.save(deployment)
        write_sync_manifest(deployment)
        timeout = validation.get('manifest', {}).get('default_timeout_seconds') or 3600
        try:
            await run_in_threadpool(api.kernels_push, str(folder), str(timeout), 'NvidiaTeslaT4' if enable_gpu else None)
            visible, visibility_message = await verify_kernel_visible(api, kaggle_ref, remote_slug)
            metadata = dict(deployment.deployment_metadata or {})
            metadata['kaggle_visibility_check'] = visibility_message
            deployment.deployment_metadata = metadata
            if not visible:
                deployment.last_status = 'failed'
                deployment = await repo.save(deployment)
                write_sync_manifest(deployment)
                raise HTTPException(status_code=502, detail=visibility_message)
            if 'not listed yet' in visibility_message:
                metadata['status_poll_warning'] = visibility_message
                deployment.deployment_metadata = metadata
            deployment.last_status = 'pushed'
            deployment = await repo.save(deployment)
            write_sync_manifest(deployment)
            # Sync should only push/update the Kaggle notebook and return quickly.
            # Long Kaggle execution/output polling is handled by explicit trigger jobs.
            return deployment
            started = now()
            while (now() - started).total_seconds() <= int(timeout):
                try:
                    status = await run_in_threadpool(get_kernel_status, api, kaggle_ref)
                except (HTTPError, ValueError) as exc:
                    if is_kaggle_status_poll_access_error(exc):
                        deployment.last_status = 'running'
                        metadata = dict(deployment.deployment_metadata or {})
                        metadata['last_kaggle_status'] = kaggle_status_poll_error_detail(exc)
                        metadata['status_poll_warning'] = 'Kaggle status polling is not readable yet; continuing to poll after push.'
                        deployment.deployment_metadata = metadata
                        deployment = await repo.save(deployment)
                        write_sync_manifest(deployment)
                        await asyncio.sleep(30)
                        continue
                    raise
                mapped = normalize_kaggle_status(status)
                deployment.last_status = mapped if mapped in {'completed', 'failed'} else 'running'
                metadata = dict(deployment.deployment_metadata or {})
                metadata['last_kaggle_status'] = describe_kaggle_status(status)
                deployment.deployment_metadata = metadata
                deployment = await repo.save(deployment)
                write_sync_manifest(deployment)
                if mapped == 'completed':
                    metadata = dict(deployment.deployment_metadata or {})
                    try:
                        metadata['local_output_path'] = await download_sync_outputs(api, account_name, notebook_id, remote_slug, kaggle_ref)
                    except Exception as exc:
                        metadata['local_output_error'] = str(exc)
                    deployment.deployment_metadata = metadata
                    deployment = await repo.save(deployment)
                    write_sync_manifest(deployment)
                    break
                if mapped == 'failed':
                    debug = await capture_kaggle_debug_logs(api, account_name, notebook_id, remote_slug, kaggle_ref, describe_kaggle_status(status))
                    metadata = dict(deployment.deployment_metadata or {})
                    metadata['debug_log_path'] = debug.get('log_path')
                    metadata['debug_log_tail'] = debug.get('log_tail')
                    metadata['debug_log_error'] = debug.get('log_error')
                    deployment.deployment_metadata = metadata
                    deployment = await repo.save(deployment)
                    write_sync_manifest(deployment)
                    break
                await asyncio.sleep(30)
            else:
                deployment.last_status = 'failed'
                metadata = dict(deployment.deployment_metadata or {})
                metadata['last_kaggle_status'] = f'Sync polling timeout after {timeout} seconds'
                deployment.deployment_metadata = metadata
                deployment = await repo.save(deployment)
        except HTTPError as exc:
            _, detail = http_error_detail(exc)
            deployment.last_status = 'failed'
            metadata = dict(deployment.deployment_metadata or {})
            metadata['last_kaggle_status'] = detail
            deployment.deployment_metadata = metadata
            deployment = await repo.save(deployment)
            write_sync_manifest(deployment)
            raise HTTPException(status_code=502, detail=f'Kaggle SaveKernel/status failed: {detail}') from exc
        finally:
            await run_in_threadpool(shutil.rmtree, folder, True)
        return deployment
