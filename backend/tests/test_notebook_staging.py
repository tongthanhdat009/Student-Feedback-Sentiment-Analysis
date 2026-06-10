import json
from pathlib import Path

import pytest

from app.services.notebook_inventory import NotebookInventory
from app.services.notebook_staging import NotebookStaging


def make_folder(root: Path):
    folder = root / 'demo'
    folder.mkdir()
    (folder / 'notebook.ipynb').write_text('{"cells": []}', encoding='utf-8')
    (folder / 'kernel-metadata.json').write_text(json.dumps({'id': 'old/demo', 'title': 'Demo', 'code_file': 'notebook.ipynb', 'kernel_type': 'notebook'}), encoding='utf-8')
    (folder / 'notebook.yaml').write_text('slug: demo\ntitle: Demo\nentry_file: notebook.ipynb\ndefault_timeout_seconds: 123\n', encoding='utf-8')
    return folder




def test_default_staging_root_is_repo_storage():
    staging = NotebookStaging()

    assert staging.staging_root.is_absolute()
    assert staging.staging_root.name == 'staging'
    assert staging.staging_root.parent.name == 'storage'

def test_stage_writes_cp1252_safe_notebook(tmp_path):
    source_root = tmp_path / 'notebooks'
    source_root.mkdir()
    folder = make_folder(source_root)
    (folder / 'notebook.ipynb').write_text(json.dumps({'cells': [{'cell_type': 'markdown', 'source': ['Tiếng Việt – dữ liệu']}]}, ensure_ascii=False), encoding='utf-8')
    staging_root = tmp_path / 'staging'

    staging, _, _ = NotebookStaging(str(staging_root), NotebookInventory(str(source_root))).stage('demo', '12345678-abcd', 'alice', 'alice/uit-vsfc-processed')

    raw = (staging / 'notebook.ipynb').read_bytes()
    raw.decode('cp1252')
    raw.decode('ascii')
    assert b'\\u1ebf' in raw


def test_sync_staging_sets_private_kaggle_metadata(tmp_path):
    source_root = tmp_path / 'notebooks'
    source_root.mkdir()
    make_folder(source_root)
    staging_root = tmp_path / 'staging'

    staging, _, _ = NotebookStaging(str(staging_root), NotebookInventory(str(source_root))).stage(
        'demo', 'sync-demo', 'alice', 'alice/uit-vsfc-processed', is_private=True,
    )

    staged_meta = json.loads((staging / 'kernel-metadata.json').read_text(encoding='utf-8'))
    assert staged_meta['is_private'] is True


def test_sync_debug_helpers_write_log_file(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app.services import notebook_deployment_service as svc

    monkeypatch.setattr(svc, 'get_settings', lambda: SimpleNamespace(kaggle_output_dir=str(tmp_path)))

    path = svc.write_local_debug_file('acc', 'notebook-id', 'remote-slug', 'kaggle.log', 'Traceback: boom')

    assert Path(path).read_text(encoding='utf-8') == 'Traceback: boom'
    assert str(tmp_path / 'sync' / 'acc' / 'notebook-id' / 'remote-slug' / '_debug' / 'kaggle.log') == path


def test_sync_manifest_helpers_write_local_link_file(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from uuid import uuid4

    from app.services import notebook_deployment_service as svc

    monkeypatch.setattr(svc, 'get_settings', lambda: SimpleNamespace(kaggle_output_dir=str(tmp_path)))
    deployment = SimpleNamespace(
        id=uuid4(), account_id=uuid4(), notebook_id='demo', kaggle_ref='alice/demo-202606080905',
        remote_slug='demo-202606080905', remote_title='Demo 20260608 0905', last_status='pushed',
        deployment_metadata={'kaggle_url': 'https://www.kaggle.com/code/alice/demo-202606080905'},
        last_synced_at=datetime(2026, 6, 8, 9, 5, tzinfo=timezone.utc),
        updated_at=datetime(2026, 6, 8, 9, 6, tzinfo=timezone.utc),
    )

    svc.write_sync_manifest(deployment)

    manifest = json.loads((tmp_path / 'synced_notebooks.json').read_text(encoding='utf-8'))
    assert manifest[0]['url'] == 'https://www.kaggle.com/code/alice/demo-202606080905'
    assert manifest[0]['status'] == 'pushed'


def test_sync_handles_kaggle_status_403_as_pushed_warning():
    from pathlib import Path

    source = Path('app/services/notebook_deployment_service.py').read_text(encoding='utf-8')

    assert 'is_kaggle_status_poll_access_error' in source
    assert 'except (HTTPError, ValueError)' in source
    assert "deployment.last_status = 'running'" in source
    assert 'continuing to poll after push' in source


def test_sync_detects_kaggle_status_value_error_as_poll_warning():
    from app.services.notebook_deployment_service import is_kaggle_status_poll_access_error

    exc = ValueError("Cannot access kernel 'jadt145/demo' (Permission 'kernels.get' was denied). It can also occur if the notebook is private.")

    assert is_kaggle_status_poll_access_error(exc) is True


def test_sync_uses_stable_remote_slug_for_code_updates():
    from app.services.notebook_deployment_service import stable_remote_slug, stable_remote_title

    assert stable_remote_slug('phobert-tfidf', 'phobert-tfidf', 'PhoBERT TFIDF') == 'phobert-tfidf'
    assert stable_remote_slug(None, 'phobert-tfidf', 'PhoBERT TFIDF') == 'phobert-tfidf'
    assert stable_remote_title('phobert-tfidf') == 'phobert-tfidf'
    assert stable_remote_title('phobert-baseline-local') == 'phobert-baseline-local'


def test_sync_stable_title_is_slug_based_and_kaggle_safe():
    from app.services.notebook_deployment_service import stable_remote_title

    title = stable_remote_title('phobert-sentiwordnet-refactored-lightfusion-topicanalysis')

    assert title == 'phobert-sentiwordnet-refactored-lightfusion-topica'
    assert len(title) <= 50




def test_sync_visibility_helpers_match_ref_or_slug():
    from app.services.notebook_deployment_service import list_contains_kernel

    items = [
        {'ref': 'jadt145/phobert-tfidf'},
        {'id': 'jadt145/phobert-sentiwordnet'},
        {'slug': 'phobert-baseline-local'},
    ]

    assert list_contains_kernel(items, 'jadt145/phobert-sentiwordnet', 'phobert-sentiwordnet') is True
    assert list_contains_kernel(items, 'jadt145/phobert-baseline-local', 'phobert-baseline-local') is True
    assert list_contains_kernel(items, 'jadt145/missing', 'missing') is False



def test_sync_visibility_accepts_direct_status_when_list_lags():
    import asyncio
    from app.services import notebook_deployment_service as svc

    class Api:
        def kernels_list(self, **kwargs):
            return []
        def kernel_status(self, ref):
            return {'status': 'running'}

    ok, message = asyncio.run(svc.verify_kernel_visible(Api(), 'jadt145/phobert-sentiwordnet', 'phobert-sentiwordnet', attempts=1))

    assert ok is True
    assert 'direct status is readable' in message





def test_sync_visibility_accepts_no_runs_found_after_push():
    import asyncio
    from requests import HTTPError, Response
    from app.services import notebook_deployment_service as svc

    class Api:
        def kernels_list(self, **kwargs):
            return []
        def kernel_status(self, ref):
            response = Response()
            response.status_code = 404
            response._content = b'{"error":{"code":404,"message":"No runs found for this kernel.","status":"NOT_FOUND"}}'
            error = HTTPError('404 Client Error')
            error.response = response
            raise error

    ok, message = asyncio.run(svc.verify_kernel_visible(Api(), 'jadt145/phobert-sentiwordnet', 'phobert-sentiwordnet', attempts=1))

    assert ok is True
    assert 'no run is visible yet' in message

def test_sync_visibility_rejects_inaccessible_status_when_list_lags():
    import asyncio
    from app.services import notebook_deployment_service as svc

    class Api:
        def kernels_list(self, **kwargs):
            return []
        def kernel_status(self, ref):
            raise ValueError('Cannot access kernel. Permission denied or wrong kernel slug')

    ok, message = asyncio.run(svc.verify_kernel_visible(Api(), 'jadt145/phobert-sentiwordnet', 'phobert-sentiwordnet', attempts=1))

    assert ok is False
    assert 'not accessible after push' in message

def test_stage_copies_and_rewrites_metadata(tmp_path):
    source_root = tmp_path / 'notebooks'
    source_root.mkdir()
    source = make_folder(source_root)
    staging_root = tmp_path / 'staging'
    staging, kaggle_ref, timeout = NotebookStaging(str(staging_root), NotebookInventory(str(source_root))).stage('demo', '12345678-abcd', 'alice', 'alice/uit-vsfc-processed')
    assert staging == staging_root / '12345678-abcd'
    assert kaggle_ref == 'alice/demo-12345678'
    assert timeout == 123
    assert (staging / 'notebook.ipynb').exists()
    assert (staging / 'notebook.yaml').exists()
    staged_meta = json.loads((staging / 'kernel-metadata.json').read_text(encoding='utf-8'))
    source_meta = json.loads((source / 'kernel-metadata.json').read_text(encoding='utf-8'))
    assert staged_meta['id'] == 'alice/demo-12345678'
    assert staged_meta['title'] == 'Demo - 12345678'
    assert staged_meta['dataset_sources'] == ['alice/uit-vsfc-processed']
    assert source_meta['id'] == 'old/demo'


def test_stage_rejects_placeholder_dataset(tmp_path):
    source_root = tmp_path / 'notebooks'
    source_root.mkdir()
    make_folder(source_root)
    staging_root = tmp_path / 'staging'
    with pytest.raises(ValueError, match='owner/dataset-slug'):
        NotebookStaging(str(staging_root), NotebookInventory(str(source_root))).stage('demo', '12345678-abcd', 'alice', 'owner/dataset-slug')
