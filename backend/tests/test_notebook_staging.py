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
