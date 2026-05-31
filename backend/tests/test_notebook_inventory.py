import json
from pathlib import Path

from app.services.notebook_inventory import NotebookInventory


def make_folder(root: Path, name='demo'):
    folder = root / name
    folder.mkdir()
    (folder / 'notebook.ipynb').write_text('{}', encoding='utf-8')
    (folder / 'kernel-metadata.json').write_text(json.dumps({'code_file': 'notebook.ipynb', 'kernel_type': 'notebook', 'title': 'Demo'}), encoding='utf-8')
    (folder / 'notebook.yaml').write_text(f'slug: {name}\ntitle: Demo\nentry_file: notebook.ipynb\n', encoding='utf-8')
    return folder


def test_inventory_valid_shape(tmp_path):
    make_folder(tmp_path)
    items = NotebookInventory(str(tmp_path)).list()
    assert items[0]['slug'] == 'demo'
    assert items[0]['notebook_id'] == 'demo'
    assert items[0]['valid'] is True
    assert items[0]['errors'] == []
    assert items[0]['manifest']['entry_file'] == 'notebook.ipynb'


def test_inventory_reports_missing_manifest(tmp_path):
    folder = tmp_path / 'bad'
    folder.mkdir()
    (folder / 'notebook.ipynb').write_text('{}', encoding='utf-8')
    (folder / 'kernel-metadata.json').write_text('{}', encoding='utf-8')
    item = NotebookInventory(str(tmp_path)).list()[0]
    assert item['valid'] is False
    assert 'notebook.yaml is required' in item['errors']
