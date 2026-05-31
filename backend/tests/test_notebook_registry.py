import json
from pathlib import Path

from app.services.notebook_registry import NotebookRegistry, is_safe_slug


def write_notebook(folder: Path, slug='demo'):
    folder.mkdir()
    (folder / 'notebook.ipynb').write_text('{}', encoding='utf-8')
    (folder / 'kernel-metadata.json').write_text(json.dumps({'id': 'u/demo', 'title': 'Demo', 'code_file': 'notebook.ipynb', 'kernel_type': 'notebook'}), encoding='utf-8')
    (folder / 'notebook.yaml').write_text(f'slug: {slug}\ntitle: Demo\nentry_file: notebook.ipynb\n', encoding='utf-8')


def test_safe_slug_rules():
    assert is_safe_slug('phobert-baseline')
    assert not is_safe_slug('../x')
    assert not is_safe_slug('/x')
    assert not is_safe_slug('a/b')


def test_valid_folder(tmp_path):
    folder = tmp_path / 'demo'
    write_notebook(folder)
    result = NotebookRegistry().validate_folder(folder)
    assert result.valid is True
    assert result.manifest['slug'] == 'demo'
    assert result.metadata['kernel_type'] == 'notebook'


def test_invalid_manifest_slug(tmp_path):
    folder = tmp_path / 'demo'
    write_notebook(folder, slug='other')
    result = NotebookRegistry().validate_folder(folder)
    assert result.valid is False
    assert 'notebook.yaml.slug must match folder name' in result.errors
