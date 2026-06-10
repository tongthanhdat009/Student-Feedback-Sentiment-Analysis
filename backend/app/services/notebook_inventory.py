from pathlib import Path
from ..config import get_settings
from ..utils.path_guard import safe_child
from .notebook_registry import NotebookRegistry, is_safe_slug


class NotebookInventory:
    def __init__(self, root: str | None = None):
        configured = Path(root or get_settings().kaggle_notebook_dir)
        if not configured.is_absolute():
            candidates = [
                Path.cwd() / configured,
                Path(__file__).resolve().parents[3] / configured,
                Path(__file__).resolve().parents[3] / 'notebook' / 'kaggle',
            ]
            self.root = next((p for p in candidates if p.exists()), candidates[-1])
        else:
            self.root = configured
        self.registry = NotebookRegistry()

    def list(self):
        root = self.root.resolve(); items=[]
        if not root.exists(): return []
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            v = self.registry.validate_folder(folder)
            items.append({
                'slug': v.slug,
                'notebook_id': v.slug,
                'path': v.path,
                'title': v.title,
                'valid': v.valid,
                'errors': v.errors,
                'manifest': v.manifest,
                'metadata': v.metadata,
            })
        return items

    def validate(self, notebook_id: str):
        folder = safe_child(self.root, notebook_id)
        if not folder.exists() or not folder.is_dir() or not is_safe_slug(notebook_id):
            return {'slug': notebook_id, 'notebook_id': notebook_id, 'path': str(folder), 'title': None, 'valid': False, 'errors': ['Notebook not found'], 'manifest': {}, 'metadata': {}}
        v = self.registry.validate_folder(folder)
        return {'slug': v.slug, 'notebook_id': v.slug, 'path': v.path, 'title': v.title, 'valid': v.valid, 'errors': v.errors, 'manifest': v.manifest, 'metadata': v.metadata}

    def get_folder(self, notebook_id: str) -> Path:
        folder = safe_child(self.root, notebook_id)
        result = self.validate(notebook_id)
        if not result['valid']:
            raise FileNotFoundError('; '.join(result['errors']))
        return folder

    def get_manifest(self, notebook_id: str):
        return self.registry.load_manifest(self.get_folder(notebook_id))
