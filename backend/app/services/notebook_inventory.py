import json
from pathlib import Path
from ..config import get_settings
from ..utils.path_guard import safe_child

class NotebookInventory:
    def __init__(self, root: str | None = None): self.root = Path(root or get_settings().kaggle_notebook_dir)
    def list(self):
        root = self.root.resolve(); items=[]
        if not root.exists(): return []
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            nb = folder / 'notebook.ipynb'; meta = folder / 'kernel-metadata.json'
            if nb.exists() and meta.exists():
                data = json.loads(meta.read_text(encoding='utf-8'))
                items.append({'notebook_id': folder.name, 'path': str(folder), 'metadata': data})
        return items
    def get_folder(self, notebook_id: str) -> Path:
        folder = safe_child(self.root, notebook_id)
        if not (folder / 'notebook.ipynb').exists() or not (folder / 'kernel-metadata.json').exists():
            raise FileNotFoundError('Notebook not found')
        return folder
