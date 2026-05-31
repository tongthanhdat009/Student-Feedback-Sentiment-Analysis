import json
import shutil
from pathlib import Path
from uuid import UUID

from ..config import get_settings
from .notebook_inventory import NotebookInventory


class NotebookStaging:
    def __init__(self, staging_root: str | None = None, inventory: NotebookInventory | None = None):
        self.staging_root = Path(staging_root or '../storage/staging')
        self.inventory = inventory or NotebookInventory()

    def stage(self, notebook_id: str, job_id: UUID | str, kaggle_username: str) -> tuple[Path, str, int | None]:
        source = self.inventory.get_folder(notebook_id)
        manifest = self.inventory.get_manifest(notebook_id)
        entry_file = manifest['entry_file']
        staging = self.staging_root / str(job_id)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        for name in [entry_file, 'kernel-metadata.json', 'notebook.yaml']:
            shutil.copy2(source / name, staging / name)

        meta_path = staging / 'kernel-metadata.json'
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        run_slug = f'{notebook_id}-{str(job_id)[:8]}'
        kaggle_ref = f'{kaggle_username}/{run_slug}'
        metadata['id'] = kaggle_ref
        if metadata.get('title'):
            metadata['title'] = f"{metadata['title']} - {str(job_id)[:8]}"
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return staging, kaggle_ref, manifest.get('default_timeout_seconds')
