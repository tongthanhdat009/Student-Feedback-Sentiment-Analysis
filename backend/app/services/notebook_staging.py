import json

import shutil

from pathlib import Path

from uuid import UUID



from ..config import get_settings

from .notebook_inventory import NotebookInventory





PLACEHOLDER_DATASET_SOURCE = 'owner/dataset-slug'


def resolve_dataset_source(dataset_source: str | None = None) -> str:
    resolved = (dataset_source or get_settings().kaggle_default_dataset_source or '').strip()
    if not resolved:
        raise ValueError('dataset_source is required')
    if resolved == PLACEHOLDER_DATASET_SOURCE:
        raise ValueError('dataset_source must be a real Kaggle dataset ref, not owner/dataset-slug')
    parts = resolved.split('/')
    if len(parts) != 2 or not all(parts):
        raise ValueError('dataset_source must use owner/dataset-slug format')
    return resolved


class NotebookStaging:

    def __init__(self, staging_root: str | None = None, inventory: NotebookInventory | None = None):

        self.staging_root = Path(staging_root or '../storage/staging')

        self.inventory = inventory or NotebookInventory()



    def stage(self, notebook_id: str, job_id: UUID | str, kaggle_username: str, dataset_source: str | None = None) -> tuple[Path, str, int | None]:

        dataset_source = resolve_dataset_source(dataset_source)

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

        metadata['dataset_sources'] = [dataset_source]

        if metadata.get('title'):

            metadata['title'] = f"{metadata['title']} - {str(job_id)[:8]}"

        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

        return staging, kaggle_ref, manifest.get('default_timeout_seconds')
