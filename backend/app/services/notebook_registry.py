import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class NotebookValidation:
    slug: str
    path: str
    title: str | None
    valid: bool
    errors: list[str]
    manifest: dict[str, Any]
    metadata: dict[str, Any]


def is_safe_slug(slug: str) -> bool:
    return bool(slug) and '/' not in slug and '\\' not in slug and '..' not in slug and not os.path.isabs(slug)


class NotebookRegistry:
    def validate_folder(self, folder: Path) -> NotebookValidation:
        slug = folder.name
        errors: list[str] = []
        manifest: dict[str, Any] = {}
        metadata: dict[str, Any] = {}

        if not is_safe_slug(slug):
            errors.append('Folder slug is unsafe')

        nb = folder / 'notebook.ipynb'
        meta_path = folder / 'kernel-metadata.json'
        manifest_path = folder / 'notebook.yaml'
        for path, label in [(nb, 'notebook.ipynb'), (meta_path, 'kernel-metadata.json'), (manifest_path, 'notebook.yaml')]:
            if not path.exists():
                errors.append(f'{label} is required')

        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding='utf-8'))
                if not isinstance(metadata, dict):
                    errors.append('kernel-metadata.json must contain an object')
                    metadata = {}
            except Exception as exc:
                errors.append(f'kernel-metadata.json is invalid JSON: {exc}')

        if manifest_path.exists():
            try:
                loaded = yaml.safe_load(manifest_path.read_text(encoding='utf-8'))
                if not isinstance(loaded, dict):
                    errors.append('notebook.yaml must contain an object')
                else:
                    manifest = loaded
            except Exception as exc:
                errors.append(f'notebook.yaml is invalid YAML: {exc}')

        entry_file = manifest.get('entry_file')
        if manifest:
            if manifest.get('slug') != slug:
                errors.append('notebook.yaml.slug must match folder name')
            if not entry_file:
                errors.append('notebook.yaml.entry_file is required')
            elif not is_safe_slug(str(entry_file)):
                errors.append('notebook.yaml.entry_file is unsafe')
            elif not (folder / str(entry_file)).exists():
                errors.append('manifest entry_file does not exist')

        if metadata:
            if entry_file and metadata.get('code_file') != entry_file:
                errors.append('kernel-metadata.json.code_file must match notebook.yaml.entry_file')
            if metadata.get('kernel_type') != 'notebook':
                errors.append('kernel-metadata.json.kernel_type must be notebook')

        return NotebookValidation(
            slug=slug,
            path=str(folder),
            title=manifest.get('title') or metadata.get('title'),
            valid=not errors,
            errors=errors,
            manifest=manifest,
            metadata=metadata,
        )

    def load_manifest(self, folder: Path) -> dict[str, Any]:
        validation = self.validate_folder(folder)
        if not validation.valid:
            raise ValueError('; '.join(validation.errors))
        return validation.manifest
