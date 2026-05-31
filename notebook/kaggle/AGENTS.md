<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# kaggle

## Purpose
Kaggle-optimized notebook variants for training on cloud GPU (Kaggle T4/P100). Each subdirectory contains notebook.ipynb + kernel-metadata.json for Kaggle API push.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `phobert-baseline/` | Kaggle notebook for PhoBERT baseline on T4 GPU |

## For AI Agents
- Uses paths like `/kaggle/input/uit-vsfc` instead of local paths
- Consumed by backend `NotebookInventory` service for Kaggle push workflow
- Outdated variants moved to `notebook/out date/kaggle/`

## Local Notebook Template

To add a new Kaggle-managed notebook on local disk, create this folder shape:

```text
notebook/kaggle/<slug>/
├── notebook.yaml
├── kernel-metadata.json
└── notebook.ipynb
```

Rules:
- `<slug>` must be safe: no slash, no `..`, not absolute.
- `notebook.yaml.slug` must equal folder name.
- `notebook.yaml.entry_file` must equal `kernel-metadata.json.code_file`.
- `kernel-metadata.json.kernel_type` must be `notebook`.
- Keep `kernel-metadata.json.id` as `<kaggle_username>/<slug>` in source. Worker rewrites staged copy to unique run ref.
- Do not put Kaggle API tokens/secrets in any notebook file.

Example `notebook.yaml`:

```yaml
slug: my-new-notebook
title: My New Notebook
description: Training notebook for Kaggle runner
entry_file: notebook.ipynb
default_accelerator: NvidiaTeslaT4
default_timeout_seconds: 3600
tags:
  - nlp
  - sentiment
artifacts:
  - "*.csv"
  - "*.json"
  - "*.pt"
params:
  epochs:
    type: int
    default: 5
  batch_size:
    type: int
    default: 16
```

Example `kernel-metadata.json`:

```json
{
  "id": "<kaggle_username>/my-new-notebook",
  "title": "My New Notebook",
  "code_file": "notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["owner/dataset-slug"]
}
```

Recommended notebook output for parser/S3 metadata:

```python
import json
from pathlib import Path

results = {
    "metrics": {"accuracy": 0.0, "f1": 0.0, "loss": 0.0},
    "params": {"epochs": 5, "batch_size": 16},
}
Path("training_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
```

After creating files, click Refresh on UI Notebook Inventory. Invalid notebooks show validation errors instead of being pushed.

Kaggle `403 GetKernelSessionStatus` means account/API token can push or list differently than status polling permits. Check exact Kaggle user/token, account permissions, and Kaggle UI for the unique ref.

<!-- MANUAL: -->
