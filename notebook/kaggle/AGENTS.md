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

<!-- MANUAL: -->
