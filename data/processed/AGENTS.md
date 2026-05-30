<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# processed

## Purpose
Preprocessed UIT-VSFC dataset splits after teencode normalization, lowercasing, and special character removal.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `train/` | 11,426 training samples |
| `validation/` | 1,583 validation samples |
| `test/` | 3,166 test samples |

## For AI Agents

- Each split dir: sents.txt (text), sentiments.txt (0/1/2 labels), topics.txt (topic IDs)
- Generated from raw/ via `data_utils.preprocess_and_save_all()`

<!-- MANUAL: -->
