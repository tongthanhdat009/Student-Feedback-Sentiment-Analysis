<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# raw

## Purpose
Original UIT-VSFC dataset before preprocessing. Same structure as processed/ but with raw Vietnamese text (contains teencode, special chars, mixed case).

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `train/` | 11,426 raw training samples |
| `validation/` | 1,583 raw validation samples |
| `test/` | 3,166 raw test samples |

## For AI Agents

- Each split dir: sents.txt, sentiments.txt, topics.txt
- Process via `data_utils.preprocess_and_save_all()` to regenerate processed/

<!-- MANUAL: -->
