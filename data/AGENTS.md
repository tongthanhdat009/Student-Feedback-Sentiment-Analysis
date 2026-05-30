<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# data

## Purpose
UIT-VSFC dataset (Vietnamese Student Feedback) and VietSentiWordNet sentiment lexicon. ~16K samples across 3 classes: Negative (0), Neutral (1), Positive (2).

## Key Files
| File | Description |
|------|-------------|
| `sentiwordnet-dataset/VietSentiWordnet_Ver1.3.5.txt` | Vietnamese sentiment lexicon with pos/neg scores per word |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `processed/` | Preprocessed train/validation/test splits (see `processed/AGENTS.md`) |
| `raw/` | Original data before preprocessing (see `raw/AGENTS.md`) |
| `sentiwordnet-dataset/` | VietSentiWordNet lexicon (see `sentiwordnet-dataset/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Processed splits loaded via `src.data_utils.load_data()`
- Raw splits loaded via `src.data_utils.load_raw_data()`
- VietSentiWordNet loaded via `src.data_utils.load_sentiwordnet()`
- All splits: sents.txt, sentiments.txt, topics.txt (one per line)

## Dependencies

### Internal
- Consumed by `src/`, `notebook/`, `tools/`

<!-- MANUAL: -->
