<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# sentiwordnet-dataset

## Purpose
VietSentiWordNet sentiment lexicon for Vietnamese. Used to extract lexicon-based sentiment features (8 basic or 35 extended).

## Key Files
| File | Description |
|------|-------------|
| `VietSentiWordnet_Ver1.3.5.txt` | Tab-separated: synset_id, type, pos_score, neg_score, terms. ~1000s of entries with Vietnamese words mapped to positive/negative scores. |

## For AI Agents

- Load via `data_utils.load_sentiwordnet()`
- Returns dict: word -> {pos_score: float, neg_score: float}
- Feature extraction: `data_utils.get_swn_features()` (8-dim) or `get_swn_features_extended()` (35-dim)

<!-- MANUAL: -->
