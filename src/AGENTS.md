<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# src

## Purpose
Core Python package for Vietnamese Student Feedback Sentiment Analysis. Provides data loading, preprocessing, feature extraction, hybrid fusion, and ensemble utilities.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Package exports: data utils, hybrid fusion, ensemble |
| `data_utils.py` | Centralized data loading (UIT-VSFC), teencode normalization (100+ mappings), Vietnamese preprocessing, SentiWordNet lexicon loader, 8 basic + 35 extended feature extractors |
| `hybrid_fusion.py` | Weighted feature fusion: simplex grid search, LearnableModalityGate (PyTorch), blend_probabilities, evaluate_predictions |
| `swn_neutral_ensemble.py` | Ensemble evaluation: neutral_aware_ensemble() fusion strategy, pair analysis, metric reporting |

## For AI Agents

### Working In This Directory
- All imports use relative paths (from .data_utils import ...)
- data_utils.py is the primary entry point for data loading
- hybrid_fusion.py provides reusable search/evaluate infrastructure
- swn_neutral_ensemble.py used for ensemble-based evaluation

## Dependencies

### Internal
- `data/` — UIT-VSFC dataset and VietSentiWordNet lexicon

### External
- numpy, torch, sklearn, transformers, pandas

<!-- MANUAL: -->
