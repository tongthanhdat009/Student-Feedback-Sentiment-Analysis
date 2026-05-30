<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# tools

## Purpose
CLI tools for model tuning and analysis.

## Key Files
| File | Description |
|------|-------------|
| `tune_hybrid_weights.py` | CLI tool for simplex grid search over PhoBERT/TF-IDF/SentiWordNet feature weights. Uses PhoBERTClassifier, LogisticRegression, cache embeddings, writes summaries + models |

## For AI Agents

### Working In This Directory
- Run from project root: `python tools/tune_hybrid_weights.py --mode phobert_swn`
- Requires trained PhoBERT model at `results/PhoBERT/baseline/models/phobert_model.pt`
- Modes: phobert_swn (2-block), phobert_tfidf_swn (3-block)

## Dependencies

### Internal
- `src/` — data_utils, hybrid_fusion modules

### External
- torch, transformers, sklearn, joblib, pandas

<!-- MANUAL: -->
