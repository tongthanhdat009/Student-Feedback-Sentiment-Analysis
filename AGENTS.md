<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# Student-Feedback-Sentiment-Analysis

## Purpose
Vietnamese Student Feedback Sentiment Analysis using UIT-VSFC dataset. Implements 4 approaches for 3-class sentiment classification (Negative=0/Neutral=1/Positive=2): PhoBERT baseline, TF-IDF hybrid, SentiWordNet hybrid, full hybrid. Feature fusion via weighted search + LogisticRegression/XGBoost.

## Key Files
| File | Description |
|------|-------------|
| `CLAUDE.md` | Project guide: architecture, workflow, data structure, config |
| `README.md` | Project overview and setup instructions |
| `requirements.txt` | Python dependencies (torch, transformers, sklearn, etc.) |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `src/` | Python source: data utils, hybrid fusion, ensemble logic (see `src/AGENTS.md`) |
| `tools/` | Weight tuning CLI tool (see `tools/AGENTS.md`) |
| `data/` | UIT-VSFC + VietSentiWordNet dataset splits (see `data/AGENTS.md`) |
| `notebook/` | Jupyter notebooks for all experiments (see `notebook/AGENTS.md`) |
| `docs/` | Analysis reports and optimization docs (see `docs/AGENTS.md`) |
| `results/` | Experiment outputs per model variant (see `results/AGENTS.md`) |
| `plan/` | Planning artifacts (see `plan/AGENTS.md`) |
| `backend/` | FastAPI backend — Kaggle Notebook Manager (see `backend/AGENTS.md`) |
| `frontend/` | React + Vite SPA for Kaggle dashboard (see `frontend/AGENTS.md`) |
| `storage/` | Local Kaggle output artifacts storage (see `storage/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Python project, run `pip install -r requirements.txt` first
- Notebooks configured for Google Colab—update `BASE_DIR` in Config for local run
- Use `src/data_utils.py` for centralized data loading/preprocessing
- Results go in `results/{ModelType}/{baseline|improvements}/{YYYYMMDD}/` with required `summaries/training_results.txt`

### Testing Requirements
- Run notebooks end-to-end to verify changes
- Validate result format matches `training_results.txt` spec in `CLAUDE.md`

### Common Patterns
- All experiments: 3-class sentiment, train 11426 / val 1583 / test 3166
- Hybrid models: extract embeddings, normalize, concatenate, train classifier
- SentiWordNet: 8 basic or 35 extended features
- Weighted fusion: simplex grid search over feature block weights

## Dependencies

### Internal
- `src/` — Core modules used by notebooks and tools
- `notebook/` — Experiment notebooks consuming `src/`

### External
- PyTorch 2.x — Deep learning
- transformers — PhoBERT model
- scikit-learn — TF-IDF, LogisticRegression, metrics
- XGBoost — Optional classifier
- pandas, numpy, matplotlib — Data handling and visualization

<!-- MANUAL: -->
