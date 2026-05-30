<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# notebook

## Purpose
Jupyter notebooks implementing all sentiment analysis experiments on UIT-VSFC dataset. Each major experiment has a main notebook + topic analysis variant.

## Key Files
| File | Description |
|------|-------------|
| `PhoBERT_Baseline.ipynb` | vinai/phobert-base fine-tune baseline |
| `PhoBERT_Sentiwordnet_Refactored_LightFusion_TopicAnalysis.ipynb` | SWN hybrid + topic analysis |
| `PhoBERT_TFIDF_Refactored_LightFusion.ipynb` | PhoBERT + TF-IDF (5000 features) weighted fusion |
| `PhoBERT_TF-IDF_Sentiwordnet_Baseline_Positional.ipynb` | Full 3-way hybrid (PhoBERT + TF-IDF + SWN) |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `kaggle/` | Kaggle-optimized notebook variants for cloud GPU (see `kaggle/AGENTS.md`) |
| `out date/` | Archived/outdated notebook versions (see `out date/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Configured for Google Colab — update `BASE_DIR` in Config class for local run
- Use `tools/tune_hybrid_weights.py` for CLI-based weight tuning instead of notebook

## Dependencies

### Internal
- `src/` — Core Python modules
- `results/` — Model checkpoints and experiment outputs

<!-- MANUAL: -->
