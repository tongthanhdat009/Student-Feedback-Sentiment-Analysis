<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# results

## Purpose
Container for all experiment outputs organized by model variant. Each model dir contains baseline or improvements with timestamped runs.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `PhoBERT/` | Baseline fine-tuned phobert-base (see `PhoBERT/AGENTS.md`) |
| `PhoBERT_Baseline/` | Alternative baseline run (see `PhoBERT_Baseline/AGENTS.md`) |
| `PhoBERT_Sentiwordnet_PosEnc_SiLU_Residual/` | SWN + positional encoding + SiLU + residual (see `AGENTS.md`) |
| `PhoBERT_Sentiwordnet_Refactored_LightFusion/` | SWN lightweight fusion, 6 improvement runs (see `AGENTS.md`) |
| `PhoBERT_TFIDF_E2E_PosEnc_SiLU_Residual/` | TF-IDF end-to-end + pos encoding (see `AGENTS.md`) |
| `PhoBERT_TFIDF_Refactored_LightFusion/` | TF-IDF lightweight fusion (see `AGENTS.md`) |
| `PhoBERT_TFIDF_Refactored_LightFusion_Optimized/` | Optimized TF-IDF fusion (see `AGENTS.md`) |
| `PhoBERT_TF-IDF_Sentiwordnet/` | Full 3-way hybrid (see `AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Each experiment run: artifacts/, models/, summaries/ (training_results.txt + experiment_summary.json), visualizations/
- Timestamp format: YYYYMMDD_HHMMSS
- Results format defined in `CLAUDE.md` Training Results Format section

<!-- MANUAL: -->
