<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# seeds

## Purpose
Development database seeding scripts for populating test data.

## Key Files
| File | Description |
|------|-------------|
| `seed_dev.py` | Creates dev Kaggle account from env vars (DEV_KAGGLE_ACCOUNT_NAME/USERNAME/KEY) |
| `seed_notebooks.py` | Seeds notebook metadata into database |
| `__init__.py` | Package marker (empty) |

## For AI Agents
- Run: `python -m app.seeds.seed_dev` from `backend/`
- Requires env vars set for dev account seeding

<!-- MANUAL: -->
