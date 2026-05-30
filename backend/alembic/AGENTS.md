<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# alembic

## Purpose
Alembic database migration configuration for the Kaggle Notebook Manager PostgreSQL schema.

## Key Files
| File | Description |
|------|-------------|
| `env.py` | Alembic env config — sets SQLAlchemy URL from Settings, imports all models for auto-detection |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `versions/` | Migration revision scripts (see `versions/AGENTS.md`) |

## For AI Agents
- Run migrations: `alembic upgrade head` from `backend/`
- New revision: `alembic revision --autogenerate -m "description"`
- Configured for async PostgreSQL via async engine

<!-- MANUAL: -->
