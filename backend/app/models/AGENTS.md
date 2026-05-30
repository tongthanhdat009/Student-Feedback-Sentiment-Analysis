<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# models

## Purpose
SQLAlchemy ORM model definitions for the Kaggle Notebook Manager database schema. Defines table structure, relationships, and constraints.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Re-exports KaggleAccount, KaggleJob for Alembic auto-detection |
| `account.py` | KaggleAccount model — Kaggle credentials (encrypted), active status, usage tracking |
| `job.py` | KaggleJob model — job type, status, target notebook, S3 storage metadata |
| `api_response.py` | Generic ApiResponse Pydantic model (not ORM — shared utility) |

## For AI Agents
- Models inherit from `database.Base`
- KaggleAccount has one-to-many relationship to KaggleJob
- Foreign key constraints enforced at DB level
- Import all models in `__init__.py` for Alembic to detect schema changes

<!-- MANUAL: -->
