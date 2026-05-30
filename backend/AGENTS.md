<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# backend

## Purpose
FastAPI backend for Kaggle Notebook Manager — manages Kaggle accounts, orchestrates notebook execution via Kaggle API, handles job scheduling, and stores artifacts in S3-compatible storage. PostgreSQL for persistence, async SQLAlchemy for data access.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `app/` | Application core: API controllers, domain models, business services, data access layer (see `app/AGENTS.md`) |
| `alembic/` | Database migration configuration and versioned schemas (see `alembic/AGENTS.md`) |
| `tests/` | Backend test suite (see `tests/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Python 3.11+ with FastAPI + asyncpg + SQLAlchemy async
- Config via pydantic-settings, env file at project root
- Run: `uvicorn app.main:app --reload` from `backend/`

### Testing Requirements
- `pytest` from `backend/` directory
- Requires PostgreSQL accessible at DATABASE_URL

### Common Patterns
- Controllers: FastAPI APIRouter with prefix, Depends for DI
- Services: stateless classes with AsyncSession injected via constructor
- Repositories: one class per model, CRUD operations, raw SQLAlchemy queries
- Auth: X-API-Key header, dependency via `require_api_key()`

## Dependencies

### Internal
- `notebook/kaggle/` — Local Kaggle notebook source files

### External
- FastAPI — Web framework
- SQLAlchemy (async) — ORM
- asyncpg — PostgreSQL driver
- kagglehub / kaggle-api — Kaggle API client
- boto3 — S3-compatible storage
- cryptography — Fernet encryption for Kaggle keys

<!-- MANUAL: -->
