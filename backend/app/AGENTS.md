<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# app

## Purpose
Core application layer for Kaggle Notebook Manager. FastAPI app with layered architecture: controllers (routing) → services (business logic) → repositories (data access).

## Key Files
| File | Description |
|------|-------------|
| `main.py` | FastAPI application entrypoint, CORS config, router registration, startup hook for job recovery |
| `config.py` | pydantic-settings Settings class (env vars: DB URL, Kaggle paths, S3, encryption keys) |
| `database.py` | SQLAlchemy async engine, session factory, Base declarative class, get_session dependency |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `controllers/` | FastAPI route handlers (see `controllers/AGENTS.md`) |
| `models/` | SQLAlchemy ORM model definitions (see `models/AGENTS.md`) |
| `repositories/` | Data access layer with CRUD operations (see `repositories/AGENTS.md`) |
| `schemas/` | Pydantic request/response schemas (see `schemas/AGENTS.md`) |
| `services/` | Business logic layer (see `services/AGENTS.md`) |
| `utils/` | Auth, encryption, path security utilities (see `utils/AGENTS.md`) |
| `seeds/` | Development database seeding scripts (see `seeds/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Layered architecture: Controller → Service → Repository (no skipping layers)
- Controllers inject dependencies via FastAPI `Depends`
- Services receive AsyncSession, repositories receive AsyncSession
- Auth via X-API-Key header checked in `utils.auth.require_api_key`

### Testing Requirements
- Tests in `backend/tests/`
- Mock AsyncSession for service/repository tests

### Common Patterns
- All controllers depend on `require_api_key` router-level dependency
- EncryptionService wraps Fernet for Kaggle API key storage
- JobWorker uses asyncio task-based background execution

## Dependencies

### Internal
- `backend/alembic/` — Migrations for database schema
- `notebook/kaggle/` — Notebook files for inventory

### External
- FastAPI, SQLAlchemy, asyncpg, boto3, cryptography, kaggle-api

<!-- MANUAL: -->
