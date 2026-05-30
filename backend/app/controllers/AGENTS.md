<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# controllers

## Purpose
FastAPI route handlers exposing REST API endpoints for the Kaggle Notebook Manager. Each controller maps to a domain area with APIRouter and prefix.

## Key Files
| File | Description |
|------|-------------|
| `health_controller.py` | GET /api/kaggle/health — service health check |
| `account_controller.py` | CRUD for Kaggle accounts under /api/kaggle/accounts |
| `notebook_controller.py` | Notebook inventory listing and trigger under /api/kaggle/notebooks |
| `job_controller.py` | Job listing, status, output download, artifact URL under /api/kaggle/jobs |
| `s3_controller.py` | S3 artifact management under /api/kaggle/s3 (minimal) |
| `__init__.py` | Package marker (empty) |

## For AI Agents
- All controllers use `require_api_key` dependency at router level
- Session injection via `Depends(get_session)`
- Service layer instantiated inline in handler functions
- Responses use Pydantic schemas for serialization

<!-- MANUAL: -->
