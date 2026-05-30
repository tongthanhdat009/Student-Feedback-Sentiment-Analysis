<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# api

## Purpose
HTTP API client layer for communicating with the FastAPI backend.

## Key Files
| File | Description |
|------|-------------|
| `client.ts` | Generic fetch wrapper — sets base URL, JSON headers, X-API-Key auth |
| `kaggleApi.ts` | Typed API methods — health, CRUD accounts, inventory, trigger, jobs, download, artifact URL |

## For AI Agents
- BASE_URL from VITE_API_BASE env var (default http://127.0.0.1:8000)
- API_KEY from VITE_ADMIN_API_KEY env var
- All methods return typed promises via generics

<!-- MANUAL: -->
