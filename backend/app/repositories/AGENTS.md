<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# repositories

## Purpose
Data access layer implementing repository pattern over SQLAlchemy async sessions. Each repository handles CRUD for one model.

## Key Files
| File | Description |
|------|-------------|
| `account_repository.py` | AccountRepository — list, get_by_name, add, delete for KaggleAccount |
| `job_repository.py` | JobRepository — list (desc created_at), get, add, save, stale_running for KaggleJob |

## For AI Agents
- Repositories receive AsyncSession via constructor
- No business logic — pure data access
- Services coordinate across repositories

<!-- MANUAL: -->
