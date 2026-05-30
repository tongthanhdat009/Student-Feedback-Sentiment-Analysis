<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# schemas

## Purpose
Pydantic models for API request/response serialization and validation.

## Key Files
| File | Description |
|------|-------------|
| `account.py` | AccountCreate (create request), AccountRead (response, from_attributes=True) |
| `job.py` | NotebookTriggerRequest (trigger payload), JobRead (response, from_attributes=True) |
| `__init__.py` | Package marker (empty) |

## For AI Agents
- Schemas use `from_attributes = True` for ORM-to-Pydantic conversion
- AccountCreate validates max_length=100 for name field

<!-- MANUAL: -->
