<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# utils

## Purpose
Cross-cutting utilities for authentication, encryption, and path security.

## Key Files
| File | Description |
|------|-------------|
| `auth.py` | require_api_key FastAPI dependency — validates X-API-Key header against settings |
| `encryption.py` | EncryptionService — Fernet symmetric encryption for Kaggle API keys |
| `path_guard.py` | safe_child (path traversal prevention), normalize_s3_key (S3 key validation) |

## For AI Agents
- No __init__.py — utilities imported directly by name
- EncryptionService raises ValueError if key is the placeholder 'generate-fernet-key'

<!-- MANUAL: -->
