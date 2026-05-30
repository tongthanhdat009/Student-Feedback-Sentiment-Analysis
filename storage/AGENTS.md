<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# storage

## Purpose
Local storage directory for Kaggle notebook output artifacts. Mirrors S3 bucket structure for local development without cloud storage.

## Key Files
| File | Description |
|------|-------------|
| `.gitignore` | Ensures output files are not committed |
| `kaggle_outputs/.gitkeep` | Placeholder to track empty directory |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `kaggle_outputs/` | Extracted notebook output zip contents from Kaggle API |

## For AI Agents
- Backend config `kaggle_output_dir` points here for local dev
- S3 service falls back to this directory when S3 not configured

<!-- MANUAL: -->
