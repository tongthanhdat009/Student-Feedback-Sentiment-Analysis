<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-30 | Updated: 2026-05-30 -->

# services

## Purpose
Business logic layer for the Kaggle Notebook Manager. Orchestrates Kaggle API interactions, account management, notebook operations, and job execution.

## Key Files
| File | Description |
|------|-------------|
| `account_service.py` | AccountService — list, create (with Fernet encryption), delete accounts |
| `kaggle_client_factory.py` | KaggleClientFactory — thread-safe KaggleApi instantiation with env var swap |
| `notebook_inventory.py` | NotebookInventory — scans notebook/kaggle/ for valid notebook directories |
| `notebook_staging.py` | NotebookStaging — resolves notebook path for staging before Kaggle push |
| `notebook_service.py` | NotebookService — orchestrates inventory listing + trigger flow |
| `s3_service.py` | S3Service — upload, presigned GET URL, list objects in S3 bucket |
| `job_worker.py` | JobWorker — async background job execution (trigger + download), startup recovery |

## For AI Agents
- Services receive AsyncSession via constructor
- AccountService uses EncryptionService for Kaggle key protection
- JobWorker uses asyncio.create_task for background execution
- KaggleClientFactory swaps env vars per-account for API auth

<!-- MANUAL: -->
