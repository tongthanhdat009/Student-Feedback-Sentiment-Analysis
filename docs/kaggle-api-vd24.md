# Plan: Kaggle Notebook Manager MVP

## Context verified

- Repo hiện là Python ML project cho Vietnamese Student Feedback Sentiment Analysis.
- `backend/` chưa tồn tại.
- `frontend/` chưa tồn tại.
- `docs/KAGGLE_API_IMPLEMENTATION_GUIDE.md` có spec đầy đủ cho FastAPI + React/Tailwind + Kaggle SDK.
- `notebook/out date/kaggle/` có notebook Kaggle cũ nhưng **không dùng**.
- `notebook/kaggle/` hiện cần được chuẩn hóa cho MVP.

## User decisions

- MVP scope: **Notebook-only**.
- Auth MVP: **API key header**.
- Job runner: **in-process worker**.
- DB: **PostgreSQL**.
- Notebook source: **tạo mới `notebook/kaggle/`**, không tạo mới từ notebooks trong thư mục `out date`.
- Dataset source placeholder/config: **`owner/dataset-slug`**.
- Result handling: **download raw Kaggle output**, không parse metrics vào dashboard ở MVP.
- Model artifact handling: **download model từ Kaggle output rồi upload/lưu vào S3-compatible storage**.
- S3 env format follows `docs/s3-api-patterns.md` naming style: `S3Storage__*`; user will fill values later.

## Target architecture

```text
React + Tailwind
  → FastAPI REST API
    → PostgreSQL
    → Kaggle Python SDK
    → notebook/kaggle/<notebook_id>/
    → storage/kaggle_outputs/<job_id>/
    → S3-compatible storage for model/raw output artifacts
```

Frontend không gọi Kaggle trực tiếp. Backend giữ Kaggle username/key, mã hóa key trong DB, trigger notebook, poll status, download raw output.

## MVP boundaries

### Include

- Account CRUD + test Kaggle auth.
- Encrypted Kaggle key storage.
- API-key protected backend endpoints.
- Notebook inventory from `notebook/kaggle/`.
- Trigger Kaggle notebook/kernel.
- DB-backed job status/history.
- In-process worker for long-running Kaggle calls.
- Raw output download into `storage/kaggle_outputs/{job_id}/`.
- Upload downloaded Kaggle model/raw output artifacts to S3-compatible storage.
- Generate presigned GET URL for S3 artifact download.
- React dashboard/accounts/notebooks UI.

### Exclude for MVP

- Dataset browser/download/create/version APIs.
- Competition APIs/submission.
- JWT login/users/RBAC.
- Celery/RQ/Redis.
- Parsing `training_results.txt` into metrics.
- Manual multipart upload UI for S3.
- Any notebook under `notebook/out date/kaggle/`.

## Backend implementation plan

### 1. Scaffold backend

Create:

```text
backend/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── dependencies.py
│   ├── controllers/
│   │   ├── account_controller.py
│   │   ├── notebook_controller.py
│   │   ├── job_controller.py
│   │   └── health_controller.py
│   ├── services/
│   │   ├── account_service.py
│   │   ├── kaggle_client_factory.py
│   │   ├── notebook_inventory.py
│   │   ├── notebook_staging.py
│   │   ├── notebook_service.py
│   │   └── job_worker.py
│   ├── repositories/
│   │   ├── account_repository.py
│   │   └── job_repository.py
│   ├── models/
│   │   ├── account.py
│   │   ├── job.py
│   │   └── api_response.py
│   └── utils/
│       ├── encryption.py
│       ├── path_guard.py
│       └── auth.py
├── alembic/
├── requirements.txt
└── .env.example
```

Use FastAPI, SQLAlchemy async ORM, asyncpg, Alembic migrations, seed scripts, Kaggle SDK, cryptography, boto3/botocore for S3-compatible object storage.

### 2. Config/env

`.env.example`:

```env
APP_NAME=Kaggle Notebook Manager
ENV=development
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/kaggle_manager
ADMIN_API_KEY=change-me
FERNET_KEY=generate-fernet-key
KAGGLE_NOTEBOOK_DIR=../notebook/kaggle
KAGGLE_OUTPUT_DIR=../storage/kaggle_outputs
KAGGLE_DEFAULT_DATASET_SOURCE=owner/dataset-slug

# S3-compatible storage for Kaggle model/output artifacts
S3Storage__BucketName=
S3Storage__Region=
S3Storage__AccessKey=
S3Storage__SecretKey=
S3Storage__SessionToken=
S3Storage__ServiceUrl=
S3Storage__UsePathStyle=
S3Storage__DefaultPresignedUrlExpirationInSeconds=3600
S3Storage__MaxPresignedUrlExpirationInSeconds=604800
```

### 3. PostgreSQL schema + migration/seed

Use SQLAlchemy async ORM models as source of truth, Alembic for schema migrations, and explicit seed scripts for dev/default config.

Tables:

```sql
kaggle_accounts (
  id UUID PRIMARY KEY,
  name VARCHAR(100) UNIQUE NOT NULL,
  kaggle_username VARCHAR(255) NOT NULL,
  kaggle_key_encrypted TEXT NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  last_used_at TIMESTAMPTZ NULL
)
```

```sql
kaggle_jobs (
  id UUID PRIMARY KEY,
  account_id UUID REFERENCES kaggle_accounts(id),
  job_type VARCHAR(50) NOT NULL,
  target_ref VARCHAR(255) NOT NULL,
  status VARCHAR(50) NOT NULL DEFAULT 'pending',
  message TEXT NULL,
  output_path TEXT NULL,
  s3_object_key TEXT NULL,
  s3_presigned_url TEXT NULL,
  s3_presigned_url_expires_at TIMESTAMPTZ NULL,
  created_at TIMESTAMPTZ NOT NULL,
  started_at TIMESTAMPTZ NULL,
  finished_at TIMESTAMPTZ NULL
)
```

Job statuses:

```text
pending → running → completed | failed
```

Job types:

```text
notebook_trigger
notebook_output_download
s3_upload
```

Migration requirements:

```text
backend/alembic/env.py                 # async SQLAlchemy metadata wiring
backend/alembic/versions/*.py          # generated migrations
backend/app/models/account.py          # SQLAlchemy ORM model
backend/app/models/job.py              # SQLAlchemy ORM model
backend/app/schemas/*.py               # Pydantic request/response schemas
```

Seed requirements:

```text
backend/app/seeds/seed_dev.py          # dev seed entrypoint
backend/app/seeds/seed_notebooks.py    # optional notebook inventory seed/validation helper
```

Seed behavior:

- Create no Kaggle account by default unless env explicitly provides safe dev values.
- Seed/validate notebook inventory metadata from `notebook/kaggle/`.
- Never seed real Kaggle keys into source files.
- Allow command:

```text
python -m app.seeds.seed_dev
```

Validation commands:

```text
alembic upgrade head
python -m app.seeds.seed_dev
python -m pytest backend/tests
```

### 3.1 S3 schema fields

`kaggle_jobs` stores S3 artifact metadata:

```text
s3_object_key: canonical object key, e.g. kaggle-outputs/main/phobert-baseline/<job_id>/model.zip
s3_presigned_url: optional latest generated GET URL for UI download
s3_presigned_url_expires_at: expiry timestamp; regenerate when expired
```

If one job uploads multiple files, store the primary model archive in `s3_object_key` for MVP. Future v2 can add `kaggle_job_artifacts` table for many files.

### 4. API key auth

Protect all `/api/kaggle/*` endpoints using:

```http
X-API-Key: <ADMIN_API_KEY>
```

Reject missing/wrong key with `401`.

### 5. Encryption

Use Fernet for `kaggle_key`.

Rules:

- Store only encrypted key.
- Never return key to frontend.
- Decrypt only inside service calling Kaggle SDK.

### 6. Kaggle client factory

Implement thread-safe `KaggleClientFactory` using temporary `KAGGLE_USERNAME`/`KAGGLE_KEY` env mutation under lock, as described in `docs/KAGGLE_API_IMPLEMENTATION_GUIDE.md`.

Wrap blocking SDK calls with `run_in_threadpool` or in-process worker thread.

### 6.1 S3 object storage service

Adapt `docs/s3-api-patterns.md` to Python using `boto3`/`botocore` instead of .NET `AWSSDK.S3`.

Add files:

```text
backend/app/services/s3_service.py
backend/app/controllers/s3_controller.py        # optional/admin artifact endpoints
```

Add dependencies:

```txt
boto3>=1.34.0
botocore>=1.34.0
```

Config mapping keeps user-provided env names:

```text
S3Storage__BucketName                         → bucket name
S3Storage__Region                             → boto3 region_name
S3Storage__AccessKey                          → aws_access_key_id
S3Storage__SecretKey                          → aws_secret_access_key
S3Storage__SessionToken                       → aws_session_token optional
S3Storage__ServiceUrl                         → endpoint_url for MinIO/Ceph/S3-compatible
S3Storage__UsePathStyle                       → Config(s3={"addressing_style":"path"}) when true
S3Storage__DefaultPresignedUrlExpirationInSeconds → default ExpiresIn
S3Storage__MaxPresignedUrlExpirationInSeconds     → clamp max ExpiresIn, AWS max 604800
```

`S3Service` responsibilities:

- Build singleton `boto3.client("s3")`.
- Upload local Kaggle output/model files using streaming `upload_file`/`upload_fileobj`.
- Generate presigned GET URL using `generate_presigned_url("get_object")`.
- List objects by prefix for job/artifact browsing.
- Delete object only in admin/future cleanup flow.
- Normalize object keys; reject `../`, absolute paths, backslash traversal.
- Wrap `ClientError` into app-level errors.

boto3 config equivalent to guide:

```python
Config(
    region_name=settings.s3.region,
    signature_version="s3v4",
    s3={"addressing_style": "path" if settings.s3.use_path_style else "auto"},
    retries={"max_attempts": 3, "mode": "adaptive"},
)
```

S3 object key scheme:

```text
kaggle-outputs/{account_name}/{notebook_id}/{job_id}/{filename}
```

Examples:

```text
kaggle-outputs/main/phobert-baseline/a1b2c3d4/model.zip
kaggle-outputs/main/phobert-baseline/a1b2c3d4/output.zip
```

Security rules:

- Do not expose S3 secret/access keys to frontend.
- Bucket should remain private; frontend downloads via presigned GET URL only.
- Clamp presigned expiry to max env value.
- Use path/prefix guard so uploads cannot escape `kaggle-outputs/` namespace.
- Stream large files; do not read full model artifact into memory.

### 7. Notebook inventory

Only scan:

```text
notebook/kaggle/
```

Do not scan/use:

```text
notebook/out date/kaggle/
```

Expected kernel folder shape:

```text
notebook/kaggle/<notebook_id>/
├── notebook.ipynb
└── kernel-metadata.json
```

Inventory endpoint returns folders that contain both files.

Metadata template should include dataset placeholder:

```json
{
  "id": "<kaggle_username>/<notebook_id>",
  "title": "<notebook title>",
  "code_file": "notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["owner/dataset-slug"]
}
```

### 8. Notebook trigger flow

Endpoint:

```text
POST /api/kaggle/notebooks/trigger
```

Request:

```json
{
  "account": "main",
  "notebook_id": "phobert-baseline"
}
```

Flow:

```text
validate API key
→ validate account exists
→ validate notebook_id exists under notebook/kaggle
→ insert kaggle_jobs(status=pending, job_type=notebook_trigger)
→ enqueue in-process worker
→ return job_id immediately
```

Worker:

```text
load job/account
→ status=running
→ decrypt Kaggle key
→ create Kaggle client
→ stage/validate kernel folder
→ api.kernels_push(kernel_folder)
→ status=completed or failed
```

### 9. Output download + S3 upload flow

Endpoint option:

```text
POST /api/kaggle/jobs/{id}/download-output
```

Flow:

```text
validate job completed
→ determine owner/slug from job target_ref/message metadata
→ api.kernels_output(ref, path=storage/kaggle_outputs/{job_id})
→ update local output_path
→ locate model/output artifacts in downloaded folder
→ upload artifacts to S3 using streaming boto3 upload_file/upload_fileobj
→ object_key = kaggle-outputs/{account_name}/{notebook_id}/{job_id}/{filename}
→ store primary s3_object_key on kaggle_jobs
→ generate presigned GET URL
→ store s3_presigned_url + s3_presigned_url_expires_at
→ return raw local output path + S3 object key + presigned download URL
```

MVP downloads raw Kaggle output and uploads model/raw artifacts to S3. No metric parsing.

### 10. Job recovery behavior

Because runner is in-process:

- On app startup, find jobs `status in ('pending', 'running')`.
- Mark stale `running` jobs as `failed` with message `Interrupted by server restart`.
- Optionally re-enqueue `pending` jobs.

This makes failure explicit.

## Backend API map

```text
GET    /api/kaggle/health

GET    /api/kaggle/accounts
POST   /api/kaggle/accounts
DELETE /api/kaggle/accounts/{name}
POST   /api/kaggle/accounts/{name}/test

GET    /api/kaggle/notebooks/inventory
GET    /api/kaggle/notebooks?account=main
POST   /api/kaggle/notebooks/trigger
GET    /api/kaggle/notebooks/status/{owner}/{slug}?account=main
POST   /api/kaggle/jobs/{id}/download-output
GET    /api/kaggle/jobs/{id}/artifact-url

GET    /api/kaggle/jobs
GET    /api/kaggle/jobs/{id}
```

Artifact URL endpoint regenerates a presigned GET URL when the stored URL is missing/expired.

## Frontend implementation plan

### 1. Scaffold frontend

Create Vite React + TypeScript + Tailwind app under `frontend/`.

Structure:

```text
frontend/src/
├── api/
│   ├── client.ts
│   └── kaggleApi.ts
├── types/kaggle.ts
├── routes/
│   ├── Dashboard.tsx
│   ├── AccountsPage.tsx
│   └── NotebooksPage.tsx
├── components/kaggle/
│   ├── AccountCreateModal.tsx
│   ├── AccountTable.tsx
│   ├── NotebookTriggerModal.tsx
│   ├── JobTable.tsx
│   └── StatusBadge.tsx
├── App.tsx
└── main.tsx
```

### 2. API client

Use Axios with base URL and API key:

```env
VITE_API_URL=http://localhost:8000
VITE_ADMIN_API_KEY=change-me
```

Attach:

```http
X-API-Key: VITE_ADMIN_API_KEY
```

Note: MVP only. For production, do not expose admin API key in browser; replace with login/session.

### 3. Pages

#### Dashboard

Show:

- health status
- accounts count
- running jobs count
- failed jobs count
- recent jobs table

#### Accounts

Show:

- list accounts
- create account modal
- test account button
- delete account button
- no key display

#### Notebooks

Show:

- inventory from `notebook/kaggle/`
- account dropdown
- trigger button
- latest job status
- poll every 5–10 seconds while pending/running
- download raw output button when completed
- upload/download artifact status from S3
- presigned S3 artifact download button when `s3_object_key` exists

## Notebook preparation plan

Create new Kaggle kernel folders under:

```text
notebook/kaggle/
```

Do **not** copy from:

```text
notebook/out date/kaggle/
```

Use current non-outdated notebooks as source if desired, but convert manually/explicitly into kernel folders with:

```text
notebook.ipynb
kernel-metadata.json
```

Each notebook must read data from Kaggle dataset source:

```text
owner/dataset-slug
```

Before trigger works, replace placeholder with real Kaggle dataset slug.

## Validation plan

Backend:

```text
python -m pytest backend/tests
python -m uvicorn app.main:app --reload
```

Manual API checks:

```text
GET /api/kaggle/health
POST /api/kaggle/accounts
POST /api/kaggle/accounts/{name}/test
GET /api/kaggle/notebooks/inventory
POST /api/kaggle/notebooks/trigger
GET /api/kaggle/jobs/{id}
POST /api/kaggle/jobs/{id}/download-output
GET /api/kaggle/jobs/{id}/artifact-url
```

Frontend:

```text
npm run lint
npm run build
```

Manual UI checks:

- create/test account
- list notebook inventory
- trigger notebook
- poll job
- download raw output
- upload model/raw artifact to S3
- open presigned S3 artifact URL

## Risks + mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Browser exposes `VITE_ADMIN_API_KEY` | Not prod-safe | MVP only; later replace with JWT/session |
| In-process worker dies on restart | Running job may be lost | Startup marks stale running jobs failed |
| Kaggle SDK blocking | API timeout | Worker/threadpool |
| Bad notebook paths | Path traversal / wrong files | Only allow folders under `notebook/kaggle/` |
| Placeholder dataset slug invalid | Kaggle notebook fails | Replace `owner/dataset-slug` before real run |
| Raw output large | Disk growth | Store under job folder; add cleanup later |
| S3 creds misconfigured | Upload/download fails | Health check validates bucket access; clear error mapping |
| Private model exposed | Data/model leak | Private bucket; presigned GET only; clamp expiry |
| Large model upload OOM | Backend crash | Stream `upload_file`/`upload_fileobj`; never load whole file |

## Implementation phases + validation sub-agents

After each phase, spawn focused validation sub-agent(s) before moving on. Validators must be read-only unless explicitly in Exec mode and must report: changed files reviewed, command output, failures, risks, missing tests.

### Phase 1 — Backend scaffold + config

Tasks:

1. Create backend scaffold.
2. Add PostgreSQL config + SQLAlchemy async engine/session.
3. Add Alembic async migration wiring.
4. Add `.env.example` including Kaggle + S3 env.
5. Add API key auth dependency skeleton.

Validation sub-agent:

```text
Spawn explore/validator: inspect scaffold, config loading, env names, async DB setup, Alembic wiring. Run/read-only commands where allowed: import checks, alembic current/history if DB available. Report blockers.
```

### Phase 2 — DB models, migrations, seeds

Tasks:

1. Add SQLAlchemy ORM models for `kaggle_accounts`, `kaggle_jobs`.
2. Generate Alembic migration.
3. Add seed scripts: `python -m app.seeds.seed_dev`.
4. Seed/validate notebook inventory metadata only; never seed real Kaggle keys.
5. Add repository layer.

Validation sub-agent:

```text
Spawn explore/validator: verify ORM↔migration match, indexes/constraints, seed idempotency, no secrets in seed. Run `alembic upgrade head`, seed command, DB smoke tests if env available.
```

### Phase 3 — Accounts + encryption + Kaggle auth

Tasks:

1. Add Fernet encryption util.
2. Add account CRUD APIs.
3. Add account test-auth API.
4. Ensure response schemas never expose Kaggle key.
5. Add unit tests for encryption/account service.

Validation sub-agent:

```text
Spawn explore/validator: review secret handling, API-key protection, response DTOs, test coverage. Run backend tests for accounts/encryption.
```

### Phase 4 — S3 service

Tasks:

1. Add S3 config from `S3Storage__*` env.
2. Add boto3 `S3Service` singleton.
3. Add object key normalization/path guard.
4. Add presigned GET URL generation.
5. Add S3 health check/bucket access check.

Validation sub-agent:

```text
Spawn explore/validator: verify env mapping matches user-provided names and `docs/s3-api-patterns.md`, no S3 creds exposed to frontend, presigned expiry clamped, boto3 path-style config correct. Run mocked S3 tests if available.
```

### Phase 5 — Notebook inventory + Kaggle trigger worker

Tasks:

1. Add notebook inventory scanner for `notebook/kaggle/` only.
2. Exclude `notebook/out date/kaggle/` hard.
3. Add job repository/service + in-process worker.
4. Add notebook trigger endpoint.
5. Add startup recovery for stale jobs.

Validation sub-agent:

```text
Spawn explore/validator: verify path guard, outdate exclusion, worker state transitions, restart recovery, blocking Kaggle calls isolated. Run job service tests with Kaggle mock.
```

### Phase 6 — Output download + S3 artifact upload

Tasks:

1. Add notebook status/output download endpoints.
2. Download raw Kaggle output to `storage/kaggle_outputs/{job_id}/`.
3. Locate model/raw artifacts.
4. Upload model/raw artifacts to S3 under `kaggle-outputs/{account}/{notebook}/{job_id}/{filename}`.
5. Store `s3_object_key`, `s3_presigned_url`, expiry on job.
6. Add artifact URL regeneration endpoint.

Validation sub-agent:

```text
Spawn explore/validator: verify raw output preserved, uploads stream not memory-load, DB artifact fields updated, presigned URL regenerated after expiry, errors mapped cleanly. Run mocked Kaggle + mocked S3 integration tests.
```

### Phase 7 — Notebook/kaggle preparation

Tasks:

1. Create/standardize `notebook/kaggle/<notebook_id>/` kernel folders.
2. Add `notebook.ipynb` + `kernel-metadata.json`.
3. Use dataset source placeholder `owner/dataset-slug` until real slug supplied.
4. Do not copy/use `notebook/out date/kaggle/`.

Validation sub-agent:

```text
Spawn explore/validator: inspect kernel folders, metadata validity, dataset_sources placeholder, outdate exclusion, notebook path assumptions.
```

### Phase 8 — Frontend scaffold + UI

Tasks:

1. Scaffold React + Tailwind frontend.
2. Add API client + types.
3. Build Dashboard, Accounts, Notebooks pages.
4. Add polling + raw output download/S3 artifact URL UI.
5. Ensure S3 creds never appear in frontend env.

Validation sub-agent:

```text
Spawn explore/validator: review UI/API integration, API key handling limitation, no S3 secret leakage, loading/error states. Run `npm run lint` and `npm run build`.
```

### Phase 9 — Final full-system validation

Tasks:

1. Run backend tests.
2. Run migrations from empty DB.
3. Run seed scripts.
4. Run frontend lint/build.
5. Manual smoke test API flow.

Validation sub-agents:

```text
Spawn 2 validators in parallel:
- Backend validator: DB/migrations/seeds/API/job/S3 tests.
- Frontend validator: UI routes, API client, build/lint, UX states.
```

Do not claim completion until validation reports are reviewed and blockers fixed.

## Future v2

- JWT/session auth.
- RBAC: `kaggle.view`, `kaggle.manage`, `kaggle.execute`.
- Audit logs.
- Rate limiting.
- Celery/RQ + Redis.
- Dataset/competition APIs.
- Parse `training_results.txt` into metrics dashboard.
- Artifact cleanup policy.
- Dedicated `kaggle_job_artifacts` table for multiple S3 files per job.
- Manual multipart upload UI if direct browser-to-S3 uploads become needed.
