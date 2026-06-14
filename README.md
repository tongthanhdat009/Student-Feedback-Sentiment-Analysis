# Student Feedback Sentiment Analysis

Vietnamese student-feedback sentiment classification on **UIT-VSFC**. Project combines notebook experiments, reusable Python utilities, stored experiment results, and a Kaggle run-management web app.

## What this repo does

- Classifies feedback into 3 sentiment labels: `0 = Negative`, `1 = Neutral`, `2 = Positive`.
- Uses UIT-VSFC train/validation/test splits: `11426 / 1583 / 3166` samples.
- Compares 4 main experiment families:
  - **PhoBERT baseline** — fine-tune `vinai/phobert-base`.
  - **PhoBERT + TF-IDF** — contextual embeddings + sparse n-gram features.
  - **PhoBERT + VietSentiWordNet** — contextual embeddings + lexical sentiment scores.
  - **PhoBERT + TF-IDF + VietSentiWordNet** — full hybrid feature fusion.
- Includes a **Kaggle Notebook Manager** dashboard for remote training orchestration.

## Repository map

```text
.
├── src/                         # Core Python utilities: data loading, fusion, ensemble logic
├── notebook/                    # Main experiment notebooks
│   ├── PhoBERT_Baseline.ipynb
│   ├── PhoBERT_TFIDF.ipynb
│   ├── PhoBERT_Sentiwordnet.ipynb
│   └── PhoBERT_TF-IDF_Sentiwordnet.ipynb
├── data/                        # UIT-VSFC splits + VietSentiWordNet resources
├── results/                     # Experiment outputs grouped by model/date
├── tools/                       # Weight tuning CLI/tools
├── docs/                        # Analysis + optimization reports
├── backend/                     # FastAPI Kaggle Notebook Manager API
├── frontend/                    # React + Vite Kaggle dashboard
├── storage/                     # Local Kaggle output/staging storage
├── requirements.txt             # ML/notebook deps
├── CLAUDE.md                    # Detailed architecture/workflow notes
└── README.md
```

## ML architecture

```text
UIT-VSFC text
  └─ preprocessing
      ├─ lowercase / whitespace normalize / special-char cleanup
      ├─ Vietnamese teencode normalization: "ko" → "không", "j" → "gì", ...
      └─ PhoBERT BPE tokenization

Feature branches
  ├─ PhoBERT CLS embedding: 768 dims
  ├─ TF-IDF: 3000–5000 n-gram features, usually ngram_range=(1, 3)
  └─ VietSentiWordNet: 8 basic or 35 extended sentiment features

Fusion
  ├─ concatenate / normalize / optional weighted feature search
  └─ classifier: LogisticRegression or XGBoost
```

## Dataset

**UIT-VSFC** — Vietnamese Students' Feedback Corpus.

| Split | Samples |
|---|---:|
| Train | 11,426 |
| Validation | 1,583 |
| Test | 3,166 |

Labels:

| Label | Sentiment |
|---:|---|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

Notes:

- Neutral class is small (~4% in train). Prefer **macro-F1** for model selection.
- Raw/processed dataset files live under `data/`.
- VietSentiWordNet resource lives under `data/sentiwordnet-dataset/`.

## Quick start: ML experiments

### 1. Create env + install deps

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows Git Bash/PowerShell may differ
pip install -r requirements.txt
```

For CUDA PyTorch, install matching wheel first/explicitly, e.g.:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Run notebooks

```bash
jupyter notebook notebook/PhoBERT_Baseline.ipynb
jupyter notebook notebook/PhoBERT_TFIDF.ipynb
jupyter notebook notebook/PhoBERT_Sentiwordnet.ipynb
jupyter notebook notebook/PhoBERT_TF-IDF_Sentiwordnet.ipynb
```

Colab note: notebooks may use Drive paths. If running locally, update `BASE_DIR` in config cells.

### 3. Main Google Colab notebooks

Primary experiments run on Google Colab with the main dataset under `data/`:

- `notebook/PhoBERT_Baseline.ipynb`
- `notebook/PhoBERT_TF-IDF_Sentiwordnet.ipynb`
- `notebook/PhoBERT_Sentiwordnet.ipynb`
- `notebook/PhoBERT_TFIDF.ipynb`

Data used by these notebooks:

- raw data: `data/raw/`
- processed data: `data/processed/`
- sentiment lexicon: `data/sentiwordnet-dataset/`

Kaggle is for trial/remote experiment runs only. Main validated workflow uses these 4 notebooks + `data/` on Google Colab.

## Outputs

Experiment outputs should be stored like:

```text
results/{ModelType}/{baseline|improvements}/{YYYYMMDD}/
├── models/                 # .pt / .pkl / .joblib models
├── summaries/              # summary.csv, training_results.txt
├── visualizations/         # confusion matrices, charts, training curves
└── artifacts/              # vectorizers, scalers, metadata
```

Required summary artifact: `summaries/training_results.txt` using the project format described in `CLAUDE.md`.

## Kaggle Notebook Manager

This repo also contains a full-stack MVP for remote Kaggle experiment orchestration.

### Backend

FastAPI API for accounts, datasets, notebook inventory/sync/trigger, job polling, output download, metric parsing, S3-compatible artifact upload.

```bash
cd backend
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

Important env vars:

| Var | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL async URL, e.g. `postgresql+asyncpg://user:pass@host:5432/db` |
| `ADMIN_API_KEY` | Admin API token sent as `X-API-Key` |
| `FERNET_KEY` | Encrypts stored Kaggle API credentials |
| `KAGGLE_NOTEBOOK_DIR` | Local Kaggle notebook dir, usually `../notebook/kaggle` |
| `KAGGLE_OUTPUT_DIR` | Local output dir, usually `../storage/kaggle_outputs` |
| `MAX_KAGGLE_JOBS` | Max concurrent workers, default `2` |
| `S3Storage__BucketName` | S3 bucket/prefix for artifacts |
| `S3Storage__Region` | S3 region |
| `S3Storage__AccessKey` | S3 access key |
| `S3Storage__SecretKey` | S3 secret key |
| `S3Storage__ServiceUrl` | S3-compatible endpoint, optional |
| `S3Storage__UsePathStyle` | Path-style access for MinIO/LocalStack, optional |

### Frontend

React + Vite dashboard for accounts, datasets, notebook sync/run, jobs, audit/log views.

```bash
cd frontend
npm install
npm run dev
npm run build
npm run lint
```

Frontend env:

| Var | Purpose |
|---|---|
| `VITE_API_BASE` | Backend base URL, default `http://127.0.0.1:8000` |
| `VITE_ADMIN_API_KEY` | Sent as `X-API-Key` |

## Validation commands

Use commands matching changed area:

```bash
# ML deps smoke
python -m compileall src tools

# Backend
cd backend
pytest

# Frontend
cd frontend
npm run build
npm run lint
```

Some checks need external services:

- Backend tests may require PostgreSQL at `DATABASE_URL`.
- Kaggle sync/trigger needs valid Kaggle credentials.
- S3 upload/presigned URL needs configured S3-compatible storage.

## Important caveats

- Use **macro-F1**, not accuracy/weighted-F1 alone, because Neutral is minority.
- Keep feature modalities aligned. If using SMOTE/oversampling, do not resample TF-IDF separately from PhoBERT embeddings.
- Kaggle dataset refs must use strict `owner/dataset-slug` format.
- Losing `FERNET_KEY` means existing encrypted Kaggle tokens cannot be decrypted.
- Windows encoding can break Kaggle client uploads; backend includes CP-1252-safe notebook writing.

## References

- UIT-VSFC dataset: <https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback>
- UIT-VSFC paper: <https://ieeexplore.ieee.org/document/8573337>
- PhoBERT: <https://github.com/VinAIResearch/PhoBERT>
