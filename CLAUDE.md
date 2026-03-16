# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vietnamese Student Feedback Sentiment Analysis using the **UIT-VSFC** dataset. The project implements multiple approaches for 3-class sentiment classification (Negative=0/Neutral=1/Positive=2):

- **PhoBERT Baseline**: Fine-tuned vinai/phobert-base transformer
- **TF-IDF Hybrid**: PhoBERT embeddings + TF-IDF features
- **SentiWordNet Hybrid**: PhoBERT embeddings + 8 sentiment lexicon features
- **Full Hybrid**: All features combined

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        KIẾN TRÚC HỆ THỐNG PHÂN LOẠI CẢM XÚC                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          KHỐI 1: TIỀN XỬ LÝ DỮ LIỆU                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Đầu vào: Bộ dữ liệu UIT-VSFC                                                   │
│                                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐                 │
│  │ Làm sạch    │───▶│ Chuẩn hóa        │───▶│ PhoBERT BPE  │                 │
│  │ văn bản     │    │ từ ngữ mạng      │    │ Tokenization │                 │
│  └──────────────┘    └──────────────────┘    └───────────────┘                 │
│        │                    │                        │                         │
│        ▼                    ▼                        ▼                         │
│  - Lowercase          - "ko" → "không"         - Subword tokenization          │
│  - Remove special     - "j" → "gì"              - Max length: 256              │
│    characters         - "cx" → "cũng"                                          │
│  - Normalize          - 100+ teencode mappings                                 │
│    whitespace                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        KHỐI 2: TRÍCH XUẤT ĐẶC TRƯNG                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────┐      ┌─────────────────────────────┐          │
│  │     NHÁNH 1: PhoBERT        │      │   NHÁNH 2: Traditional      │          │
│  │     (Vector ngữ cảnh sâu)   │      │   (Tần suất + Từ điển)      │          │
│  ├─────────────────────────────┤      ├─────────────────────────────┤          │
│  │                             │      │                             │          │
│  │  vinai/phobert-base         │      │  ┌─────────────────────┐   │          │
│  │  ↓                         │      │  │ TF-IDF              │   │          │
│  │  [CLS] token extraction    │      │  │ - 3000-5000 features│   │          │
│  │  ↓                         │      │  │ - n-grams: (1,3)    │   │          │
│  │  768-dim embedding         │      │  └─────────────────────┘   │          │
│  │                             │      │           +                 │          │
│  │  Fine-tuned on UIT-VSFC    │      │  ┌─────────────────────┐   │          │
│  │                             │      │  │ VietSentiWordNet    │   │          │
│  │                             │      │  │ - pos_sum, neg_sum  │   │          │
│  │                             │      │  │ - pos_max, neg_max  │   │          │
│  │                             │      │  │ - pos_mean, neg_mean│   │          │
│  │                             │      │  │ - coverage, polarity│   │          │
│  │                             │      │  │ = 8 features        │   │          │
│  │                             │      │  └─────────────────────┘   │          │
│  └─────────────────────────────┘      └─────────────────────────────┘          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      KHỐI 3: XÂY DỰNG MÔ HÌNH (LAI GHÉP)                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                        4 CẤU HÌNH THỰC NGHIỆM                             │ │
│  ├───────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                           │ │
│  │  1. PhoBERT cơ sở          2. PhoBERT + TF-IDF                           │ │
│  │     ┌─────────┐               ┌─────────┐                                │ │
│  │     │PhoBERT  │               │PhoBERT  │                                │ │
│  │     │ 768-dim │               │ 768-dim │                                │ │
│  │     └────┬────┘               └────┬────┘                                │ │
│  │          │                         │                                      │ │
│  │          ▼                    ┌────┴────┐                                 │ │
│  │     ┌─────────┐               │ TF-IDF  │                                 │ │
│  │     │Classifier│              │3000-dim │                                 │ │
│  │     └─────────┘               └────┬────┘                                 │ │
│  │                                    │                                      │ │
│  │                                    ▼                                      │ │
│  │                               ┌──────────┐                                │ │
│  │                               │Concatenate│                               │ │
│  │                               │ 3768-dim │                                │ │
│  │                               └────┬─────┘                                │ │
│  │                                    │                                      │ │
│  │                                    ▼                                      │ │
│  │                               ┌──────────┐                                │ │
│  │                               │Classifier│                                │ │
│  │                               └──────────┘                                │ │
│  │                                                                           │ │
│  │  3. PhoBERT + SentiWordNet   4. PhoBERT + TF-IDF + SentiWordNet         │ │
│  │     ┌─────────┐               ┌─────────┐                                │ │
│  │     │PhoBERT  │               │PhoBERT  │                                │ │
│  │     │ 768-dim │               │ 768-dim │                                │ │
│  │     └────┬────┘               └────┬────┘                                │ │
│  │          │                    ┌────┴────┐                                 │ │
│  │          │                    │ TF-IDF  │                                 │ │
│  │    ┌─────┴─────┐              │3000-dim │                                 │ │
│  │    │SentiWordNet│             └────┬────┘                                 │ │
│  │    │  8-dim    │              ┌────┴────┐                                 │ │
│  │    └─────┬─────┘              │SentiWordNet│                              │ │
│  │          │                    │  8-dim   │                                │ │
│  │          ▼                    └────┬────┘                                 │ │
│  │     ┌──────────┐                   │                                      │ │
│  │     │Concatenate│                  ▼                                      │ │
│  │     │ 776-dim  │              ┌──────────┐                                │ │
│  │     └────┬─────┘              │Concatenate│                               │ │
│  │          │                    │ 3776-dim │                                │ │
│  │          ▼                    └────┬─────┘                                │ │
│  │     ┌──────────┐                   │                                      │ │
│  │     │Classifier│                   ▼                                      │ │
│  │     └──────────┘              ┌──────────┐                                │ │
│  │                               │Classifier│                                │ │
│  │                               └──────────┘                                │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  Classifiers: LogisticRegression, XGBoost                                       │
│  Loss: CrossEntropyLoss (training PhoBERT)                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        KHỐI 4: ĐÁNH GIÁ & PHÂN TÍCH                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │  Tối ưu hóa     │    │  Phân loại      │    │  Đo lường       │            │
│  │  CrossEntropy   │───▶│  3 classes      │───▶│  hiệu suất      │            │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘            │
│                                                        │                       │
│                                                        ▼                       │
│                                              ┌─────────────────────┐          │
│                                              │ Metrics:            │          │
│                                              │ - Accuracy          │          │
│                                              │ - F1-score (macro)  │          │
│                                              │ - Precision/Recall  │          │
│                                              │ - Confusion Matrix  │          │
│                                              └─────────────────────┘          │
│                                                                                 │
│  Output: Tích cực (Positive) / Trung tính (Neutral) / Tiêu cực (Negative)      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Methodology Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     QUY TRÌNH PHƯƠNG PHÁP PHÂN LOẠI CẢM XÚC                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                     GIAI ĐOẠN 1: TIỀN XỬ LÝ DỮ LIỆU                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ Bộ dữ liệu UIT-VSFC                                                      │   │
│  │ Train: 11,426 | Validation: 1,583 | Test: 3,166                         │   │
│  └─────────────────────────────────────┬───────────────────────────────────┘   │
│                                        │                                        │
│                                        ▼                                        │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐          │
│  │  Làm sạch văn bản │───▶│Chuẩn hóa teencode│───▶│  Tokenization   │          │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘          │
│           │                       │                       │                    │
│           ▼                       ▼                       ▼                    │
│    • Lowercase           • ko → không          • PhoBERT BPE                   │
│    • Remove special      • j → gì              • Subword tokenization          │
│      characters          • cx → cũng           • Max length: 256              │
│    • Normalize           • dc → được                                          │
│      whitespace          • 178+ mappings                                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                     GIAI ĐOẠN 2: TRÍCH XUẤT ĐẶC TRƯNG                          │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│                      ┌─────────────────────────────────────┐                   │
│                      │    Văn bản đã tiền xử lý            │                   │
│                      └──────────────┬──────────────────────┘                   │
│                                     │                                          │
│           ┌─────────────────────────┼─────────────────────────┐                │
│           │                         │                         │                │
│           ▼                         ▼                         ▼                │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐           │
│  │  NHÁNH 1        │     │  NHÁNH 2        │     │  NHÁNH 3        │           │
│  │  PhoBERT        │     │  TF-IDF         │     │ VietSentiWordNet│           │
│  ├─────────────────┤     ├─────────────────┤     ├─────────────────┤           │
│  │                 │     │                 │     │                 │           │
│  │ vinai/          │     │ 3,000-5,000     │     │ 8 features:     │           │
│  │ phobert-base    │     │ features        │     │ • pos_sum       │           │
│  │                 │     │                 │     │ • neg_sum       │           │
│  │ [CLS] token     │     │ n-grams: (1,3)  │     │ • pos_max       │           │
│  │ extraction      │     │                 │     │ • neg_max       │           │
│  │                 │     │ min_df=3        │     │ • pos_mean      │           │
│  │ 768-dim         │     │ max_df=0.9      │     │ • neg_mean      │           │
│  │ embedding       │     │                 │     │ • coverage      │           │
│  │                 │     │                 │     │ • polarity      │           │
│  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘           │
│           │                       │                       │                    │
│           └───────────────────────┼───────────────────────┘                    │
│                                   │                                            │
└───────────────────────────────────┼────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     GIAI ĐOẠN 3: LAI GHÉP ĐẶC TRƯNG                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                        ┌─────────────────────────────┐                         │
│                        │     Feature Fusion          │                         │
│                        └──────────────┬──────────────┘                         │
│                                       │                                         │
│           ┌───────────────────────────┼───────────────────────────┐            │
│           │                           │                           │            │
│           ▼                           ▼                           ▼            │
│  ┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐       │
│  │  Concatenate    │     │ StandardScaler      │     │  SMOTE          │       │
│  │  embeddings     │     │ normalization       │     │  (class balance)│       │
│  └────────┬────────┘     └──────────┬──────────┘     └────────┬────────┘       │
│           │                         │                         │                │
│           └─────────────────────────┼─────────────────────────┘                │
│                                     │                                          │
│                                     ▼                                          │
│                        ┌─────────────────────────────┐                         │
│                        │  Hybrid Feature Vector      │                         │
│                        │  768 + 5000 + 8 = 5,776 dim │                         │
│                        └─────────────────────────────┘                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     GIAI ĐOẠN 4: HUẤN LUYỆN MÔ HÌNH                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                        4 CẤU HÌNH THỰC NGHIỆM                             │ │
│  ├───────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                           │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │ │
│  │  │ 1. PhoBERT      │  │ 2. PhoBERT+     │  │ 3. PhoBERT+     │            │ │
│  │  │    Baseline     │  │    TF-IDF       │  │    SentiWordNet │            │ │
│  │  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤            │ │
│  │  │                 │  │                 │  │                 │            │ │
│  │  │ PhoBERT: 768    │  │ PhoBERT: 768    │  │ PhoBERT: 768    │            │ │
│  │  │                 │  │ TF-IDF: 3000    │  │ SentiWordNet: 8 │            │ │
│  │  │                 │  │ Total: 3768     │  │ Total: 776      │            │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘            │ │
│  │                                                                           │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │ │
│  │  │ 4. Full Hybrid (PhoBERT + TF-IDF + SentiWordNet)                   │  │ │
│  │  ├─────────────────────────────────────────────────────────────────────┤  │ │
│  │  │ PhoBERT: 768 | TF-IDF: 3000 | SentiWordNet: 8 | Total: 3,776 dim   │  │ │
│  │  └─────────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                           │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                          │
│                                     ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        CLASSIFIERS                                       │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐             │   │
│  │  │ Logistic       │  │ XGBoost        │  │ Late Fusion    │             │   │
│  │  │ Regression     │  │                │  │ (Ensemble)     │             │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                          │
│                                     ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        LOSS FUNCTION                                     │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │  CrossEntropyLoss | Optimizer: AdamW | Learning Rate: 2e-5             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     GIAI ĐOẠN 5: ĐÁNH GIÁ                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        METRICS                                           │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │  • Accuracy           • Precision (per class)                           │   │
│  │  • F1-score (macro)   • Recall (per class)                              │   │
│  │  • Confusion Matrix                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                          │
│                                     ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        KẾT QUẢ PHÂN LOẠI                                  │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐            │   │
│  │  │   POSITIVE    │    │   NEUTRAL     │    │   NEGATIVE    │            │   │
│  │  │   (Tích cực)  │    │   (Trung tính)│    │   (Tiêu cực) │            │   │
│  │  │   Label: 2    │    │   Label: 1    │    │   Label: 0   │            │   │
│  │  └───────────────┘    └───────────────┘    └───────────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Running Notebooks

```bash
pip install -r requirements.txt
jupyter notebook
```

**Note**: Notebooks are configured for Google Colab with paths like `/content/drive/MyDrive/Student-Feedback-Sentiment-Analysis`. When running locally, update `BASE_DIR` in Config classes.

## Data Structure

```
data/
├── processed/{train,validation,test}/   # sents.txt, sentiments.txt, topics.txt
├── raw/                                  # Original UIT-VSFC data
└── sentiwordnet-dataset/                 # VietSentiWordnet_Ver1.3.5.txt
```

**Dataset**: Train 11,426 / Val 1,583 / Test 3,166 samples. **Class imbalance**: Neutral is only 4% of training data.

## Data Utilities Module

Use `src/data_utils.py` for centralized data handling:

```python
from src.data_utils import (
    load_data,              # Load UIT-VSFC split
    load_all_splits,        # Load all splits at once
    load_sentiwordnet,      # Load VietSentiWordNet lexicon
    normalize_teencode,     # Normalize Vietnamese internet slang
    preprocess_vietnamese,  # Full Vietnamese text preprocessing
    get_swn_features,       # Extract 8 SentiWordNet features
    extract_swn_features_batch,  # Batch feature extraction
    SWN_FEATURE_NAMES,      # ['pos_sum', 'neg_sum', ...]
    LABEL_MAP,              # {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
)

# Example usage
texts, labels = load_data('data/processed', 'train')
word_to_scores = load_sentiwordnet('data/sentiwordnet-dataset/VietSentiWordnet_Ver1.3.5.txt')
features = get_swn_features(texts[0], word_to_scores)  # Returns 8 features

# Normalize teencode
normalized = normalize_teencode("ko bt j cả")  # -> "không biết gì cả"
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `EDA.ipynb` | Exploratory data analysis |
| `PhoBERT_Baseline.ipynb` | Fine-tuned PhoBERT (768-dim, CrossEntropyLoss) |
| `PhoBERT_TF-IDF_Baseline.ipynb` | PhoBERT + TF-IDF (5768 features) |
| `PhoBERT_Sentiwordnet_Baseline.ipynb` | PhoBERT + 8 SentiWordNet features (776-dim) |
| `PhoBERT_TF-IDF_Sentiwordnet_Baseline.ipynb` | All features combined |

## Key Hyperparameters

- PhoBERT: `vinai/phobert-base`, lr=2e-5, warmup=10%, early stopping patience=5
- TF-IDF: 5000 features, 1-2 grams
- LogisticRegression: C=1.0, max_iter=1000

## Vietnamese Text Preprocessing

The preprocessing pipeline includes:

### 1. Text Cleaning
```python
def preprocess_vietnamese(text, normalize_slang=True):
    text = text.lower()
    if normalize_slang:
        text = normalize_teencode(text)  # Convert teencode to standard Vietnamese
    text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    return ' '.join(text.split())
```

### 2. Teencode Normalization
Common Vietnamese internet slang conversions:
| Teencode | Standard |
|----------|----------|
| ko, k, hok, kg | không |
| j, gj | gì |
| cx | cũng |
| dc, đc | được |
| bt, bik | biết |
| lm, lém | lắm |
| qa, qá | quá |
| e | em |
| thầy, thay | thầy |

See `TEENCODE_DICT` in `src/data_utils.py` for full list (100+ mappings).

## Results Directory Structure

```
results/{ModelType}/{baseline|improvements}/{YYYYMMDD}/
├── models/           # phobert_model.pt, xgboost_model.pkl
├── summaries/        # summary.csv, training_results.txt (REQUIRED)
├── visualizations/   # confusion_matrix.png, training_history.png
└── artifacts/        # tfidf_vectorizer.pkl, scaler.pkl
```

**Rules**:
- `baseline/` has NO timestamp (reference version)
- `improvements/YYYYMMDD/` has timestamp for each experiment iteration
- Every experiment MUST include `summaries/training_results.txt`

## Training Results Format (REQUIRED)

```
========================================
TRAINING RESULTS - {Model Name}
========================================
Date: {YYYY-MM-DD HH:MM:SS}
Model Type: {PhoBERT / PhoBERT_TF-IDF / PhoBERT_Sentiwordnet}
Experiment: {baseline / improvements/}

----------------------------------------
HYPERPARAMETERS
----------------------------------------
Learning Rate: {value}
Batch Size: {value}
...

----------------------------------------
TEST RESULTS
----------------------------------------
Test Accuracy: {value}
Test F1 (macro): {value}

Per-Class Metrics:
  Negative: P={value}, R={value}, F1={value}
  Neutral:  P={value}, R={value}, F1={value}
  Positive: P={value}, R={value}, F1={value}

----------------------------------------
CONFUSION MATRIX
----------------------------------------
[[...]]
```

## Common Issues

- **Class Imbalance**: Neutral is 4% of training data. Solutions: SMOTE, class-weighted loss, focal loss.
- **Memory**: PhoBERT embedding extraction is slow. Cache extracted embeddings.
- **Model Loading**: Use `load_model_safe()` helper to handle state_dict with/without "module." prefix from DataParallel.

## Best Results

| Model | Test Acc | Test F1 | Notes |
|-------|----------|---------|-------|
| PhoBERT Baseline | 0.934 | 0.931 | CrossEntropyLoss |
| TF-IDF Hybrid + XGBoost + SMOTE | 0.933 | 0.932 | Best for Neutral class |
