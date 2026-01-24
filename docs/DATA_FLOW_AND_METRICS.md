# 📊 LUỒNG DỮ LIỆU VÀ PHƯƠNG PHÁP ĐÁNH GIÁ

## Mục lục
1. [Tổng quan luồng dữ liệu](#1-tổng-quan-luồng-dữ-liệu)
2. [Chi tiết từng giai đoạn](#2-chi-tiết-từng-giai-đoạn)
3. [Train / Validation / Test Split](#3-train--validation--test-split)
4. [Các metrics đánh giá](#4-các-metrics-đánh-giá)
5. [Chi tiết từng model](#5-chi-tiết-từng-model)

---

## 1. Tổng quan luồng dữ liệu

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LUỒNG DỮ LIỆU TỔNG THỂ                          │
└─────────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐
     │ Huggingface  │
     │   Dataset    │
     │  (Parquet)   │
     └──────┬───────┘
            │ download_dataset.py
            ▼
     ┌──────────────┐
     │   Raw Data   │
     │  (CSV files) │
     │ 16,175 samples│
     └──────┬───────┘
            │ build_dataset.py
            ▼
     ┌──────────────┐
     │  Processed   │
     │    Data      │
     │ + TF-IDF     │
     │ + Labels     │
     └──────┬───────┘
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│   SVM   │   │  LSTM   │   │ PhoBERT │
│ Training│   │ Training│   │ Training│
└────┬────┘   └────┬────┘   └────┬────┘
     │             │             │
     └──────┬──────┴──────┬──────┘
            ▼             ▼
     ┌──────────────┐   ┌──────────────┐
     │    Saved     │   │   Results    │
     │    Models    │   │ (metrics.json)│
     └──────────────┘   └──────────────┘
            │
            ▼
     ┌──────────────┐
     │  Dashboard   │
     │ (Streamlit)  │
     └──────────────┘
```

---

## 2. Chi tiết từng giai đoạn

### 2.1. Download Dataset (`data/download_dataset.py`)

**Input:** Huggingface API (Parquet files)
**Output:** CSV files trong `data/raw/uit_vsfc/`

```
Huggingface API
    │
    ├── https://huggingface.co/api/datasets/uitnlp/vietnamese_students_feedback/parquet/default/train
    ├── https://huggingface.co/api/datasets/uitnlp/vietnamese_students_feedback/parquet/default/validation
    └── https://huggingface.co/api/datasets/uitnlp/vietnamese_students_feedback/parquet/default/test
    │
    ▼
CSV Files:
    ├── train.csv       (11,426 samples)
    ├── validation.csv  (1,583 samples)
    └── test.csv        (3,166 samples)
```

**Cấu trúc dữ liệu:**
| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `sentence` | string | Văn bản phản hồi tiếng Việt |
| `sentiment` | int | 0=Negative, 1=Neutral, 2=Positive |
| `topic` | int | 0=Lecturer, 1=Training Program, 2=Facility, 3=Others |

---

### 2.2. Preprocessing (`preprocessing/build_dataset.py`)

**Input:** Raw CSV files
**Output:** Processed data + TF-IDF features

```
Raw CSV
    │
    ├── 1. Clean Text
    │   ├── Lowercase
    │   ├── Remove special characters
    │   └── Remove extra spaces
    │
    ├── 2. Build TF-IDF (cho SVM)
    │   ├── Fit on train set
    │   ├── Transform train/val/test
    │   └── max_features=5000, ngram_range=(1,2)
    │
    └── 3. Save Labels
        ├── sentiment labels
        └── topic labels

Output Files:
    ├── train_processed.csv
    ├── validation_processed.csv
    ├── test_processed.csv
    ├── tfidf_vectorizer.pkl
    ├── tfidf_train.pkl
    ├── tfidf_validation.pkl
    ├── tfidf_test.pkl
    ├── labels.pkl
    └── metadata.json
```

---

## 3. Train / Validation / Test Split

### 3.1. Phân chia dữ liệu

```
                    UIT-VSFC Dataset
                    (16,175 samples)
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   ┌─────────┐      ┌─────────┐       ┌─────────┐
   │  TRAIN  │      │VALIDATE │       │  TEST   │
   │ 11,426  │      │  1,583  │       │  3,166  │
   │ (70.6%) │      │ (9.8%)  │       │ (19.6%) │
   └────┬────┘      └────┬────┘       └────┬────┘
        │                │                 │
        ▼                ▼                 ▼
   Học patterns    Tune hyperparams   Đánh giá cuối
   Cập nhật weights Early stopping    Báo cáo metrics
```

### 3.2. Vai trò của từng tập

| Tập | Vai trò | Khi nào sử dụng |
|-----|---------|-----------------|
| **Train** | Model học từ dữ liệu này | Mỗi epoch, cập nhật weights |
| **Validation** | Đánh giá trong quá trình train | Sau mỗi epoch, không cập nhật weights |
| **Test** | Đánh giá cuối cùng | Chỉ 1 lần sau khi train xong |

### 3.3. Tại sao cần 3 tập?

```
❌ Train + Test only:
   - Có thể overfit vào test set khi tune hyperparameters
   - Không biết khi nào dừng training

✅ Train + Validation + Test:
   - Validation: tune hyperparams, early stopping
   - Test: đánh giá unbiased cuối cùng
   - Kết quả test phản ánh thực tế
```

---

## 4. Các Metrics Đánh Giá

### 4.1. Confusion Matrix

Với bài toán 3 classes (Negative, Neutral, Positive):

```
                    Predicted
                Neg   Neu   Pos
            ┌──────┬──────┬──────┐
       Neg  │  TN  │ FNeu │ FPos │  ← Actual Negative
Actual      ├──────┼──────┼──────┤
       Neu  │ FNeg │  TN  │ FPos │  ← Actual Neutral
            ├──────┼──────┼──────┤
       Pos  │ FNeg │ FNeu │  TP  │  ← Actual Positive
            └──────┴──────┴──────┘
```

### 4.2. Accuracy

**Công thức:**
```
                    Số dự đoán đúng
Accuracy = ─────────────────────────────
                  Tổng số samples

           TP + TN (tất cả classes)
         = ────────────────────────
               Tổng số samples
```

**Ví dụ:**
```
Predictions: [Pos, Neg, Pos, Neu, Pos, Neg, Pos, Neg]
Actual:      [Pos, Neg, Neg, Neu, Pos, Pos, Pos, Neg]
                ✓    ✓    ✗    ✓    ✓    ✗    ✓    ✓

Accuracy = 6/8 = 0.75 = 75%
```

**Code:**
```python
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true, y_pred)
```

---

### 4.3. Precision, Recall, F1-Score

#### Precision (Độ chính xác)
```
                        True Positives
Precision (class c) = ───────────────────────────────────
                      True Positives + False Positives

"Trong những cái model dự đoán là class c, bao nhiêu % đúng?"
```

#### Recall (Độ phủ / Sensitivity)
```
                     True Positives
Recall (class c) = ───────────────────────────────────
                   True Positives + False Negatives

"Trong những cái thực sự là class c, model tìm được bao nhiêu %?"
```

#### F1-Score (Harmonic Mean)
```
              2 × Precision × Recall
F1-Score = ─────────────────────────────
              Precision + Recall

"Cân bằng giữa Precision và Recall"
```

**Ví dụ chi tiết cho class "Positive":**
```
Actual:      [Pos, Neg, Pos, Neu, Pos, Neg, Pos, Neg]
Predictions: [Pos, Neg, Neg, Neu, Pos, Pos, Pos, Neg]

Với class Positive:
- True Positive (TP): Actual=Pos, Pred=Pos → 3 (vị trí 0, 4, 6)
- False Positive (FP): Actual≠Pos, Pred=Pos → 1 (vị trí 5)
- False Negative (FN): Actual=Pos, Pred≠Pos → 1 (vị trí 2)

Precision = TP / (TP + FP) = 3 / (3 + 1) = 0.75
Recall    = TP / (TP + FN) = 3 / (3 + 1) = 0.75
F1        = 2 × 0.75 × 0.75 / (0.75 + 0.75) = 0.75
```

---

### 4.4. Weighted F1-Score

Vì các classes có số lượng khác nhau, ta dùng **weighted average**:

```
                    Σ (F1_c × support_c)
Weighted F1 = ──────────────────────────────
                    Σ support_c

Trong đó: support_c = số samples của class c
```

**Ví dụ:**
```
Class       | F1    | Support | Weighted
------------|-------|---------|----------
Negative    | 0.80  | 100     | 80
Neutral     | 0.60  | 20      | 12
Positive    | 0.85  | 80      | 68
------------|-------|---------|----------
Total       |       | 200     | 160

Weighted F1 = 160 / 200 = 0.80
```

**Code:**
```python
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred, average='weighted')
```

---

## 5. Chi tiết từng Model

### 5.1. SVM (Support Vector Machine)

```
┌─────────────────────────────────────────────────────────┐
│                    SVM PIPELINE                         │
└─────────────────────────────────────────────────────────┘

Input Text          TF-IDF Vectorizer         SVM Classifier
     │                    │                        │
     ▼                    ▼                        ▼
"Giảng viên      [0.2, 0.0, 0.5,         Find optimal
 rất nhiệt tình"  0.3, 0.1, ...]          hyperplane
     │                    │                        │
     └────────────────────┴────────────────────────┘
                          │
                          ▼
                    Prediction: Positive (2)
```

**Cách hoạt động:**
1. Text → TF-IDF vector (sparse, high-dimensional)
2. SVM tìm hyperplane phân tách các classes
3. RBF kernel xử lý non-linear boundaries

**Hyperparameters:**
- `kernel='rbf'`: Radial Basis Function
- `C=1.0`: Regularization parameter
- `gamma='scale'`: Kernel coefficient

---

### 5.2. LSTM (Long Short-Term Memory)

```
┌─────────────────────────────────────────────────────────┐
│                    LSTM PIPELINE                        │
└─────────────────────────────────────────────────────────┘

Input Text          Embedding              Bi-LSTM           FC
     │                 │                     │               │
     ▼                 ▼                     ▼               ▼
"Giảng viên    [vec1, vec2,         →  LSTM  →        Linear
 rất tốt"       vec3, vec4]         ←  LSTM  ←         │
     │                 │                     │               │
     ▼                 ▼                     ▼               ▼
[15, 42,       256-dim for         Hidden state       3 classes
 78, 91]       each word            (256-dim)         softmax
```

**Training Loop:**
```python
for epoch in range(20):
    # Forward pass
    for batch in train_loader:
        outputs = model(batch.text)
        loss = CrossEntropyLoss(outputs, batch.label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    # Validate (không backward)
    with torch.no_grad():
        val_acc = evaluate(model, val_loader)
```

**Hyperparameters:**
- `embedding_dim=256`
- `hidden_dim=128`
- `num_layers=2`
- `dropout=0.3`
- `bidirectional=True`

---

### 5.3. PhoBERT (Vietnamese BERT)

```
┌─────────────────────────────────────────────────────────┐
│                   PhoBERT PIPELINE                      │
└─────────────────────────────────────────────────────────┘

Input Text         PhoBERT Tokenizer        PhoBERT Model      Classifier
     │                   │                       │                 │
     ▼                   ▼                       ▼                 ▼
"Giảng viên       [CLS] Giảng viên        12 Transformer     Linear(768→3)
 rất nhiệt tình"   rất nhiệt tình [SEP]   Layers                │
     │                   │                       │                 │
     ▼                   ▼                       ▼                 ▼
                  input_ids +              [CLS] embedding    Prediction
                  attention_mask            (768-dim)          Positive
```

**Fine-tuning Process:**
```python
# Load pretrained PhoBERT (từ cache)
model = RobertaForSequenceClassification.from_pretrained(
    'vinai/phobert-base',
    num_labels=3
)

# Fine-tune with small learning rate
optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(5):
    for batch in train_loader:
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

**Hyperparameters:**
- `max_length=256`
- `batch_size=16`
- `epochs=5`
- `learning_rate=2e-5`

---

## 6. So sánh các Models

### 6.1. Đặc điểm

| Đặc điểm | SVM | LSTM | PhoBERT |
|----------|-----|------|---------|
| **Loại** | Classical ML | Deep Learning | Transformer |
| **Input** | TF-IDF vectors | Word indices | Subword tokens |
| **Parameters** | ~10K | ~3M | ~135M |
| **Training time** | Seconds | Minutes | Hours |
| **Context** | Bag of words | Sequential | Bidirectional attention |
| **Pretrained?** | No | No | Yes (PhoBERT-base) |

### 6.2. Kỳ vọng Performance

```
Accuracy (thường):

PhoBERT > LSTM > SVM

Lý do:
- PhoBERT: Pretrained on large Vietnamese corpus
- LSTM: Learns sequential patterns
- SVM: Only uses word frequencies (TF-IDF)
```

---

## 7. Tóm tắt Workflow

```
1. DOWNLOAD
   └── python data/download_dataset.py
       └── Output: data/raw/uit_vsfc/*.csv

2. PREPROCESS
   └── python preprocessing/build_dataset.py
       └── Output: data/processed/acsa/*

3. TRAIN SVM
   └── python training/train_svm.py
       ├── Input: TF-IDF features
       ├── Train on train set
       ├── Evaluate on val/test
       └── Output: saved_models/svm/

4. TRAIN LSTM
   └── python training/train_lstm.py
       ├── Input: Word indices
       ├── Train 20 epochs
       ├── Validate each epoch
       └── Output: saved_models/lstm/

5. TRAIN PhoBERT
   └── python training/train_phobert.py
       ├── Input: PhoBERT tokens
       ├── Fine-tune 5 epochs
       ├── Validate each epoch
       └── Output: saved_models/phobert/

6. DASHBOARD
   └── streamlit run dashboard/app.py
       └── Visualize all results
```

---

## 8. References

- **UIT-VSFC Paper**: https://ieeexplore.ieee.org/document/8573337
- **PhoBERT**: https://github.com/VinAIResearch/PhoBERT
- **Scikit-learn Metrics**: https://scikit-learn.org/stable/modules/model_evaluation.html
- **PyTorch LSTM**: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- **Huggingface Transformers**: https://huggingface.co/docs/transformers/
